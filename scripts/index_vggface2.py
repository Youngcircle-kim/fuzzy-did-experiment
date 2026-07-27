from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


LOGGER = logging.getLogger("index_vggface2")


class DatasetIndexError(RuntimeError):
    """Raised when VGGFace2 indexing fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a metadata index for a VGGFace2 directory."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/vggface2.yaml"),
        help="Path to the VGGFace2 YAML configuration.",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override the raw dataset directory from the configuration.",
    )

    parser.add_argument(
        "--output-index",
        type=Path,
        default=None,
        help="Override the output Parquet path.",
    )

    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Open every image with Pillow to detect corrupted files.",
    )

    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Calculate SHA-256 for every image.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing index.",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise DatasetIndexError(
            f"configuration file does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise DatasetIndexError(
            f"failed to read configuration file: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise DatasetIndexError(
            f"invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise DatasetIndexError(
            f"configuration root must be a mapping: {resolved}"
        )

    return loaded


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_extensions(extensions: list[str]) -> set[str]:
    normalized: set[str] = set()

    for extension in extensions:
        extension = extension.lower()

        if not extension.startswith("."):
            extension = f".{extension}"

        normalized.add(extension)

    return normalized


def iter_image_paths(
    root: Path,
    extensions: set[str],
) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def infer_identity_id(
    image_path: Path,
    dataset_root: Path,
) -> str:
    """
    Infer identity from the directory hierarchy.

    Preferred VGGFace2 identity format:
        n000001
        n001234

    If no such directory exists, the immediate parent directory is used.
    """

    relative = image_path.relative_to(dataset_root)
    parent_parts = relative.parent.parts

    for part in reversed(parent_parts):
        if part.startswith("n") and part[1:].isdigit():
            return part

    if not parent_parts:
        return "unknown"

    return parent_parts[-1]


def infer_subset(
    image_path: Path,
    dataset_root: Path,
) -> str:
    """
    Infer train/test/validation subset from the path.

    Unknown layouts remain 'unknown' and are not silently mapped to train.
    """

    relative = image_path.relative_to(dataset_root)
    lower_parts = [part.lower() for part in relative.parts]

    train_names = {
        "train",
        "train_data",
        "training",
    }

    test_names = {
        "test",
        "test_data",
        "testing",
    }

    validation_names = {
        "val",
        "valid",
        "validation",
        "eval",
        "evaluation",
    }

    for part in lower_parts:
        if part in train_names:
            return "train"

        if part in test_names:
            return "test"

        if part in validation_names:
            return "validation"

    return "unknown"


def build_image_id(
    image_path: Path,
    dataset_root: Path,
) -> str:
    relative_path = image_path.relative_to(dataset_root)
    path_without_suffix = relative_path.with_suffix("")

    return str(path_without_suffix).replace("/", "__")


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_image(path: Path) -> tuple[bool, str | None]:
    """
    Verify that Pillow can parse the image structure.

    Image.verify() checks file integrity without fully decoding all pixels.
    """

    try:
        with Image.open(path) as image:
            image.verify()

        return True, None

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ) as exc:
        return False, str(exc)


def get_image_metadata(
    image_path: Path,
    dataset_root: Path,
    validate_images: bool,
    checksum: bool,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        stat = image_path.stat()
    except OSError as exc:
        return None, {
            "image_path": str(image_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    if validate_images:
        valid, error_message = validate_image(image_path)

        if not valid:
            return None, {
                "image_path": str(image_path),
                "error_type": "InvalidImage",
                "error_message": error_message or "unknown image error",
            }

    relative_path = image_path.relative_to(dataset_root)

    record: dict[str, Any] = {
        "identity_id": infer_identity_id(
            image_path=image_path,
            dataset_root=dataset_root,
        ),
        "subset": infer_subset(
            image_path=image_path,
            dataset_root=dataset_root,
        ),
        "image_id": build_image_id(
            image_path=image_path,
            dataset_root=dataset_root,
        ),
        "image_path": str(image_path.resolve()),
        "relative_path": str(relative_path),
        "extension": image_path.suffix.lower(),
        "file_size_bytes": stat.st_size,
    }

    if checksum:
        try:
            record["sha256"] = calculate_sha256(image_path)
        except OSError as exc:
            return None, {
                "image_path": str(image_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    return record, None


def save_invalid_files(
    invalid_files: list[dict[str, str]],
    output_path: Path,
) -> None:
    ensure_directory(output_path.parent)

    fieldnames = [
        "image_path",
        "error_type",
        "error_message",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(invalid_files)

    LOGGER.info(
        "saved invalid-file report: %s",
        output_path,
    )


def build_summary(
    dataframe: pd.DataFrame,
    raw_dir: Path,
    invalid_file_count: int,
) -> dict[str, Any]:
    identity_counts = (
        dataframe.groupby("identity_id")
        .size()
        .sort_values()
    )

    subset_counts = (
        dataframe["subset"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    extension_counts = (
        dataframe["extension"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if identity_counts.empty:
        minimum_images = 0
        maximum_images = 0
        mean_images = 0.0
        median_images = 0.0
    else:
        minimum_images = int(identity_counts.min())
        maximum_images = int(identity_counts.max())
        mean_images = float(identity_counts.mean())
        median_images = float(identity_counts.median())

    return {
        "dataset_name": "vggface2",
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir.resolve()),
        "image_count": int(len(dataframe)),
        "identity_count": int(
            dataframe["identity_id"].nunique()
        ),
        "invalid_file_count": invalid_file_count,
        "subset_image_counts": {
            str(key): int(value)
            for key, value in subset_counts.items()
        },
        "extension_counts": {
            str(key): int(value)
            for key, value in extension_counts.items()
        },
        "minimum_images_per_identity": minimum_images,
        "maximum_images_per_identity": maximum_images,
        "mean_images_per_identity": mean_images,
        "median_images_per_identity": median_images,
        "total_file_size_bytes": int(
            dataframe["file_size_bytes"].sum()
        ),
    }


def save_summary(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    ensure_directory(output_path.parent)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info("saved summary: %s", output_path)


def build_index(
    raw_dir: Path,
    output_index: Path,
    output_summary: Path,
    output_invalid_files: Path,
    image_extensions: list[str],
    validate_images: bool,
    checksum: bool,
    overwrite: bool,
) -> None:
    if not raw_dir.is_dir():
        raise DatasetIndexError(
            f"raw dataset directory does not exist: {raw_dir}"
        )

    if output_index.exists() and not overwrite:
        raise DatasetIndexError(
            f"index already exists: {output_index}\n"
            "Use --overwrite to replace it."
        )

    extensions = normalize_extensions(image_extensions)

    LOGGER.info("raw directory: %s", raw_dir)
    LOGGER.info("image extensions: %s", sorted(extensions))
    LOGGER.info("validate images: %s", validate_images)
    LOGGER.info("calculate checksum: %s", checksum)

    image_paths = iter_image_paths(
        root=raw_dir,
        extensions=extensions,
    )

    records: list[dict[str, Any]] = []
    invalid_files: list[dict[str, str]] = []

    for image_path in tqdm(
        image_paths,
        desc="Indexing VGGFace2",
        unit="image",
        dynamic_ncols=True,
    ):
        record, invalid_record = get_image_metadata(
            image_path=image_path,
            dataset_root=raw_dir,
            validate_images=validate_images,
            checksum=checksum,
        )

        if record is not None:
            records.append(record)

        if invalid_record is not None:
            invalid_files.append(invalid_record)

    if not records:
        raise DatasetIndexError(
            f"no valid image files found under: {raw_dir}"
        )

    dataframe = pd.DataFrame.from_records(records)

    column_order = [
        "identity_id",
        "subset",
        "image_id",
        "image_path",
        "relative_path",
        "extension",
        "file_size_bytes",
    ]

    if checksum:
        column_order.append("sha256")

    dataframe = dataframe[column_order]

    dataframe = dataframe.sort_values(
        by=[
            "subset",
            "identity_id",
            "relative_path",
        ],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_image_ids = dataframe["image_id"].duplicated(
        keep=False
    )

    if duplicate_image_ids.any():
        duplicated_count = int(duplicate_image_ids.sum())

        raise DatasetIndexError(
            f"duplicate image IDs detected: {duplicated_count}"
        )

    ensure_directory(output_index.parent)

    dataframe.to_parquet(
        output_index,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    LOGGER.info(
        "saved index: %s",
        output_index,
    )

    if invalid_files:
        save_invalid_files(
            invalid_files=invalid_files,
            output_path=output_invalid_files,
        )
    else:
        LOGGER.info("no invalid files detected")

        if output_invalid_files.exists():
            output_invalid_files.unlink()

    summary = build_summary(
        dataframe=dataframe,
        raw_dir=raw_dir,
        invalid_file_count=len(invalid_files),
    )

    save_summary(
        summary=summary,
        output_path=output_summary,
    )

    LOGGER.info(
        "indexed images: %s",
        f"{summary['image_count']:,}",
    )

    LOGGER.info(
        "indexed identities: %s",
        f"{summary['identity_count']:,}",
    )

    LOGGER.info(
        "subset counts: %s",
        summary["subset_image_counts"],
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_yaml(args.config)
        dataset_config = config["dataset"]
        paths_config = dataset_config["paths"]
        indexing_config = dataset_config["indexing"]

        raw_dir = (
            args.raw_dir
            if args.raw_dir is not None
            else Path(paths_config["raw_dir"])
        ).expanduser().resolve()

        output_index = (
            args.output_index
            if args.output_index is not None
            else Path(indexing_config["output_index"])
        ).expanduser().resolve()

        output_summary = Path(
            indexing_config["output_summary"]
        ).expanduser().resolve()

        output_invalid_files = Path(
            indexing_config["output_invalid_files"]
        ).expanduser().resolve()

        validate_images = bool(
            indexing_config.get(
                "validate_images",
                False,
            )
        )

        if args.validate_images:
            validate_images = True

        checksum = bool(
            indexing_config.get(
                "checksum",
                False,
            )
        )

        if args.checksum:
            checksum = True

        image_extensions = list(
            indexing_config["image_extensions"]
        )

        build_index(
            raw_dir=raw_dir,
            output_index=output_index,
            output_summary=output_summary,
            output_invalid_files=output_invalid_files,
            image_extensions=image_extensions,
            validate_images=validate_images,
            checksum=checksum,
            overwrite=args.overwrite,
        )

        return 0

    except KeyError as exc:
        LOGGER.error(
            "missing required configuration key: %s",
            exc,
        )
        return 2

    except DatasetIndexError as exc:
        LOGGER.error("%s", exc)
        return 1

    except KeyboardInterrupt:
        LOGGER.warning("indexing interrupted by user")
        return 130

    except Exception:
        LOGGER.exception("unexpected indexing error")
        return 99


if __name__ == "__main__":
    sys.exit(main())