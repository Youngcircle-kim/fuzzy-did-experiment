from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kagglehub
import yaml


LOGGER = logging.getLogger("download_vggface2")


class DatasetDownloadError(RuntimeError):
    """Raised when dataset download or extraction fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare VGGFace2 from a configurable Kaggle mirror."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/vggface2.yaml"),
        help="Path to the VGGFace2 dataset YAML configuration.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download again even when the target download directory is populated.",
    )

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract archives again even when the raw directory is populated.",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-download image and identity validation.",
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
        raise DatasetDownloadError(f"config file does not exist: {resolved}")

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetDownloadError(
            f"failed to read YAML configuration: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise DatasetDownloadError(
            f"configuration root must be a mapping: {resolved}"
        )

    return loaded


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def check_disk_space(target_dir: Path, required_gib: int = 150) -> None:
    """
    Check free disk space before download.

    VGGFace2 mirrors differ in compression and internal layout.
    150 GiB is a conservative working-space requirement covering
    the archive, extraction, and temporary files.
    """

    existing_parent = target_dir

    while not existing_parent.exists():
        existing_parent = existing_parent.parent

    usage = shutil.disk_usage(existing_parent)
    free_gib = usage.free / (1024**3)

    LOGGER.info("free disk space: %.2f GiB", free_gib)

    if free_gib < required_gib:
        raise DatasetDownloadError(
            f"insufficient free disk space: {free_gib:.2f} GiB available, "
            f"at least {required_gib} GiB recommended"
        )


def copy_downloaded_content(
    source_dir: Path,
    destination_dir: Path,
    overwrite: bool,
) -> None:
    ensure_directory(destination_dir)

    for source in source_dir.iterdir():
        destination = destination_dir / source.name

        if destination.exists():
            if not overwrite:
                LOGGER.info("skip existing path: %s", destination)
                continue

            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        if source.is_dir():
            LOGGER.info("copy directory: %s", source.name)
            shutil.copytree(source, destination)
        else:
            LOGGER.info("copy file: %s", source.name)
            shutil.copy2(source, destination)


def download_from_kaggle(
    dataset_handle: str,
    download_dir: Path,
    force_download: bool,
) -> Path:
    if directory_has_files(download_dir) and not force_download:
        LOGGER.info(
            "download directory is already populated; reusing: %s",
            download_dir,
        )
        return download_dir

    if force_download and download_dir.exists():
        LOGGER.warning("removing previous download directory: %s", download_dir)
        shutil.rmtree(download_dir)

    ensure_directory(download_dir)

    LOGGER.info("downloading Kaggle dataset: %s", dataset_handle)

    try:
        cached_path = Path(
            kagglehub.dataset_download(
                dataset_handle,
                force_download=force_download,
            )
        ).resolve()
    except Exception as exc:
        raise DatasetDownloadError(
            "Kaggle download failed. Verify the dataset handle, Kaggle "
            "credentials, network access, and dataset terms."
        ) from exc

    if not cached_path.exists():
        raise DatasetDownloadError(
            f"KaggleHub returned a nonexistent path: {cached_path}"
        )

    LOGGER.info("KaggleHub cache path: %s", cached_path)

    # KaggleHub stores datasets in its own cache. Copy the files to the
    # experiment-controlled data/downloads directory for reproducibility.
    copy_downloaded_content(
        source_dir=cached_path,
        destination_dir=download_dir,
        overwrite=force_download,
    )

    return download_dir


def is_supported_archive(path: Path) -> bool:
    lower_name = path.name.lower()

    return lower_name.endswith(
        (
            ".zip",
            ".tar",
            ".tar.gz",
            ".tgz",
            ".tar.bz2",
            ".tbz2",
            ".tar.xz",
            ".txz",
        )
    )


def find_archives(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and is_supported_archive(path)
    )


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        destination_root = destination.resolve()

        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()

            if not member_path.is_relative_to(destination_root):
                raise DatasetDownloadError(
                    f"unsafe path found in ZIP archive: {member.filename}"
                )

        archive.extractall(destination)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        destination_root = destination.resolve()

        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()

            if not member_path.is_relative_to(destination_root):
                raise DatasetDownloadError(
                    f"unsafe path found in TAR archive: {member.name}"
                )

        archive.extractall(destination)


def extract_archive(archive_path: Path, destination: Path) -> None:
    LOGGER.info("extracting archive: %s", archive_path)

    if zipfile.is_zipfile(archive_path):
        safe_extract_zip(archive_path, destination)
        return

    if tarfile.is_tarfile(archive_path):
        safe_extract_tar(archive_path, destination)
        return

    raise DatasetDownloadError(
        f"unsupported or corrupted archive: {archive_path}"
    )


def prepare_raw_dataset(
    download_dir: Path,
    raw_dir: Path,
    extraction_enabled: bool,
    force_extract: bool,
) -> list[Path]:
    if directory_has_files(raw_dir) and not force_extract:
        LOGGER.info("raw directory is already populated; reusing: %s", raw_dir)
        return []

    if force_extract and raw_dir.exists():
        LOGGER.warning("removing previous raw directory: %s", raw_dir)
        shutil.rmtree(raw_dir)

    ensure_directory(raw_dir)

    archives = find_archives(download_dir)

    if archives and extraction_enabled:
        for archive in archives:
            extract_archive(archive, raw_dir)

        return archives

    # Some Kaggle datasets are already unpacked by KaggleHub.
    LOGGER.info(
        "no archive extraction required; copying unpacked dataset files"
    )

    copy_downloaded_content(
        source_dir=download_dir,
        destination_dir=raw_dir,
        overwrite=force_extract,
    )

    return []


def find_images(
    root: Path,
    extensions: set[str],
) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]


def infer_identity_id(image_path: Path, dataset_root: Path) -> str | None:
    """
    Infer identity from directory layout.

    Typical VGGFace2 layouts include:
        root/train/n000001/image.jpg
        root/test/n000001/image.jpg
        root/n000001/image.jpg
    """

    relative = image_path.relative_to(dataset_root)
    parent_parts = relative.parent.parts

    if not parent_parts:
        return None

    for part in reversed(parent_parts):
        if part.startswith("n") and part[1:].isdigit():
            return part

    # Fallback: immediate parent directory.
    return relative.parent.name


def validate_dataset(
    raw_dir: Path,
    image_extensions: list[str],
    minimum_image_count: int,
    minimum_identity_count: int,
) -> dict[str, Any]:
    normalized_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in image_extensions
    }

    LOGGER.info("scanning images under: %s", raw_dir)

    images = find_images(raw_dir, normalized_extensions)

    identity_counter: Counter[str] = Counter()

    for image in images:
        identity_id = infer_identity_id(image, raw_dir)

        if identity_id:
            identity_counter[identity_id] += 1

    image_count = len(images)
    identity_count = len(identity_counter)

    LOGGER.info("detected images: %d", image_count)
    LOGGER.info("detected identities: %d", identity_count)

    if image_count < minimum_image_count:
        raise DatasetDownloadError(
            f"dataset validation failed: expected at least "
            f"{minimum_image_count:,} images, found {image_count:,}"
        )

    if identity_count < minimum_identity_count:
        raise DatasetDownloadError(
            f"dataset validation failed: expected at least "
            f"{minimum_identity_count:,} identities, found "
            f"{identity_count:,}"
        )

    counts = list(identity_counter.values())

    return {
        "image_count": image_count,
        "identity_count": identity_count,
        "minimum_images_per_identity": min(counts) if counts else 0,
        "maximum_images_per_identity": max(counts) if counts else 0,
        "mean_images_per_identity": (
            image_count / identity_count if identity_count else 0.0
        ),
        "image_extensions": sorted(normalized_extensions),
    }


def calculate_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_file_manifest(download_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    for path in sorted(download_dir.rglob("*")):
        if not path.is_file():
            continue

        manifest.append(
            {
                "relative_path": str(path.relative_to(download_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
        )

    return manifest


def save_manifest(
    output_path: Path,
    dataset_handle: str,
    download_dir: Path,
    raw_dir: Path,
    validation_result: dict[str, Any] | None,
) -> None:
    manifest = {
        "dataset_name": "vggface2",
        "provider": "kaggle",
        "dataset_handle": dataset_handle,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_dir": str(download_dir.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "downloaded_files": build_file_manifest(download_dir),
        "validation": validation_result,
    }

    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    LOGGER.info("saved manifest: %s", output_path)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_yaml(args.config)
        dataset_config = config["dataset"]

        source = dataset_config["source"]
        paths = dataset_config["paths"]
        extraction = dataset_config["extraction"]
        validation = dataset_config["validation"]

        provider = source["provider"]

        if provider != "kaggle":
            raise DatasetDownloadError(
                f"unsupported dataset provider: {provider}"
            )

        dataset_handle = str(source["dataset_handle"])
        download_dir = Path(paths["download_dir"]).expanduser().resolve()
        raw_dir = Path(paths["raw_dir"]).expanduser().resolve()

        check_disk_space(download_dir, required_gib=150)

        download_from_kaggle(
            dataset_handle=dataset_handle,
            download_dir=download_dir,
            force_download=args.force_download,
        )

        extracted_archives = prepare_raw_dataset(
            download_dir=download_dir,
            raw_dir=raw_dir,
            extraction_enabled=bool(extraction["enabled"]),
            force_extract=args.force_extract,
        )

        validation_result: dict[str, Any] | None = None

        if bool(validation["enabled"]) and not args.skip_validation:
            validation_result = validate_dataset(
                raw_dir=raw_dir,
                image_extensions=list(validation["image_extensions"]),
                minimum_image_count=int(
                    validation["minimum_image_count"]
                ),
                minimum_identity_count=int(
                    validation["minimum_identity_count"]
                ),
            )

        manifest_path = raw_dir / "download_manifest.json"

        save_manifest(
            output_path=manifest_path,
            dataset_handle=dataset_handle,
            download_dir=download_dir,
            raw_dir=raw_dir,
            validation_result=validation_result,
        )

        if (
            bool(extraction["remove_archive_after_extract"])
            and extracted_archives
        ):
            for archive in extracted_archives:
                LOGGER.warning("removing archive: %s", archive)
                archive.unlink()

        LOGGER.info("VGGFace2 preparation completed")
        LOGGER.info("raw dataset path: %s", raw_dir)

        return 0

    except KeyError as exc:
        LOGGER.error("missing required configuration key: %s", exc)
        return 2
    except DatasetDownloadError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("download interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())