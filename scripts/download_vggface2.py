from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


LOGGER = logging.getLogger("download_vggface2")


class DatasetDownloadError(RuntimeError):
    """Raised when dataset download, extraction, or validation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare VGGFace2 from a Kaggle mirror."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/vggface2.yaml"),
        help="Path to the VGGFace2 YAML configuration.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force Kaggle CLI to download the dataset again.",
    )

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Remove the existing raw directory and extract again.",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip image and identity count validation.",
    )

    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 calculation for downloaded archive files.",
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
        raise DatasetDownloadError(
            f"configuration file does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise DatasetDownloadError(
            f"failed to read configuration file: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise DatasetDownloadError(
            f"invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise DatasetDownloadError(
            f"configuration root must be a mapping: {resolved}"
        )

    return loaded


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def directory_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False

    return any(item.is_file() for item in path.iterdir())


def check_disk_space(
    target_dir: Path,
    required_gib: int,
) -> None:
    existing_parent = target_dir

    while not existing_parent.exists():
        parent = existing_parent.parent

        if parent == existing_parent:
            raise DatasetDownloadError(
                f"could not find an existing parent directory for: {target_dir}"
            )

        existing_parent = parent

    usage = shutil.disk_usage(existing_parent)
    free_gib = usage.free / (1024**3)

    LOGGER.info("free disk space: %.2f GiB", free_gib)

    if free_gib < required_gib:
        raise DatasetDownloadError(
            f"insufficient disk space: {free_gib:.2f} GiB available, "
            f"{required_gib} GiB required"
        )


def check_kaggle_cli() -> str:
    kaggle_executable = shutil.which("kaggle")

    if kaggle_executable is None:
        raise DatasetDownloadError(
            "Kaggle CLI was not found. Install it with:\n"
            "  pip install kaggle"
        )

    try:
        result = subprocess.run(
            [kaggle_executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DatasetDownloadError(
            "Kaggle CLI exists but could not be executed."
        ) from exc

    version_output = result.stdout.strip() or result.stderr.strip()
    LOGGER.info("Kaggle CLI: %s", version_output)

    return kaggle_executable


def check_kaggle_authentication(
    kaggle_executable: str,
) -> None:
    """
    Verify that the CLI can access the Kaggle API.

    This does not download the dataset. It sends a small dataset-list request.
    """

    try:
        subprocess.run(
            [
                kaggle_executable,
                "datasets",
                "list",
                "-s",
                "vggface2",
                "--max-size",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""

        raise DatasetDownloadError(
            "Kaggle authentication failed.\n"
            "Place the API token at ~/.kaggle/access_token or run:\n"
            "  kaggle auth login\n"
            f"Kaggle error: {stderr}"
        ) from exc


def download_from_kaggle(
    kaggle_executable: str,
    dataset_handle: str,
    download_dir: Path,
    force_download: bool,
) -> Path:
    ensure_directory(download_dir)

    if directory_has_files(download_dir) and not force_download:
        LOGGER.info(
            "download directory already contains files; reusing: %s",
            download_dir,
        )
        return download_dir

    command = [
        kaggle_executable,
        "datasets",
        "download",
        "-d",
        dataset_handle,
        "-p",
        str(download_dir),
    ]

    if force_download:
        command.append("--force")

    LOGGER.info("downloading Kaggle dataset: %s", dataset_handle)
    LOGGER.debug("command: %s", " ".join(command))

    try:
        subprocess.run(
            command,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DatasetDownloadError(
            "Kaggle dataset download failed. Check the dataset handle, "
            "authentication, network connection, and dataset access terms."
        ) from exc

    if not directory_has_files(download_dir):
        raise DatasetDownloadError(
            f"download finished but no files were found: {download_dir}"
        )

    return download_dir


def is_supported_archive(path: Path) -> bool:
    name = path.name.lower()

    return name.endswith(
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


def validate_archive_member_path(
    destination: Path,
    member_name: str,
) -> None:
    destination_root = destination.resolve()
    member_path = (destination / member_name).resolve()

    if not member_path.is_relative_to(destination_root):
        raise DatasetDownloadError(
            f"unsafe archive path detected: {member_name}"
        )


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
) -> None:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            validate_archive_member_path(
                destination=destination,
                member_name=member.filename,
            )

        archive.extractall(destination)


def safe_extract_tar(
    archive_path: Path,
    destination: Path,
) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            validate_archive_member_path(
                destination=destination,
                member_name=member.name,
            )

            # Reject links because they can point outside the extraction root.
            if member.issym() or member.islnk():
                raise DatasetDownloadError(
                    f"symbolic or hard link is not allowed in archive: "
                    f"{member.name}"
                )

        archive.extractall(destination)


def extract_archive(
    archive_path: Path,
    destination: Path,
) -> None:
    LOGGER.info("extracting: %s", archive_path.name)

    try:
        if zipfile.is_zipfile(archive_path):
            safe_extract_zip(archive_path, destination)
            return

        if tarfile.is_tarfile(archive_path):
            safe_extract_tar(archive_path, destination)
            return
    except (
        OSError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as exc:
        raise DatasetDownloadError(
            f"failed to extract archive: {archive_path}"
        ) from exc

    raise DatasetDownloadError(
        f"unsupported or corrupted archive: {archive_path}"
    )


def copy_unpacked_content(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    """
    Copy unpacked Kaggle content when no archive exists.

    This path is uncommon when using the Kaggle CLI, but is retained for
    compatibility with datasets distributed as individual files.
    """

    ensure_directory(destination_dir)

    for source in source_dir.iterdir():
        destination = destination_dir / source.name

        if destination.exists():
            LOGGER.info("skip existing path: %s", destination)
            continue

        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def prepare_raw_dataset(
    download_dir: Path,
    raw_dir: Path,
    extraction_enabled: bool,
    force_extract: bool,
) -> list[Path]:
    if raw_dir.exists() and directory_has_files(raw_dir) and not force_extract:
        LOGGER.info(
            "raw directory already contains files; reusing: %s",
            raw_dir,
        )
        return []

    if force_extract and raw_dir.exists():
        LOGGER.warning("removing raw directory: %s", raw_dir)
        shutil.rmtree(raw_dir)

    ensure_directory(raw_dir)

    archives = find_archives(download_dir)

    if archives:
        if not extraction_enabled:
            LOGGER.info("archive extraction is disabled")
            return archives

        LOGGER.info("found %d archive(s)", len(archives))

        for archive in archives:
            extract_archive(
                archive_path=archive,
                destination=raw_dir,
            )

        return archives

    LOGGER.info(
        "no supported archive found; copying unpacked content"
    )

    copy_unpacked_content(
        source_dir=download_dir,
        destination_dir=raw_dir,
    )

    return []


def normalize_extensions(
    extensions: list[str],
) -> set[str]:
    return {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }


def iter_images(
    root: Path,
    extensions: set[str],
) -> Iterator[Path]:
    """
    Yield image paths one at a time.

    This avoids loading millions of Path objects into memory.
    """

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def infer_identity_id(
    image_path: Path,
    dataset_root: Path,
) -> str | None:
    """
    Infer identity from common VGGFace2 layouts.

    Examples:
        root/train/n000001/image.jpg
        root/test/n000001/image.jpg
        root/n000001/image.jpg
    """

    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        return None

    parent_parts = relative.parent.parts

    if not parent_parts:
        return None

    for part in reversed(parent_parts):
        if part.startswith("n") and part[1:].isdigit():
            return part

    return relative.parent.name or None


def infer_subset(
    image_path: Path,
    dataset_root: Path,
) -> str:
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        return "unknown"

    parts = {part.lower() for part in relative.parts}

    if "train" in parts:
        return "train"

    if "test" in parts:
        return "test"

    return "unknown"


def validate_dataset(
    raw_dir: Path,
    image_extensions: list[str],
    minimum_image_count: int,
    minimum_identity_count: int,
) -> dict[str, Any]:
    extensions = normalize_extensions(image_extensions)

    LOGGER.info("scanning images under: %s", raw_dir)

    identity_counter: Counter[str] = Counter()
    subset_counter: Counter[str] = Counter()
    extension_counter: Counter[str] = Counter()

    image_count = 0

    for image_path in iter_images(raw_dir, extensions):
        image_count += 1

        identity_id = infer_identity_id(
            image_path=image_path,
            dataset_root=raw_dir,
        )

        if identity_id is not None:
            identity_counter[identity_id] += 1

        subset = infer_subset(
            image_path=image_path,
            dataset_root=raw_dir,
        )
        subset_counter[subset] += 1
        extension_counter[image_path.suffix.lower()] += 1

        if image_count % 100_000 == 0:
            LOGGER.info(
                "validation progress: %,d images scanned",
                image_count,
            )

    identity_count = len(identity_counter)

    LOGGER.info("detected images: %,d", image_count)
    LOGGER.info("detected identities: %,d", identity_count)
    LOGGER.info("subset counts: %s", dict(subset_counter))

    if image_count < minimum_image_count:
        raise DatasetDownloadError(
            "dataset validation failed: expected at least "
            f"{minimum_image_count:,} images, found {image_count:,}"
        )

    if identity_count < minimum_identity_count:
        raise DatasetDownloadError(
            "dataset validation failed: expected at least "
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
            image_count / identity_count
            if identity_count > 0
            else 0.0
        ),
        "subset_image_counts": dict(subset_counter),
        "extension_counts": dict(extension_counter),
        "image_extensions": sorted(extensions),
    }


def calculate_sha256(
    path: Path,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def build_download_manifest(
    download_dir: Path,
    calculate_checksums: bool,
) -> list[dict[str, Any]]:
    """
    Build a manifest only for files in the download directory.

    Raw extracted images are intentionally excluded because hashing millions
    of image files would be unnecessarily expensive.
    """

    manifest: list[dict[str, Any]] = []

    for path in sorted(download_dir.rglob("*")):
        if not path.is_file():
            continue

        file_info: dict[str, Any] = {
            "relative_path": str(path.relative_to(download_dir)),
            "size_bytes": path.stat().st_size,
        }

        if calculate_checksums:
            LOGGER.info("calculating SHA-256: %s", path.name)
            file_info["sha256"] = calculate_sha256(path)

        manifest.append(file_info)

    return manifest


def save_manifest(
    output_path: Path,
    dataset_handle: str,
    download_dir: Path,
    raw_dir: Path,
    validation_result: dict[str, Any] | None,
    calculate_checksums: bool,
) -> None:
    manifest = {
        "dataset_name": "vggface2",
        "provider": "kaggle-cli",
        "dataset_handle": dataset_handle,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_dir": str(download_dir.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "downloaded_files": build_download_manifest(
            download_dir=download_dir,
            calculate_checksums=calculate_checksums,
        ),
        "validation": validation_result,
    }

    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

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

        provider = str(source["provider"])

        if provider not in {"kaggle", "kaggle-cli"}:
            raise DatasetDownloadError(
                f"unsupported dataset provider: {provider}"
            )

        dataset_handle = str(source["dataset_handle"])

        download_dir = (
            Path(paths["download_dir"])
            .expanduser()
            .resolve()
        )

        raw_dir = (
            Path(paths["raw_dir"])
            .expanduser()
            .resolve()
        )

        required_disk_gib = int(
            dataset_config.get("required_disk_gib", 150)
        )

        check_disk_space(
            target_dir=download_dir,
            required_gib=required_disk_gib,
        )

        kaggle_executable = check_kaggle_cli()

        check_kaggle_authentication(
            kaggle_executable=kaggle_executable,
        )

        download_from_kaggle(
            kaggle_executable=kaggle_executable,
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

        if (
            bool(validation["enabled"])
            and not args.skip_validation
        ):
            validation_result = validate_dataset(
                raw_dir=raw_dir,
                image_extensions=list(
                    validation["image_extensions"]
                ),
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
            calculate_checksums=not args.skip_checksum,
        )

        remove_archives = bool(
            extraction.get(
                "remove_archive_after_extract",
                False,
            )
        )

        if remove_archives and extracted_archives:
            for archive in extracted_archives:
                LOGGER.warning(
                    "removing downloaded archive: %s",
                    archive,
                )
                archive.unlink()

        LOGGER.info("VGGFace2 preparation completed")
        LOGGER.info("raw dataset path: %s", raw_dir)

        return 0

    except KeyError as exc:
        LOGGER.error(
            "missing required configuration key: %s",
            exc,
        )
        return 2

    except DatasetDownloadError as exc:
        LOGGER.error("%s", exc)
        return 1

    except KeyboardInterrupt:
        LOGGER.warning("operation interrupted by user")
        return 130

    except Exception:
        LOGGER.exception("unexpected error")
        return 99


if __name__ == "__main__":
    sys.exit(main())