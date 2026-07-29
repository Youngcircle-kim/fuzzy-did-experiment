from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class IdentityCachePaths:
    cache_path: Path
    status_path: Path
    failure_path: Path


class EmbeddingCacheError(RuntimeError):
    """Raised when embedding cache operations fail."""


class EmbeddingCache:
    """Identity-level NPZ embedding cache."""

    def __init__(
        self,
        cache_root: str | Path,
    ) -> None:
        self.cache_root = (
            Path(cache_root)
            .expanduser()
            .resolve()
        )

        self.identity_dir = (
            self.cache_root / "identities"
        )
        self.status_dir = (
            self.cache_root / "status"
        )
        self.failure_dir = (
            self.cache_root / "failures"
        )
        self.manifest_dir = (
            self.cache_root / "manifests"
        )

        for directory in (
            self.identity_dir,
            self.status_dir,
            self.failure_dir,
            self.manifest_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def _safe_identity_id(
        identity_id: str,
    ) -> str:
        return (
            identity_id
            .replace("/", "_")
            .replace("\\", "_")
        )

    def paths_for(
        self,
        identity_id: str,
    ) -> IdentityCachePaths:
        safe_identity_id = self._safe_identity_id(
            identity_id
        )

        return IdentityCachePaths(
            cache_path=(
                self.identity_dir
                / f"{safe_identity_id}.npz"
            ),
            status_path=(
                self.status_dir
                / f"{safe_identity_id}.json"
            ),
            failure_path=(
                self.failure_dir
                / f"{safe_identity_id}.jsonl"
            ),
        )

    def read_status(
        self,
        identity_id: str,
    ) -> dict[str, Any] | None:
        path = self.paths_for(
            identity_id
        ).status_path

        if not path.is_file():
            return None

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                loaded = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(loaded, dict):
            return None

        return loaded

    def is_complete(
        self,
        identity_id: str,
        expected_image_ids: list[str],
        expected_dimension: int,
    ) -> bool:
        paths = self.paths_for(identity_id)
        status = self.read_status(identity_id)

        if status is None:
            return False

        if status.get("status") != "complete":
            return False

        if not paths.cache_path.is_file():
            return False

        try:
            with np.load(
                paths.cache_path,
                allow_pickle=False,
            ) as data:
                cached_image_ids = (
                    data["image_ids"]
                    .astype(str)
                    .tolist()
                )

                embeddings = data[
                    "embeddings"
                ]

        except Exception:
            return False

        if cached_image_ids != expected_image_ids:
            return False

        if embeddings.ndim != 2:
            return False

        if embeddings.shape != (
            len(expected_image_ids),
            expected_dimension,
        ):
            return False

        if not np.all(
            np.isfinite(embeddings)
        ):
            return False

        return True

    @staticmethod
    def _atomic_write_json(
        output_path: Path,
        payload: dict[str, Any],
    ) -> None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        descriptor, temporary_name = (
            tempfile.mkstemp(
                prefix=f".{output_path.stem}_",
                suffix=".json",
                dir=output_path.parent,
            )
        )

        os.close(descriptor)
        temporary_path = Path(
            temporary_name
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
                file.flush()
                os.fsync(file.fileno())

            temporary_path.replace(
                output_path
            )

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

    def save_identity(
        self,
        *,
        identity_id: str,
        image_ids: list[str],
        image_paths: list[str],
        relative_paths: list[str],
        experiment_groups: list[str],
        sample_roles: list[str],
        candidate_ranks: list[int],
        embeddings: FloatArray,
        face_confidences: FloatArray,
    ) -> Path:
        paths = self.paths_for(identity_id)

        row_count = len(image_ids)

        lengths = {
            "image_paths": len(image_paths),
            "relative_paths": len(
                relative_paths
            ),
            "experiment_groups": len(
                experiment_groups
            ),
            "sample_roles": len(
                sample_roles
            ),
            "candidate_ranks": len(
                candidate_ranks
            ),
            "face_confidences": len(
                face_confidences
            ),
        }

        invalid_lengths = {
            key: value
            for key, value in lengths.items()
            if value != row_count
        }

        if invalid_lengths:
            raise EmbeddingCacheError(
                "Cache metadata length mismatch: "
                f"expected={row_count}, "
                f"actual={invalid_lengths}"
            )

        if embeddings.ndim != 2:
            raise EmbeddingCacheError(
                "Embeddings must have shape [N, D]."
            )

        if embeddings.shape[0] != row_count:
            raise EmbeddingCacheError(
                "Embedding row count differs from image count."
            )

        descriptor, temporary_name = (
            tempfile.mkstemp(
                prefix=f".{identity_id}_",
                suffix=".npz",
                dir=self.identity_dir,
            )
        )

        os.close(descriptor)
        temporary_path = Path(
            temporary_name
        )

        try:
            np.savez_compressed(
                temporary_path,
                identity_id=np.asarray(
                    [identity_id],
                    dtype=np.str_,
                ),
                image_ids=np.asarray(
                    image_ids,
                    dtype=np.str_,
                ),
                image_paths=np.asarray(
                    image_paths,
                    dtype=np.str_,
                ),
                relative_paths=np.asarray(
                    relative_paths,
                    dtype=np.str_,
                ),
                experiment_groups=np.asarray(
                    experiment_groups,
                    dtype=np.str_,
                ),
                sample_roles=np.asarray(
                    sample_roles,
                    dtype=np.str_,
                ),
                enrollment_candidate_ranks=(
                    np.asarray(
                        candidate_ranks,
                        dtype=np.int16,
                    )
                ),
                embeddings=embeddings.astype(
                    np.float32,
                    copy=False,
                ),
                face_confidences=(
                    face_confidences.astype(
                        np.float32,
                        copy=False,
                    )
                ),
            )

            temporary_path.replace(
                paths.cache_path
            )

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

        return paths.cache_path

    def save_status(
        self,
        identity_id: str,
        payload: dict[str, Any],
    ) -> None:
        status_payload = {
            "identity_id": identity_id,
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            **payload,
        }

        self._atomic_write_json(
            self.paths_for(
                identity_id
            ).status_path,
            status_payload,
        )

    def save_failures(
        self,
        identity_id: str,
        failures: list[dict[str, Any]],
    ) -> None:
        failure_path = self.paths_for(
            identity_id
        ).failure_path

        if not failures:
            failure_path.unlink(
                missing_ok=True
            )
            return

        descriptor, temporary_name = (
            tempfile.mkstemp(
                prefix=f".{identity_id}_",
                suffix=".jsonl",
                dir=self.failure_dir,
            )
        )

        os.close(descriptor)
        temporary_path = Path(
            temporary_name
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                for failure in failures:
                    file.write(
                        json.dumps(
                            failure,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                file.flush()
                os.fsync(file.fileno())

            temporary_path.replace(
                failure_path
            )

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

    def append_manifest(
        self,
        shard_index: int,
        payload: dict[str, Any],
    ) -> None:
        output_path = (
            self.manifest_dir
            / f"shard_{shard_index:03d}.jsonl"
        )

        record = {
            "recorded_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            **payload,
        }

        with output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()
            os.fsync(file.fileno())