from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int16]
StringArray = npt.NDArray[np.str_]


class EmbeddingRepositoryError(RuntimeError):
    """Raised when an embedding cache cannot be read or validated."""


@dataclass(frozen=True)
class IdentityEmbeddingCache:
    """
    In-memory representation of one identity-level embedding cache.
    """

    identity_id: str
    image_ids: StringArray
    image_paths: StringArray
    relative_paths: StringArray
    experiment_groups: StringArray
    sample_roles: StringArray
    enrollment_candidate_ranks: IntArray
    embeddings: FloatArray
    face_confidences: FloatArray

    @property
    def image_count(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def embedding_dimension(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def experiment_group(self) -> str:
        unique_groups = np.unique(
            self.experiment_groups.astype(str)
        )

        if len(unique_groups) != 1:
            raise EmbeddingRepositoryError(
                f"Identity {self.identity_id} has multiple "
                f"experiment groups: {unique_groups.tolist()}"
            )

        return str(unique_groups[0])

    def enrollment_mask(
        self,
        enrollment_count: int,
    ) -> npt.NDArray[np.bool_]:
        """
        Return a mask selecting candidate ranks 1..enrollment_count.
        """

        if enrollment_count <= 0:
            raise EmbeddingRepositoryError(
                "enrollment_count must be positive"
            )

        return (
            (self.enrollment_candidate_ranks >= 1)
            & (
                self.enrollment_candidate_ranks
                <= enrollment_count
            )
        )

    def enrollment_embeddings(
        self,
        enrollment_count: int,
    ) -> FloatArray:
        mask = self.enrollment_mask(
            enrollment_count=enrollment_count
        )

        selected = self.embeddings[mask]

        if len(selected) != enrollment_count:
            available_ranks = sorted(
                self.enrollment_candidate_ranks[
                    self.enrollment_candidate_ranks > 0
                ]
                .astype(int)
                .tolist()
            )

            raise EmbeddingRepositoryError(
                f"Identity {self.identity_id} does not contain "
                f"exactly {enrollment_count} enrollment embeddings. "
                f"Selected={len(selected)}, "
                f"available ranks={available_ranks}"
            )

        return selected.astype(
            np.float32,
            copy=False,
        )

    def enrollment_ranks(
        self,
        enrollment_count: int,
    ) -> IntArray:
        mask = self.enrollment_mask(
            enrollment_count=enrollment_count
        )

        ranks = self.enrollment_candidate_ranks[
            mask
        ]

        order = np.argsort(ranks)

        return ranks[order].astype(
            np.int16,
            copy=False,
        )


class EmbeddingRepository:
    """
    Read-only repository for identity-level NPZ embedding caches.

    Expected structure:

        cache_root/
        ├── identities/
        │   ├── n000001.npz
        │   └── ...
        ├── status/
        └── failures/
    """

    REQUIRED_ARRAYS = {
        "identity_id",
        "image_ids",
        "image_paths",
        "relative_paths",
        "experiment_groups",
        "sample_roles",
        "enrollment_candidate_ranks",
        "embeddings",
        "face_confidences",
    }

    def __init__(
        self,
        cache_root: str | Path,
        *,
        expected_embedding_dimension: int = 512,
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

        self.expected_embedding_dimension = (
            expected_embedding_dimension
        )

        if not self.identity_dir.is_dir():
            raise EmbeddingRepositoryError(
                f"Identity cache directory does not exist: "
                f"{self.identity_dir}"
            )

    def identity_paths(self) -> list[Path]:
        return sorted(
            self.identity_dir.glob("*.npz")
        )

    def identity_ids(self) -> list[str]:
        return [
            path.stem
            for path in self.identity_paths()
        ]

    def __len__(self) -> int:
        return len(self.identity_paths())

    def read_status(
        self,
        identity_id: str,
    ) -> dict | None:
        path = (
            self.status_dir
            / f"{identity_id}.json"
        )

        if not path.is_file():
            return None

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                loaded = json.load(file)
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

        return loaded if isinstance(loaded, dict) else None

    def load(
        self,
        identity_id: str,
    ) -> IdentityEmbeddingCache:
        cache_path = (
            self.identity_dir
            / f"{identity_id}.npz"
        )

        if not cache_path.is_file():
            raise EmbeddingRepositoryError(
                f"Embedding cache does not exist: "
                f"{cache_path}"
            )

        try:
            with np.load(
                cache_path,
                allow_pickle=False,
            ) as data:
                missing_arrays = (
                    self.REQUIRED_ARRAYS
                    - set(data.files)
                )

                if missing_arrays:
                    raise EmbeddingRepositoryError(
                        f"Cache {cache_path} is missing arrays: "
                        f"{sorted(missing_arrays)}"
                    )

                cached_identity_ids = (
                    data["identity_id"]
                    .astype(str)
                    .tolist()
                )

                if len(cached_identity_ids) != 1:
                    raise EmbeddingRepositoryError(
                        f"Cache contains invalid identity_id array: "
                        f"{cached_identity_ids}"
                    )

                cached_identity_id = str(
                    cached_identity_ids[0]
                )

                if cached_identity_id != identity_id:
                    raise EmbeddingRepositoryError(
                        f"Identity mismatch: "
                        f"filename={identity_id}, "
                        f"cache={cached_identity_id}"
                    )

                cache = IdentityEmbeddingCache(
                    identity_id=cached_identity_id,
                    image_ids=data[
                        "image_ids"
                    ].astype(np.str_),
                    image_paths=data[
                        "image_paths"
                    ].astype(np.str_),
                    relative_paths=data[
                        "relative_paths"
                    ].astype(np.str_),
                    experiment_groups=data[
                        "experiment_groups"
                    ].astype(np.str_),
                    sample_roles=data[
                        "sample_roles"
                    ].astype(np.str_),
                    enrollment_candidate_ranks=data[
                        "enrollment_candidate_ranks"
                    ].astype(np.int16),
                    embeddings=data[
                        "embeddings"
                    ].astype(np.float32),
                    face_confidences=data[
                        "face_confidences"
                    ].astype(np.float32),
                )

        except EmbeddingRepositoryError:
            raise
        except Exception as exc:
            raise EmbeddingRepositoryError(
                f"Failed to load cache: {cache_path}"
            ) from exc

        self._validate_cache(cache)

        return cache

    def _validate_cache(
        self,
        cache: IdentityEmbeddingCache,
    ) -> None:
        if cache.embeddings.ndim != 2:
            raise EmbeddingRepositoryError(
                f"{cache.identity_id}: embeddings must have "
                f"shape [N, D], got {cache.embeddings.shape}"
            )

        if (
            cache.embedding_dimension
            != self.expected_embedding_dimension
        ):
            raise EmbeddingRepositoryError(
                f"{cache.identity_id}: unexpected embedding "
                f"dimension. Expected "
                f"{self.expected_embedding_dimension}, "
                f"got {cache.embedding_dimension}"
            )

        row_count = cache.image_count

        arrays = {
            "image_ids": cache.image_ids,
            "image_paths": cache.image_paths,
            "relative_paths": cache.relative_paths,
            "experiment_groups": (
                cache.experiment_groups
            ),
            "sample_roles": cache.sample_roles,
            "enrollment_candidate_ranks": (
                cache.enrollment_candidate_ranks
            ),
            "face_confidences": (
                cache.face_confidences
            ),
        }

        invalid_lengths = {
            name: len(array)
            for name, array in arrays.items()
            if len(array) != row_count
        }

        if invalid_lengths:
            raise EmbeddingRepositoryError(
                f"{cache.identity_id}: metadata row-count "
                f"mismatch. Expected={row_count}, "
                f"actual={invalid_lengths}"
            )

        if not np.all(
            np.isfinite(cache.embeddings)
        ):
            raise EmbeddingRepositoryError(
                f"{cache.identity_id}: embeddings contain "
                "NaN or infinity"
            )

        if len(np.unique(cache.image_ids)) != row_count:
            raise EmbeddingRepositoryError(
                f"{cache.identity_id}: duplicate image IDs "
                "inside cache"
            )

        positive_ranks = sorted(
            cache.enrollment_candidate_ranks[
                cache.enrollment_candidate_ranks > 0
            ]
            .astype(int)
            .tolist()
        )

        expected_ranks = list(
            range(1, 11)
        )

        if positive_ranks != expected_ranks:
            raise EmbeddingRepositoryError(
                f"{cache.identity_id}: invalid enrollment "
                f"candidate ranks. Expected={expected_ranks}, "
                f"actual={positive_ranks}"
            )

        # Accessing this property also validates group uniqueness.
        _ = cache.experiment_group

    def iter_caches(
        self,
        *,
        experiment_group: str | None = None,
    ) -> Iterator[IdentityEmbeddingCache]:
        for identity_id in self.identity_ids():
            cache = self.load(
                identity_id=identity_id
            )

            if (
                experiment_group is not None
                and cache.experiment_group
                != experiment_group
            ):
                continue

            yield cache