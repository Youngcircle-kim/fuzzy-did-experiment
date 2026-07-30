from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from fuzzy_did.data import (
    EmbeddingRepository,
)


FloatArray = npt.NDArray[np.float32]

AggregationMethod = Literal[
    "mean",
    "median",
]


class EnrollmentTemplateError(RuntimeError):
    """Raised when enrollment templates cannot be generated."""


@dataclass(frozen=True)
class EnrollmentTemplateSet:
    enrollment_count: int
    aggregation: AggregationMethod
    l2_normalized: bool
    identity_ids: npt.NDArray[np.str_]
    experiment_groups: npt.NDArray[np.str_]
    representative_embeddings: FloatArray
    enrollment_counts: npt.NDArray[np.int16]
    candidate_ranks: npt.NDArray[np.int16]

    @property
    def identity_count(self) -> int:
        return int(
            self.representative_embeddings.shape[0]
        )

    @property
    def embedding_dimension(self) -> int:
        return int(
            self.representative_embeddings.shape[1]
        )


def aggregate_embeddings(
    embeddings: FloatArray,
    *,
    method: AggregationMethod,
    l2_normalize: bool,
) -> FloatArray:
    if embeddings.ndim != 2:
        raise EnrollmentTemplateError(
            f"embeddings must have shape [N, D], "
            f"got {embeddings.shape}"
        )

    if len(embeddings) == 0:
        raise EnrollmentTemplateError(
            "cannot aggregate an empty embedding set"
        )

    if method == "mean":
        representative = np.mean(
            embeddings,
            axis=0,
            dtype=np.float64,
        )
    elif method == "median":
        representative = np.median(
            embeddings,
            axis=0,
        )
    else:
        raise EnrollmentTemplateError(
            f"unsupported aggregation method: "
            f"{method}"
        )

    representative = np.asarray(
        representative,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(representative)
    ):
        raise EnrollmentTemplateError(
            "representative embedding contains "
            "NaN or infinity"
        )

    if l2_normalize:
        norm = float(
            np.linalg.norm(representative)
        )

        if norm == 0.0:
            raise EnrollmentTemplateError(
                "representative embedding has zero norm"
            )

        representative = (
            representative / norm
        ).astype(
            np.float32,
            copy=False,
        )

    return representative


def build_enrollment_template_set(
    repository: EmbeddingRepository,
    *,
    enrollment_count: int,
    aggregation: AggregationMethod,
    l2_normalize: bool,
) -> EnrollmentTemplateSet:
    if enrollment_count <= 0:
        raise EnrollmentTemplateError(
            "enrollment_count must be positive"
        )

    if enrollment_count > 10:
        raise EnrollmentTemplateError(
            "enrollment_count cannot exceed the "
            "configured candidate count of 10"
        )

    identity_ids: list[str] = []
    experiment_groups: list[str] = []
    representatives: list[FloatArray] = []
    enrollment_counts: list[int] = []
    candidate_rank_rows: list[list[int]] = []

    for cache in repository.iter_caches():
        enrollment_mask = cache.enrollment_mask(
            enrollment_count=enrollment_count
        )

        selected_embeddings = (
            cache.embeddings[
                enrollment_mask
            ]
        )

        selected_ranks = (
            cache.enrollment_candidate_ranks[
                enrollment_mask
            ]
            .astype(np.int16)
        )

        if (
            len(selected_embeddings)
            != enrollment_count
        ):
            raise EnrollmentTemplateError(
                f"{cache.identity_id}: expected "
                f"{enrollment_count} enrollment "
                f"embeddings, got "
                f"{len(selected_embeddings)}"
            )

        rank_order = np.argsort(
            selected_ranks
        )

        selected_embeddings = (
            selected_embeddings[
                rank_order
            ]
        )

        selected_ranks = (
            selected_ranks[
                rank_order
            ]
        )

        expected_ranks = np.arange(
            1,
            enrollment_count + 1,
            dtype=np.int16,
        )

        if not np.array_equal(
            selected_ranks,
            expected_ranks,
        ):
            raise EnrollmentTemplateError(
                f"{cache.identity_id}: invalid ranks "
                f"for enrollment_count="
                f"{enrollment_count}. "
                f"Expected={expected_ranks.tolist()}, "
                f"actual={selected_ranks.tolist()}"
            )

        representative = aggregate_embeddings(
            selected_embeddings,
            method=aggregation,
            l2_normalize=l2_normalize,
        )

        identity_ids.append(
            cache.identity_id
        )
        experiment_groups.append(
            cache.experiment_group
        )
        representatives.append(
            representative
        )
        enrollment_counts.append(
            enrollment_count
        )
        candidate_rank_rows.append(
            selected_ranks.astype(int).tolist()
        )

    if not representatives:
        raise EnrollmentTemplateError(
            "repository did not contain any identities"
        )

    representative_matrix = np.stack(
        representatives,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.all(
        np.isfinite(representative_matrix)
    ):
        raise EnrollmentTemplateError(
            "template matrix contains NaN or infinity"
        )

    return EnrollmentTemplateSet(
        enrollment_count=enrollment_count,
        aggregation=aggregation,
        l2_normalized=l2_normalize,
        identity_ids=np.asarray(
            identity_ids,
            dtype=np.str_,
        ),
        experiment_groups=np.asarray(
            experiment_groups,
            dtype=np.str_,
        ),
        representative_embeddings=(
            representative_matrix
        ),
        enrollment_counts=np.asarray(
            enrollment_counts,
            dtype=np.int16,
        ),
        candidate_ranks=np.asarray(
            candidate_rank_rows,
            dtype=np.int16,
        ),
    )