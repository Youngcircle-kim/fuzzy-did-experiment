from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fuzzy_did.data import EmbeddingRepository


FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int16]
StringArray = npt.NDArray[np.str_]
BoolArray = npt.NDArray[np.bool_]


class FeatureSelectionError(RuntimeError):
    """Raised when subject-specific feature selection fails."""


@dataclass(frozen=True)
class FeatureSelectionConfig:
    top_k: int = 128
    stability_weight: float = 0.5
    epsilon: float = 1e-6
    mad_scale: float = 1.4826
    use_leave_one_out_background: bool = True

    def validate(
        self,
        embedding_dimension: int,
    ) -> None:
        if self.top_k <= 0:
            raise FeatureSelectionError(
                "top_k must be positive"
            )

        if self.top_k > embedding_dimension:
            raise FeatureSelectionError(
                f"top_k={self.top_k} exceeds embedding "
                f"dimension={embedding_dimension}"
            )

        if not 0.0 <= self.stability_weight <= 1.0:
            raise FeatureSelectionError(
                "stability_weight must be between 0 and 1"
            )

        if self.epsilon <= 0.0:
            raise FeatureSelectionError(
                "epsilon must be positive"
            )

        if self.mad_scale <= 0.0:
            raise FeatureSelectionError(
                "mad_scale must be positive"
            )


@dataclass(frozen=True)
class SubjectFeatureSelection:
    identity_id: str
    experiment_group: str
    subject_center: FloatArray
    subject_dispersion: FloatArray
    reference_center: FloatArray
    reference_dispersion: FloatArray
    stability_raw: FloatArray
    discrimination_raw: FloatArray
    stability_score: FloatArray
    discrimination_score: FloatArray
    combined_score: FloatArray
    selected_dimensions: IntArray
    selected_scores: FloatArray


@dataclass(frozen=True)
class FeatureSelectionSet:
    enrollment_count: int
    top_k: int
    stability_weight: float
    identity_ids: StringArray
    experiment_groups: StringArray

    subject_centers: FloatArray
    subject_dispersions: FloatArray

    reference_centers: FloatArray
    reference_dispersions: FloatArray

    stability_raw_scores: FloatArray
    discrimination_raw_scores: FloatArray

    stability_scores: FloatArray
    discrimination_scores: FloatArray
    combined_scores: FloatArray

    selected_dimensions: IntArray
    selected_scores: FloatArray

    global_background_center: FloatArray
    global_background_dispersion: FloatArray
    background_identity_ids: StringArray

    @property
    def identity_count(self) -> int:
        return int(self.identity_ids.shape[0])

    @property
    def embedding_dimension(self) -> int:
        return int(self.subject_centers.shape[1])


def dimensionwise_median(
    values: FloatArray,
) -> FloatArray:
    if values.ndim != 2:
        raise FeatureSelectionError(
            f"Expected [N, D], got {values.shape}"
        )

    if len(values) == 0:
        raise FeatureSelectionError(
            "Cannot calculate median of an empty matrix"
        )

    result = np.median(
        values,
        axis=0,
    )

    return np.asarray(
        result,
        dtype=np.float32,
    )


def dimensionwise_mad(
    values: FloatArray,
    center: FloatArray | None = None,
) -> FloatArray:
    if values.ndim != 2:
        raise FeatureSelectionError(
            f"Expected [N, D], got {values.shape}"
        )

    if len(values) == 0:
        raise FeatureSelectionError(
            "Cannot calculate MAD of an empty matrix"
        )

    if center is None:
        center = dimensionwise_median(values)

    if center.ndim != 1:
        raise FeatureSelectionError(
            f"Center must be one-dimensional, got {center.shape}"
        )

    if center.shape[0] != values.shape[1]:
        raise FeatureSelectionError(
            "Center dimension does not match value dimension"
        )

    absolute_deviations = np.abs(
        values - center[None, :]
    )

    result = np.median(
        absolute_deviations,
        axis=0,
    )

    return np.asarray(
        result,
        dtype=np.float32,
    )


def percentile_rank(
    values: FloatArray,
) -> FloatArray:
    """
    Convert one-dimensional values into deterministic percentile ranks.

    Higher values receive higher scores. The result lies in [0, 1].
    """

    if values.ndim != 1:
        raise FeatureSelectionError(
            f"percentile_rank requires 1-D input, got {values.shape}"
        )

    if not np.all(np.isfinite(values)):
        raise FeatureSelectionError(
            "percentile_rank input contains NaN or infinity"
        )

    size = len(values)

    if size == 0:
        raise FeatureSelectionError(
            "percentile_rank input is empty"
        )

    if size == 1:
        return np.ones(
            1,
            dtype=np.float32,
        )

    # Stable sorting makes the result reproducible.
    sorted_indices = np.argsort(
        values,
        kind="stable",
    )

    ranks = np.empty(
        size,
        dtype=np.float64,
    )

    ranks[sorted_indices] = np.arange(
        size,
        dtype=np.float64,
    )

    ranks /= float(size - 1)

    return ranks.astype(
        np.float32,
    )


def select_top_k(
    combined_scores: FloatArray,
    top_k: int,
) -> tuple[IntArray, FloatArray]:
    if combined_scores.ndim != 1:
        raise FeatureSelectionError(
            "combined_scores must be one-dimensional"
        )

    dimension_indices = np.arange(
        len(combined_scores),
        dtype=np.int64,
    )

    # Primary key: descending score
    # Secondary key: ascending dimension index
    ordered_indices = np.lexsort(
        (
            dimension_indices,
            -combined_scores.astype(np.float64),
        )
    )

    selected = ordered_indices[
        :top_k
    ]

    return (
        selected.astype(np.int16),
        combined_scores[selected].astype(
            np.float32,
            copy=False,
        ),
    )


def calculate_background_statistics(
    background_embeddings: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if background_embeddings.ndim != 2:
        raise FeatureSelectionError(
            "background_embeddings must have shape [N, D]"
        )

    if len(background_embeddings) < 2:
        raise FeatureSelectionError(
            "At least two background identities are required"
        )

    center = dimensionwise_median(
        background_embeddings
    )

    dispersion = dimensionwise_mad(
        background_embeddings,
        center=center,
    )

    return center, dispersion


def build_background_reference(
    *,
    identity_id: str,
    experiment_group: str,
    background_identity_ids: list[str],
    background_embeddings: FloatArray,
    global_center: FloatArray,
    global_dispersion: FloatArray,
    use_leave_one_out: bool,
) -> tuple[FloatArray, FloatArray]:
    if (
        experiment_group != "background"
        or not use_leave_one_out
    ):
        return (
            global_center.copy(),
            global_dispersion.copy(),
        )

    try:
        subject_index = background_identity_ids.index(
            identity_id
        )
    except ValueError as exc:
        raise FeatureSelectionError(
            f"Background identity {identity_id} is missing "
            "from the background reference set"
        ) from exc

    leave_one_out = np.delete(
        background_embeddings,
        subject_index,
        axis=0,
    )

    return calculate_background_statistics(
        leave_one_out
    )


def calculate_subject_selection(
    *,
    identity_id: str,
    experiment_group: str,
    enrollment_embeddings: FloatArray,
    reference_center: FloatArray,
    reference_dispersion: FloatArray,
    config: FeatureSelectionConfig,
) -> SubjectFeatureSelection:
    if enrollment_embeddings.ndim != 2:
        raise FeatureSelectionError(
            f"{identity_id}: enrollment embeddings must "
            f"have shape [N, D]"
        )

    if len(enrollment_embeddings) < 2:
        raise FeatureSelectionError(
            f"{identity_id}: at least two enrollment "
            "embeddings are required for stability estimation"
        )

    embedding_dimension = int(
        enrollment_embeddings.shape[1]
    )

    config.validate(
        embedding_dimension=embedding_dimension
    )

    subject_center = dimensionwise_median(
        enrollment_embeddings
    )

    subject_dispersion = dimensionwise_mad(
        enrollment_embeddings,
        center=subject_center,
    )

    if reference_center.shape != (
        embedding_dimension,
    ):
        raise FeatureSelectionError(
            f"{identity_id}: invalid reference-center shape "
            f"{reference_center.shape}"
        )

    if reference_dispersion.shape != (
        embedding_dimension,
    ):
        raise FeatureSelectionError(
            f"{identity_id}: invalid reference-dispersion shape "
            f"{reference_dispersion.shape}"
        )

    stability_raw = -np.log(
        subject_dispersion.astype(np.float64)
        + config.epsilon
    )

    discrimination_raw = (
        np.abs(
            subject_center.astype(np.float64)
            - reference_center.astype(np.float64)
        )
        / (
            config.mad_scale
            * reference_dispersion.astype(np.float64)
            + config.epsilon
        )
    )

    stability_raw = np.asarray(
        stability_raw,
        dtype=np.float32,
    )

    discrimination_raw = np.asarray(
        discrimination_raw,
        dtype=np.float32,
    )

    stability_score = percentile_rank(
        stability_raw
    )

    discrimination_score = percentile_rank(
        discrimination_raw
    )

    alpha = config.stability_weight

    combined_score = (
        alpha * stability_score
        + (1.0 - alpha) * discrimination_score
    ).astype(
        np.float32,
        copy=False,
    )

    selected_dimensions, selected_scores = (
        select_top_k(
            combined_scores=combined_score,
            top_k=config.top_k,
        )
    )

    return SubjectFeatureSelection(
        identity_id=identity_id,
        experiment_group=experiment_group,
        subject_center=subject_center,
        subject_dispersion=subject_dispersion,
        reference_center=reference_center,
        reference_dispersion=reference_dispersion,
        stability_raw=stability_raw,
        discrimination_raw=discrimination_raw,
        stability_score=stability_score,
        discrimination_score=discrimination_score,
        combined_score=combined_score,
        selected_dimensions=selected_dimensions,
        selected_scores=selected_scores,
    )


def build_feature_selection_set(
    repository: EmbeddingRepository,
    *,
    enrollment_count: int,
    config: FeatureSelectionConfig,
) -> FeatureSelectionSet:
    if enrollment_count < 2:
        raise FeatureSelectionError(
            "Subject-specific stability requires at least "
            "two enrollment images"
        )

    identity_caches = list(
        repository.iter_caches()
    )

    if not identity_caches:
        raise FeatureSelectionError(
            "Embedding repository is empty"
        )

    identity_ids = [
        cache.identity_id
        for cache in identity_caches
    ]

    experiment_groups = [
        cache.experiment_group
        for cache in identity_caches
    ]

    if len(set(identity_ids)) != len(identity_ids):
        raise FeatureSelectionError(
            "Duplicate identities were found in the repository"
        )

    background_caches = [
        cache
        for cache in identity_caches
        if cache.experiment_group == "background"
    ]

    if len(background_caches) < 2:
        raise FeatureSelectionError(
            "At least two background identities are required"
        )

    background_identity_ids = [
        cache.identity_id
        for cache in background_caches
    ]

    background_representatives = np.stack(
        [
            dimensionwise_median(
                cache.enrollment_embeddings(
                    enrollment_count
                )
            )
            for cache in background_caches
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    global_background_center, global_background_dispersion = (
        calculate_background_statistics(
            background_representatives
        )
    )

    selections: list[SubjectFeatureSelection] = []

    for cache in identity_caches:
        enrollment_embeddings = (
            cache.enrollment_embeddings(
                enrollment_count
            )
        )

        reference_center, reference_dispersion = (
            build_background_reference(
                identity_id=cache.identity_id,
                experiment_group=cache.experiment_group,
                background_identity_ids=(
                    background_identity_ids
                ),
                background_embeddings=(
                    background_representatives
                ),
                global_center=(
                    global_background_center
                ),
                global_dispersion=(
                    global_background_dispersion
                ),
                use_leave_one_out=(
                    config.use_leave_one_out_background
                ),
            )
        )

        selection = calculate_subject_selection(
            identity_id=cache.identity_id,
            experiment_group=cache.experiment_group,
            enrollment_embeddings=enrollment_embeddings,
            reference_center=reference_center,
            reference_dispersion=reference_dispersion,
            config=config,
        )

        selections.append(selection)

    return FeatureSelectionSet(
        enrollment_count=enrollment_count,
        top_k=config.top_k,
        stability_weight=config.stability_weight,
        identity_ids=np.asarray(
            [
                selection.identity_id
                for selection in selections
            ],
            dtype=np.str_,
        ),
        experiment_groups=np.asarray(
            [
                selection.experiment_group
                for selection in selections
            ],
            dtype=np.str_,
        ),
        subject_centers=np.stack(
            [
                selection.subject_center
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        subject_dispersions=np.stack(
            [
                selection.subject_dispersion
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        reference_centers=np.stack(
            [
                selection.reference_center
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        reference_dispersions=np.stack(
            [
                selection.reference_dispersion
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        stability_raw_scores=np.stack(
            [
                selection.stability_raw
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        discrimination_raw_scores=np.stack(
            [
                selection.discrimination_raw
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        stability_scores=np.stack(
            [
                selection.stability_score
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        discrimination_scores=np.stack(
            [
                selection.discrimination_score
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        combined_scores=np.stack(
            [
                selection.combined_score
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        selected_dimensions=np.stack(
            [
                selection.selected_dimensions
                for selection in selections
            ],
            axis=0,
        ).astype(np.int16),
        selected_scores=np.stack(
            [
                selection.selected_scores
                for selection in selections
            ],
            axis=0,
        ).astype(np.float32),
        global_background_center=(
            global_background_center.astype(
                np.float32
            )
        ),
        global_background_dispersion=(
            global_background_dispersion.astype(
                np.float32
            )
        ),
        background_identity_ids=np.asarray(
            background_identity_ids,
            dtype=np.str_,
        ),
    )