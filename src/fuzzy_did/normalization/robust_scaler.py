from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int16]
StringArray = npt.NDArray[np.str_]


class RobustNormalizationError(RuntimeError):
    """Raised when robust normalization cannot be performed."""


@dataclass(frozen=True)
class RobustScalerConfig:
    lower_quantile: float = 25.0
    upper_quantile: float = 75.0
    scale_floor: float = 1e-6

    def validate(self) -> None:
        if not 0.0 <= self.lower_quantile < 100.0:
            raise RobustNormalizationError(
                "lower_quantile must be in [0, 100)"
            )

        if not 0.0 < self.upper_quantile <= 100.0:
            raise RobustNormalizationError(
                "upper_quantile must be in (0, 100]"
            )

        if self.lower_quantile >= self.upper_quantile:
            raise RobustNormalizationError(
                "lower_quantile must be smaller than upper_quantile"
            )

        if self.scale_floor <= 0.0:
            raise RobustNormalizationError(
                "scale_floor must be positive"
            )


@dataclass(frozen=True)
class RobustScalerState:
    center: FloatArray
    scale: FloatArray
    raw_scale: FloatArray
    q1: FloatArray
    q3: FloatArray
    floored_dimensions: npt.NDArray[np.bool_]

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])


@dataclass(frozen=True)
class RobustNormalizationSet:
    enrollment_count: int
    top_k: int

    identity_ids: StringArray
    experiment_groups: StringArray
    selected_dimensions: IntArray

    raw_subject_centers: FloatArray
    normalized_subject_centers: FloatArray

    raw_selected_centers: FloatArray
    normalized_selected_centers: FloatArray

    scaler_state: RobustScalerState

    @property
    def identity_count(self) -> int:
        return int(self.identity_ids.shape[0])

    @property
    def embedding_dimension(self) -> int:
        return int(self.raw_subject_centers.shape[1])


def fit_robust_scaler(
    background_embeddings: FloatArray,
    *,
    config: RobustScalerConfig,
) -> RobustScalerState:
    """
    Fit a dimension-wise robust scaler using background identities only.
    """

    config.validate()

    values = np.asarray(
        background_embeddings,
        dtype=np.float32,
    )

    if values.ndim != 2:
        raise RobustNormalizationError(
            f"background_embeddings must have shape [N, D], "
            f"got {values.shape}"
        )

    if len(values) < 2:
        raise RobustNormalizationError(
            "At least two background identities are required"
        )

    if not np.isfinite(values).all():
        raise RobustNormalizationError(
            "background_embeddings contain NaN or infinity"
        )

    center = np.median(
        values,
        axis=0,
    ).astype(np.float32)

    q1 = np.percentile(
        values,
        config.lower_quantile,
        axis=0,
    ).astype(np.float32)

    q3 = np.percentile(
        values,
        config.upper_quantile,
        axis=0,
    ).astype(np.float32)

    raw_scale = (
        q3 - q1
    ).astype(np.float32)

    floored_dimensions = (
        raw_scale < config.scale_floor
    )

    scale = np.maximum(
        raw_scale,
        config.scale_floor,
    ).astype(np.float32)

    arrays = {
        "center": center,
        "q1": q1,
        "q3": q3,
        "raw_scale": raw_scale,
        "scale": scale,
    }

    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise RobustNormalizationError(
                f"{name} contains NaN or infinity"
            )

    if np.any(scale <= 0.0):
        raise RobustNormalizationError(
            "robust scale contains non-positive values"
        )

    return RobustScalerState(
        center=center,
        scale=scale,
        raw_scale=raw_scale,
        q1=q1,
        q3=q3,
        floored_dimensions=(
            floored_dimensions.astype(np.bool_)
        ),
    )


def transform_embeddings(
    embeddings: FloatArray,
    *,
    scaler_state: RobustScalerState,
) -> FloatArray:
    """
    Apply the fitted dimension-wise robust transformation.
    """

    values = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if values.ndim not in {1, 2}:
        raise RobustNormalizationError(
            f"embeddings must be one- or two-dimensional, "
            f"got {values.shape}"
        )

    if values.shape[-1] != scaler_state.dimension:
        raise RobustNormalizationError(
            f"embedding dimension mismatch: "
            f"expected={scaler_state.dimension}, "
            f"actual={values.shape[-1]}"
        )

    normalized = (
        (
            values.astype(np.float64)
            - scaler_state.center.astype(np.float64)
        )
        / scaler_state.scale.astype(np.float64)
    )

    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    if not np.isfinite(normalized).all():
        raise RobustNormalizationError(
            "normalized embeddings contain NaN or infinity"
        )

    return normalized


def gather_selected_dimensions(
    values: FloatArray,
    selected_dimensions: IntArray,
) -> FloatArray:
    """
    Gather identity-specific dimensions from a [N, D] matrix.

    Args:
        values:
            Matrix with shape [N, D].

        selected_dimensions:
            Identity-specific indices with shape [N, K].

    Returns:
        Matrix with shape [N, K].
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    dimensions = np.asarray(
        selected_dimensions,
        dtype=np.int64,
    )

    if values.ndim != 2:
        raise RobustNormalizationError(
            f"values must have shape [N, D], got {values.shape}"
        )

    if dimensions.ndim != 2:
        raise RobustNormalizationError(
            "selected_dimensions must have shape [N, K]"
        )

    if values.shape[0] != dimensions.shape[0]:
        raise RobustNormalizationError(
            "identity count mismatch between values and dimensions"
        )

    if (
        (dimensions < 0).any()
        or (dimensions >= values.shape[1]).any()
    ):
        raise RobustNormalizationError(
            "selected dimension index is out of range"
        )

    row_indices = np.arange(
        values.shape[0],
        dtype=np.int64,
    )[:, None]

    selected = values[
        row_indices,
        dimensions,
    ]

    return selected.astype(
        np.float32,
        copy=False,
    )


def build_robust_normalization_set(
    *,
    identity_ids: StringArray,
    experiment_groups: StringArray,
    subject_centers: FloatArray,
    selected_dimensions: IntArray,
    enrollment_count: int,
    top_k: int,
    config: RobustScalerConfig,
) -> RobustNormalizationSet:
    identity_ids = np.asarray(
        identity_ids,
        dtype=np.str_,
    )

    experiment_groups = np.asarray(
        experiment_groups,
        dtype=np.str_,
    )

    subject_centers = np.asarray(
        subject_centers,
        dtype=np.float32,
    )

    selected_dimensions = np.asarray(
        selected_dimensions,
        dtype=np.int16,
    )

    if subject_centers.ndim != 2:
        raise RobustNormalizationError(
            "subject_centers must have shape [N, D]"
        )

    identity_count = subject_centers.shape[0]

    if len(identity_ids) != identity_count:
        raise RobustNormalizationError(
            "identity_ids length mismatch"
        )

    if len(experiment_groups) != identity_count:
        raise RobustNormalizationError(
            "experiment_groups length mismatch"
        )

    if selected_dimensions.shape != (
        identity_count,
        top_k,
    ):
        raise RobustNormalizationError(
            f"selected_dimensions must have shape "
            f"({identity_count}, {top_k}), "
            f"got {selected_dimensions.shape}"
        )

    if len(np.unique(identity_ids)) != identity_count:
        raise RobustNormalizationError(
            "duplicate identity IDs detected"
        )

    if not np.isfinite(subject_centers).all():
        raise RobustNormalizationError(
            "subject_centers contain NaN or infinity"
        )

    background_mask = (
        experiment_groups.astype(str)
        == "background"
    )

    background_count = int(
        background_mask.sum()
    )

    if background_count < 2:
        raise RobustNormalizationError(
            "At least two background identities are required"
        )

    background_embeddings = subject_centers[
        background_mask
    ]

    scaler_state = fit_robust_scaler(
        background_embeddings,
        config=config,
    )

    normalized_subject_centers = transform_embeddings(
        subject_centers,
        scaler_state=scaler_state,
    )

    raw_selected_centers = gather_selected_dimensions(
        subject_centers,
        selected_dimensions,
    )

    normalized_selected_centers = gather_selected_dimensions(
        normalized_subject_centers,
        selected_dimensions,
    )

    return RobustNormalizationSet(
        enrollment_count=enrollment_count,
        top_k=top_k,
        identity_ids=identity_ids,
        experiment_groups=experiment_groups,
        selected_dimensions=selected_dimensions,
        raw_subject_centers=subject_centers,
        normalized_subject_centers=(
            normalized_subject_centers
        ),
        raw_selected_centers=(
            raw_selected_centers
        ),
        normalized_selected_centers=(
            normalized_selected_centers
        ),
        scaler_state=scaler_state,
    )