from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fuzzy_did.binarization import (
    MedianBinarizerConfig,
    binarize_values,
)
from fuzzy_did.normalization import (
    RobustScalerState,
    transform_embeddings,
)


FloatArray = npt.NDArray[np.float32]
BitArray = npt.NDArray[np.uint8]
DimensionArray = npt.NDArray[np.int64]
DistanceArray = npt.NDArray[np.int16]


class HammingTrialError(RuntimeError):
    """Raised when Hamming trials cannot be generated."""


@dataclass(frozen=True)
class HammingTrialResult:
    """
    Hamming-distance results for one enrollment template and
    multiple probe templates.
    """

    hamming_distances: DistanceArray
    normalized_distances: FloatArray

    @property
    def trial_count(self) -> int:
        return int(self.hamming_distances.shape[0])

    @property
    def template_length(self) -> int:
        if self.trial_count == 0:
            return 0

        positive_distances = self.normalized_distances > 0

        if not positive_distances.any():
            return 0

        ratios = (
            self.hamming_distances[positive_distances]
            / self.normalized_distances[positive_distances]
        )

        return int(round(float(np.median(ratios))))


def _as_binary_vector(
    values: npt.ArrayLike,
    *,
    name: str,
) -> BitArray:
    """
    Validate and normalize a one-dimensional binary vector.
    """

    array = np.asarray(values)

    if array.ndim != 1:
        raise HammingTrialError(
            f"{name} must be one-dimensional, "
            f"got shape={array.shape}"
        )

    if array.size == 0:
        raise HammingTrialError(
            f"{name} must not be empty"
        )

    if not np.issubdtype(
        array.dtype,
        np.integer,
    ) and not np.issubdtype(
        array.dtype,
        np.bool_,
    ):
        raise HammingTrialError(
            f"{name} must contain integer or boolean values, "
            f"got dtype={array.dtype}"
        )

    array = array.astype(
        np.uint8,
        copy=False,
    )

    if not np.logical_or(
        array == 0,
        array == 1,
    ).all():
        invalid_values = np.unique(
            array[
                np.logical_and(
                    array != 0,
                    array != 1,
                )
            ]
        )

        raise HammingTrialError(
            f"{name} contains non-binary values: "
            f"{invalid_values[:10].tolist()}"
        )

    return np.ascontiguousarray(array)


def _as_binary_matrix(
    values: npt.ArrayLike,
    *,
    name: str,
    expected_length: int | None = None,
    allow_empty_rows: bool = False,
) -> BitArray:
    """
    Validate and normalize a two-dimensional binary matrix.
    """

    array = np.asarray(values)

    if array.ndim != 2:
        raise HammingTrialError(
            f"{name} must have shape [N, L], "
            f"got shape={array.shape}"
        )

    if array.shape[1] == 0:
        raise HammingTrialError(
            f"{name} template length must be positive"
        )

    if (
        not allow_empty_rows
        and array.shape[0] == 0
    ):
        raise HammingTrialError(
            f"{name} must contain at least one row"
        )

    if (
        expected_length is not None
        and array.shape[1] != expected_length
    ):
        raise HammingTrialError(
            f"{name} template-length mismatch: "
            f"expected={expected_length}, "
            f"actual={array.shape[1]}"
        )

    if not np.issubdtype(
        array.dtype,
        np.integer,
    ) and not np.issubdtype(
        array.dtype,
        np.bool_,
    ):
        raise HammingTrialError(
            f"{name} must contain integer or boolean values, "
            f"got dtype={array.dtype}"
        )

    array = array.astype(
        np.uint8,
        copy=False,
    )

    if not np.logical_or(
        array == 0,
        array == 1,
    ).all():
        invalid_values = np.unique(
            array[
                np.logical_and(
                    array != 0,
                    array != 1,
                )
            ]
        )

        raise HammingTrialError(
            f"{name} contains non-binary values: "
            f"{invalid_values[:10].tolist()}"
        )

    return np.ascontiguousarray(array)


def _as_float_embedding_matrix(
    values: npt.ArrayLike,
    *,
    name: str,
) -> FloatArray:
    """
    Validate and normalize an embedding matrix.
    """

    array = np.asarray(
        values,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise HammingTrialError(
            f"{name} must have shape [N, D], "
            f"got shape={array.shape}"
        )

    if array.shape[0] == 0:
        raise HammingTrialError(
            f"{name} must contain at least one embedding"
        )

    if array.shape[1] == 0:
        raise HammingTrialError(
            f"{name} embedding dimension must be positive"
        )

    if not np.isfinite(array).all():
        invalid_count = int(
            (~np.isfinite(array)).sum()
        )

        raise HammingTrialError(
            f"{name} contains NaN or infinite values: "
            f"invalid_count={invalid_count}"
        )

    return np.ascontiguousarray(array)


def _as_dimension_vector(
    values: npt.ArrayLike,
    *,
    embedding_dimension: int,
    output_length: int | None = None,
) -> DimensionArray:
    """
    Validate claimant-specific selected feature dimensions.
    """

    raw = np.asarray(values)

    if raw.ndim != 1:
        raise HammingTrialError(
            "claimant_selected_dimensions must be "
            f"one-dimensional, got shape={raw.shape}"
        )

    if raw.size == 0:
        raise HammingTrialError(
            "claimant_selected_dimensions must not be empty"
        )

    if not np.issubdtype(
        raw.dtype,
        np.integer,
    ):
        raise HammingTrialError(
            "claimant_selected_dimensions must contain "
            f"integer indexes, got dtype={raw.dtype}"
        )

    dimensions = raw.astype(
        np.int64,
        copy=False,
    )

    if output_length is not None:
        if output_length <= 0:
            raise HammingTrialError(
                "output_length must be positive"
            )

        if output_length > len(dimensions):
            raise HammingTrialError(
                "output_length exceeds the number of selected "
                f"dimensions: output_length={output_length}, "
                f"available={len(dimensions)}"
            )

        dimensions = dimensions[
            :output_length
        ]

    unique_count = len(
        np.unique(dimensions)
    )

    if unique_count != len(dimensions):
        raise HammingTrialError(
            "Claimant selected dimensions contain duplicates: "
            f"total={len(dimensions)}, unique={unique_count}"
        )

    if (dimensions < 0).any():
        minimum = int(dimensions.min())

        raise HammingTrialError(
            "Claimant selected dimensions contain a negative "
            f"index: min={minimum}"
        )

    if (
        dimensions
        >= embedding_dimension
    ).any():
        maximum = int(dimensions.max())

        raise HammingTrialError(
            "Claimant selected dimension is out of range: "
            f"max={maximum}, "
            f"embedding_dimension={embedding_dimension}"
        )

    return np.ascontiguousarray(dimensions)


def _validate_scaler_state(
    *,
    scaler_state: RobustScalerState,
    embedding_dimension: int,
) -> None:
    """
    Check whether the robust-scaler state is compatible with the
    supplied embedding dimension.
    """

    vector_fields = {
        "center": scaler_state.center,
        "scale": scaler_state.scale,
        "raw_scale": scaler_state.raw_scale,
        "q1": scaler_state.q1,
        "q3": scaler_state.q3,
        "floored_dimensions": (
            scaler_state.floored_dimensions
        ),
    }

    for field_name, values in vector_fields.items():
        array = np.asarray(values)

        if array.ndim != 1:
            raise HammingTrialError(
                f"scaler_state.{field_name} must be "
                f"one-dimensional, got shape={array.shape}"
            )

        if len(array) != embedding_dimension:
            raise HammingTrialError(
                f"scaler_state.{field_name} dimension mismatch: "
                f"expected={embedding_dimension}, "
                f"actual={len(array)}"
            )

    numeric_fields = {
        "center": scaler_state.center,
        "scale": scaler_state.scale,
        "raw_scale": scaler_state.raw_scale,
        "q1": scaler_state.q1,
        "q3": scaler_state.q3,
    }

    for field_name, values in numeric_fields.items():
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if not np.isfinite(array).all():
            invalid_count = int(
                (~np.isfinite(array)).sum()
            )

            raise HammingTrialError(
                f"scaler_state.{field_name} contains "
                f"non-finite values: {invalid_count}"
            )

    scale = np.asarray(
        scaler_state.scale,
        dtype=np.float32,
    )

    if (scale <= 0).any():
        invalid_count = int(
            (scale <= 0).sum()
        )

        raise HammingTrialError(
            "scaler_state.scale contains non-positive "
            f"values: invalid_count={invalid_count}"
        )


def validate_binary_template(
    template: npt.ArrayLike,
) -> None:
    """
    Validate a single enrollment binary template.

    Kept as a public compatibility function.
    """

    _as_binary_vector(
        template,
        name="enrollment_template",
    )


def hamming_distance_batch(
    enrollment_template: npt.ArrayLike,
    probe_templates: npt.ArrayLike,
) -> HammingTrialResult:
    """
    Compute Hamming distances between one enrollment template and
    multiple probe templates.

    Parameters
    ----------
    enrollment_template:
        Binary array with shape [L].

    probe_templates:
        Binary array with shape [N, L].

    Returns
    -------
    HammingTrialResult
        Integer Hamming distances and normalized distances.
    """

    enrollment = _as_binary_vector(
        enrollment_template,
        name="enrollment_template",
    )

    probes = _as_binary_matrix(
        probe_templates,
        name="probe_templates",
        expected_length=len(enrollment),
    )

    distances = np.count_nonzero(
        probes != enrollment[None, :],
        axis=1,
    ).astype(
        np.int16,
        copy=False,
    )

    normalized = np.divide(
        distances,
        np.float32(len(enrollment)),
        dtype=np.float32,
    )

    return HammingTrialResult(
        hamming_distances=distances,
        normalized_distances=normalized,
    )


def transform_probe_embeddings_for_claimant(
    *,
    probe_embeddings: npt.ArrayLike,
    claimant_selected_dimensions: npt.ArrayLike,
    scaler_state: RobustScalerState,
    binarizer_config: MedianBinarizerConfig,
    output_length: int | None = None,
) -> BitArray:
    """
    Transform arbitrary probe embeddings into binary templates
    using the claimed identity's feature space.

    The probe never selects its own feature dimensions. Genuine and
    impostor probes are both normalized and projected using the
    claimant's selected dimensions.

    Parameters
    ----------
    probe_embeddings:
        Embedding matrix with shape [N, D].

    claimant_selected_dimensions:
        Ranked claimant-specific feature indexes.

    scaler_state:
        Robust-normalization state learned from the configured
        background data.

    binarizer_config:
        Configuration used to convert normalized feature values
        into binary values.

    output_length:
        Optional number of ranked dimensions to retain. For actual
        primitive BCH(127, k, d) evaluation, use output_length=127.
        If omitted, all supplied selected dimensions are used.

    Returns
    -------
    BitArray
        Binary matrix with shape [N, L].
    """

    embeddings = _as_float_embedding_matrix(
        probe_embeddings,
        name="probe_embeddings",
    )

    embedding_dimension = int(
        embeddings.shape[1]
    )

    _validate_scaler_state(
        scaler_state=scaler_state,
        embedding_dimension=embedding_dimension,
    )

    dimensions = _as_dimension_vector(
        claimant_selected_dimensions,
        embedding_dimension=embedding_dimension,
        output_length=output_length,
    )

    try:
        normalized = transform_embeddings(
            embeddings,
            scaler_state=scaler_state,
        )
    except Exception as exc:
        raise HammingTrialError(
            "Failed to robust-normalize probe embeddings"
        ) from exc

    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    if normalized.shape != embeddings.shape:
        raise HammingTrialError(
            "Unexpected normalized embedding shape: "
            f"expected={embeddings.shape}, "
            f"actual={normalized.shape}"
        )

    if not np.isfinite(normalized).all():
        invalid_count = int(
            (~np.isfinite(normalized)).sum()
        )

        raise HammingTrialError(
            "Normalized embeddings contain NaN or infinite "
            f"values: invalid_count={invalid_count}"
        )

    selected_values = normalized[
        :,
        dimensions,
    ]

    try:
        binary_templates = binarize_values(
            selected_values,
            config=binarizer_config,
        )
    except Exception as exc:
        raise HammingTrialError(
            "Failed to binarize claimant-conditioned "
            "probe feature values"
        ) from exc

    binary_templates = _as_binary_matrix(
        binary_templates,
        name="binary_probe_templates",
        expected_length=len(dimensions),
    )

    expected_shape = (
        embeddings.shape[0],
        len(dimensions),
    )

    if binary_templates.shape != expected_shape:
        raise HammingTrialError(
            "Unexpected binary probe-template shape: "
            f"expected={expected_shape}, "
            f"actual={binary_templates.shape}"
        )

    return binary_templates