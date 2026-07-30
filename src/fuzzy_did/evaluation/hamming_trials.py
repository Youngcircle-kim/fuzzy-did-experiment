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
IntArray = npt.NDArray[np.int16]


class HammingTrialError(RuntimeError):
    """Raised when Hamming trials cannot be generated."""


@dataclass(frozen=True)
class HammingTrialResult:
    hamming_distances: npt.NDArray[np.int16]
    normalized_distances: FloatArray

    @property
    def trial_count(self) -> int:
        return int(self.hamming_distances.shape[0])


def validate_binary_template(
    template: BitArray,
) -> None:
    template = np.asarray(
        template,
        dtype=np.uint8,
    )

    if template.ndim != 1:
        raise HammingTrialError(
            f"Enrollment template must be one-dimensional, "
            f"got {template.shape}"
        )

    if not np.isin(
        template,
        [0, 1],
    ).all():
        raise HammingTrialError(
            "Enrollment template contains non-binary values"
        )


def hamming_distance_batch(
    enrollment_template: BitArray,
    probe_templates: BitArray,
) -> HammingTrialResult:
    """
    Compute Hamming distances between one enrollment template
    and multiple probe templates.
    """

    enrollment = np.asarray(
        enrollment_template,
        dtype=np.uint8,
    )

    probes = np.asarray(
        probe_templates,
        dtype=np.uint8,
    )

    validate_binary_template(enrollment)

    if probes.ndim != 2:
        raise HammingTrialError(
            f"Probe templates must have shape [N, L], "
            f"got {probes.shape}"
        )

    if probes.shape[1] != enrollment.shape[0]:
        raise HammingTrialError(
            f"Template length mismatch: "
            f"enrollment={enrollment.shape[0]}, "
            f"probe={probes.shape[1]}"
        )

    if not np.isin(
        probes,
        [0, 1],
    ).all():
        raise HammingTrialError(
            "Probe templates contain non-binary values"
        )

    distances = np.count_nonzero(
        probes != enrollment[None, :],
        axis=1,
    ).astype(np.int16)

    normalized = (
        distances.astype(np.float32)
        / float(enrollment.shape[0])
    )

    return HammingTrialResult(
        hamming_distances=distances,
        normalized_distances=normalized,
    )


def transform_probe_embeddings_for_claimant(
    *,
    probe_embeddings: FloatArray,
    claimant_selected_dimensions: IntArray,
    scaler_state: RobustScalerState,
    binarizer_config: MedianBinarizerConfig,
) -> BitArray:
    """
    Transform arbitrary probe embeddings using the claimant's
    selected dimensions.

    This is required for impostor trials. A probe must be encoded
    in the same feature space as the claimed identity.
    """

    embeddings = np.asarray(
        probe_embeddings,
        dtype=np.float32,
    )

    dimensions = np.asarray(
        claimant_selected_dimensions,
        dtype=np.int16,
    )

    if embeddings.ndim != 2:
        raise HammingTrialError(
            f"probe_embeddings must have shape [N, D], "
            f"got {embeddings.shape}"
        )

    if dimensions.ndim != 1:
        raise HammingTrialError(
            "claimant_selected_dimensions must be one-dimensional"
        )

    if len(np.unique(dimensions)) != len(dimensions):
        raise HammingTrialError(
            "Claimant dimensions contain duplicates"
        )

    if (
        (dimensions < 0).any()
        or (dimensions >= embeddings.shape[1]).any()
    ):
        raise HammingTrialError(
            "Claimant dimension is out of range"
        )

    normalized = transform_embeddings(
        embeddings,
        scaler_state=scaler_state,
    )

    selected_values = normalized[
        :,
        dimensions.astype(np.int64),
    ]

    return binarize_values(
        selected_values,
        config=binarizer_config,
    )