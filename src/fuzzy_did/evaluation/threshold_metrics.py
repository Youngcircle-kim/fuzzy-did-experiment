from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


IntArray = npt.NDArray[np.int16]
FloatArray = npt.NDArray[np.float64]


class ThresholdEvaluationError(RuntimeError):
    """Raised when threshold metrics cannot be calculated."""


@dataclass(frozen=True)
class ThresholdSweepResult:
    thresholds: IntArray
    false_accept_rates: FloatArray
    false_reject_rates: FloatArray
    true_accept_rates: FloatArray
    true_reject_rates: FloatArray
    balanced_error_rates: FloatArray

    eer_threshold: int
    eer: float
    eer_far: float
    eer_frr: float

    genuine_count: int
    impostor_count: int


@dataclass(frozen=True)
class FixedThresholdResult:
    threshold: int

    genuine_count: int
    impostor_count: int

    true_accept_count: int
    false_reject_count: int
    false_accept_count: int
    true_reject_count: int

    false_accept_rate: float
    false_reject_rate: float
    true_accept_rate: float
    true_reject_rate: float

    balanced_error_rate: float
    accuracy: float


def validate_distances(
    genuine_distances: npt.ArrayLike,
    impostor_distances: npt.ArrayLike,
    *,
    template_length: int,
) -> tuple[IntArray, IntArray]:
    genuine = np.asarray(
        genuine_distances,
        dtype=np.int16,
    )

    impostor = np.asarray(
        impostor_distances,
        dtype=np.int16,
    )

    if genuine.ndim != 1:
        raise ThresholdEvaluationError(
            "genuine_distances must be one-dimensional"
        )

    if impostor.ndim != 1:
        raise ThresholdEvaluationError(
            "impostor_distances must be one-dimensional"
        )

    if len(genuine) == 0:
        raise ThresholdEvaluationError(
            "genuine_distances is empty"
        )

    if len(impostor) == 0:
        raise ThresholdEvaluationError(
            "impostor_distances is empty"
        )

    if template_length <= 0:
        raise ThresholdEvaluationError(
            "template_length must be positive"
        )

    if (
        (genuine < 0).any()
        or (genuine > template_length).any()
    ):
        raise ThresholdEvaluationError(
            "genuine distance is outside the valid range"
        )

    if (
        (impostor < 0).any()
        or (impostor > template_length).any()
    ):
        raise ThresholdEvaluationError(
            "impostor distance is outside the valid range"
        )

    return genuine, impostor


def evaluate_fixed_threshold(
    genuine_distances: npt.ArrayLike,
    impostor_distances: npt.ArrayLike,
    *,
    threshold: int,
    template_length: int,
) -> FixedThresholdResult:
    genuine, impostor = validate_distances(
        genuine_distances,
        impostor_distances,
        template_length=template_length,
    )

    if not 0 <= threshold <= template_length:
        raise ThresholdEvaluationError(
            f"threshold must be in [0, {template_length}]"
        )

    true_accept_count = int(
        (genuine <= threshold).sum()
    )

    false_reject_count = int(
        (genuine > threshold).sum()
    )

    false_accept_count = int(
        (impostor <= threshold).sum()
    )

    true_reject_count = int(
        (impostor > threshold).sum()
    )

    genuine_count = len(genuine)
    impostor_count = len(impostor)

    far = false_accept_count / impostor_count
    frr = false_reject_count / genuine_count

    tar = true_accept_count / genuine_count
    trr = true_reject_count / impostor_count

    balanced_error_rate = (
        far + frr
    ) / 2.0

    accuracy = (
        true_accept_count
        + true_reject_count
    ) / (
        genuine_count
        + impostor_count
    )

    return FixedThresholdResult(
        threshold=threshold,
        genuine_count=genuine_count,
        impostor_count=impostor_count,
        true_accept_count=true_accept_count,
        false_reject_count=false_reject_count,
        false_accept_count=false_accept_count,
        true_reject_count=true_reject_count,
        false_accept_rate=float(far),
        false_reject_rate=float(frr),
        true_accept_rate=float(tar),
        true_reject_rate=float(trr),
        balanced_error_rate=float(
            balanced_error_rate
        ),
        accuracy=float(accuracy),
    )


def sweep_thresholds(
    genuine_distances: npt.ArrayLike,
    impostor_distances: npt.ArrayLike,
    *,
    template_length: int,
) -> ThresholdSweepResult:
    genuine, impostor = validate_distances(
        genuine_distances,
        impostor_distances,
        template_length=template_length,
    )

    thresholds = np.arange(
        0,
        template_length + 1,
        dtype=np.int16,
    )

    far_values = np.empty(
        len(thresholds),
        dtype=np.float64,
    )

    frr_values = np.empty(
        len(thresholds),
        dtype=np.float64,
    )

    for index, threshold in enumerate(
        thresholds
    ):
        far_values[index] = np.mean(
            impostor <= threshold
        )

        frr_values[index] = np.mean(
            genuine > threshold
        )

    tar_values = 1.0 - frr_values
    trr_values = 1.0 - far_values

    balanced_errors = (
        far_values + frr_values
    ) / 2.0

    absolute_gap = np.abs(
        far_values - frr_values
    )

    minimum_gap = absolute_gap.min()

    candidate_indices = np.flatnonzero(
        np.isclose(
            absolute_gap,
            minimum_gap,
            rtol=0.0,
            atol=1e-15,
        )
    )

    # FAR-FRR 차이가 동일한 threshold가 여러 개면
    # balanced error가 가장 낮은 threshold 선택.
    best_candidate_offset = int(
        np.argmin(
            balanced_errors[
                candidate_indices
            ]
        )
    )

    eer_index = int(
        candidate_indices[
            best_candidate_offset
        ]
    )

    eer_far = float(
        far_values[eer_index]
    )

    eer_frr = float(
        frr_values[eer_index]
    )

    eer = (
        eer_far + eer_frr
    ) / 2.0

    return ThresholdSweepResult(
        thresholds=thresholds,
        false_accept_rates=far_values,
        false_reject_rates=frr_values,
        true_accept_rates=tar_values,
        true_reject_rates=trr_values,
        balanced_error_rates=(
            balanced_errors
        ),
        eer_threshold=int(
            thresholds[eer_index]
        ),
        eer=float(eer),
        eer_far=eer_far,
        eer_frr=eer_frr,
        genuine_count=len(genuine),
        impostor_count=len(impostor),
    )