from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int16]


class BiometricPlotError(RuntimeError):
    """Raised when biometric evaluation figures cannot be generated."""


@dataclass(frozen=True)
class CurveData:
    enrollment_count: int
    thresholds: IntArray
    false_accept_rates: FloatArray
    false_reject_rates: FloatArray
    true_accept_rates: FloatArray
    eer_threshold: int
    eer: float


@dataclass(frozen=True)
class DistanceDistributionData:
    enrollment_count: int
    genuine_distances: npt.NDArray[np.int16]
    impostor_distances: npt.NDArray[np.int16]
    selected_threshold: int


def validate_curve_data(
    curve: CurveData,
) -> None:
    arrays = {
        "thresholds": curve.thresholds,
        "false_accept_rates": curve.false_accept_rates,
        "false_reject_rates": curve.false_reject_rates,
        "true_accept_rates": curve.true_accept_rates,
    }

    lengths = {
        name: len(value)
        for name, value in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise BiometricPlotError(
            f"Curve array lengths differ: {lengths}"
        )

    if len(curve.thresholds) == 0:
        raise BiometricPlotError(
            "Curve data is empty"
        )

    for name in (
        "false_accept_rates",
        "false_reject_rates",
        "true_accept_rates",
    ):
        values = arrays[name]

        if not np.isfinite(values).all():
            raise BiometricPlotError(
                f"{name} contains NaN or infinity"
            )

        if (
            (values < 0.0).any()
            or (values > 1.0).any()
        ):
            raise BiometricPlotError(
                f"{name} contains values outside [0, 1]"
            )


def save_figure(
    figure: plt.Figure,
    output_base: Path,
    *,
    dpi: int = 300,
) -> None:
    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_roc_curve(
    curves: Sequence[CurveData],
    output_base: Path,
    *,
    title: str,
) -> None:
    if not curves:
        raise BiometricPlotError(
            "At least one ROC curve is required"
        )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.6),
    )

    for curve in curves:
        validate_curve_data(curve)

        axis.plot(
            curve.false_accept_rates,
            curve.true_accept_rates,
            linewidth=2,
            label=(
                f"Enrollment {curve.enrollment_count} "
                f"(EER={curve.eer * 100:.2f}%)"
            ),
        )

        eer_index = int(
            np.argmin(
                np.abs(
                    curve.thresholds
                    - curve.eer_threshold
                )
            )
        )

        axis.scatter(
            curve.false_accept_rates[eer_index],
            curve.true_accept_rates[eer_index],
            s=35,
            zorder=3,
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    axis.set_xlabel(
        "False Accept Rate (FAR)"
    )
    axis.set_ylabel(
        "True Accept Rate (TAR)"
    )
    axis.set_title(title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(
        True,
        linestyle=":",
        linewidth=0.7,
    )
    axis.legend(
        loc="lower right",
    )

    save_figure(
        figure,
        output_base,
    )


def plot_det_curve(
    curves: Sequence[CurveData],
    output_base: Path,
    *,
    title: str,
    use_log_scale: bool = True,
) -> None:
    if not curves:
        raise BiometricPlotError(
            "At least one DET curve is required"
        )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.6),
    )

    minimum_rate = 1e-5

    for curve in curves:
        validate_curve_data(curve)

        far = np.clip(
            curve.false_accept_rates,
            minimum_rate,
            1.0,
        )

        frr = np.clip(
            curve.false_reject_rates,
            minimum_rate,
            1.0,
        )

        axis.plot(
            far,
            frr,
            linewidth=2,
            label=(
                f"Enrollment {curve.enrollment_count} "
                f"(EER={curve.eer * 100:.2f}%)"
            ),
        )

        eer_index = int(
            np.argmin(
                np.abs(
                    curve.thresholds
                    - curve.eer_threshold
                )
            )
        )

        axis.scatter(
            far[eer_index],
            frr[eer_index],
            s=35,
            zorder=3,
        )

    if use_log_scale:
        axis.set_xscale("log")
        axis.set_yscale("log")

        axis.set_xlim(
            minimum_rate,
            1.0,
        )
        axis.set_ylim(
            minimum_rate,
            1.0,
        )
    else:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)

    axis.set_xlabel(
        "False Accept Rate (FAR)"
    )
    axis.set_ylabel(
        "False Reject Rate (FRR)"
    )
    axis.set_title(title)
    axis.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.7,
    )
    axis.legend(
        loc="upper right",
    )

    save_figure(
        figure,
        output_base,
    )


def calculate_histogram_density(
    distances: npt.ArrayLike,
    *,
    template_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        distances,
        dtype=np.int16,
    )

    if values.ndim != 1:
        raise BiometricPlotError(
            "Distances must be one-dimensional"
        )

    if len(values) == 0:
        raise BiometricPlotError(
            "Distances cannot be empty"
        )

    if (
        (values < 0).any()
        or (values > template_length).any()
    ):
        raise BiometricPlotError(
            "Distance is outside the template range"
        )

    bin_edges = np.arange(
        -0.5,
        template_length + 1.5,
        1.0,
    )

    histogram, edges = np.histogram(
        values,
        bins=bin_edges,
        density=True,
    )

    centers = (
        edges[:-1] + edges[1:]
    ) / 2.0

    return centers, histogram


def plot_distance_distribution(
    distribution: DistanceDistributionData,
    output_base: Path,
    *,
    template_length: int,
    title: str,
) -> None:
    genuine_x, genuine_density = (
        calculate_histogram_density(
            distribution.genuine_distances,
            template_length=template_length,
        )
    )

    impostor_x, impostor_density = (
        calculate_histogram_density(
            distribution.impostor_distances,
            template_length=template_length,
        )
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.6),
    )

    axis.plot(
        genuine_x,
        genuine_density,
        linewidth=2,
        label="Genuine",
    )

    axis.fill_between(
        genuine_x,
        genuine_density,
        alpha=0.25,
    )

    axis.plot(
        impostor_x,
        impostor_density,
        linewidth=2,
        label="Impostor",
    )

    axis.fill_between(
        impostor_x,
        impostor_density,
        alpha=0.25,
    )

    axis.axvline(
        distribution.selected_threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Threshold = "
            f"{distribution.selected_threshold}"
        ),
    )

    axis.set_xlabel(
        "Hamming Distance"
    )
    axis.set_ylabel(
        "Probability Density"
    )
    axis.set_title(title)
    axis.set_xlim(
        0,
        template_length,
    )
    axis.grid(
        True,
        linestyle=":",
        linewidth=0.7,
    )
    axis.legend()

    save_figure(
        figure,
        output_base,
    )


def plot_distribution_comparison(
    distributions: Sequence[
        DistanceDistributionData
    ],
    output_base: Path,
    *,
    template_length: int,
    title: str,
) -> None:
    if not distributions:
        raise BiometricPlotError(
            "At least one distribution is required"
        )

    figure, axis = plt.subplots(
        figsize=(8.2, 5.8),
    )

    # Impostor 분포는 enrollment 수에 따라 거의 동일하므로
    # 첫 번째 조건만 대표로 표시한다.
    first = distributions[0]

    impostor_x, impostor_density = (
        calculate_histogram_density(
            first.impostor_distances,
            template_length=template_length,
        )
    )

    axis.plot(
        impostor_x,
        impostor_density,
        linewidth=2.5,
        linestyle="--",
        label="Impostor",
    )

    for distribution in distributions:
        genuine_x, genuine_density = (
            calculate_histogram_density(
                distribution.genuine_distances,
                template_length=template_length,
            )
        )

        axis.plot(
            genuine_x,
            genuine_density,
            linewidth=2,
            label=(
                f"Genuine, enrollment "
                f"{distribution.enrollment_count}"
            ),
        )

    axis.set_xlabel(
        "Hamming Distance"
    )
    axis.set_ylabel(
        "Probability Density"
    )
    axis.set_title(title)
    axis.set_xlim(
        0,
        template_length,
    )
    axis.grid(
        True,
        linestyle=":",
        linewidth=0.7,
    )
    axis.legend()

    save_figure(
        figure,
        output_base,
    )