from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fuzzy_did.visualization import (
    BiometricPlotError,
    CurveData,
    DistanceDistributionData,
    plot_det_curve,
    plot_distance_distribution,
    plot_distribution_comparison,
    plot_roc_curve,
)


class VisualizationBuildError(RuntimeError):
    """Raised when evaluation figures cannot be generated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ROC, DET, and Hamming-distance distributions "
            "from biometric evaluation results."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/"
            "embedding_facenet512.yaml"
        ),
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise VisualizationBuildError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise VisualizationBuildError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise VisualizationBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise VisualizationBuildError(
            "Configuration root must be a mapping"
        )

    return loaded


def load_threshold_summary(
    path: Path,
) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise VisualizationBuildError(
            f"Threshold summary does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    return {
        int(result["enrollment_count"]): result
        for result in summary["results"]
    }


def load_curve_data(
    *,
    enrollment_count: int,
    sweep_path: Path,
    threshold_result: dict[str, Any],
    group: str,
) -> CurveData:
    if not sweep_path.is_file():
        raise VisualizationBuildError(
            f"Threshold sweep does not exist: {sweep_path}"
        )

    dataframe = pd.read_csv(
        sweep_path
    )

    prefix = (
        "evaluation"
        if group == "evaluation"
        else "development"
    )

    required_columns = {
        "threshold",
        f"{prefix}_far",
        f"{prefix}_frr",
        f"{prefix}_tar",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise VisualizationBuildError(
            f"Missing sweep columns: "
            f"{sorted(missing_columns)}"
        )

    if group == "evaluation":
        eer_threshold = int(
            threshold_result["evaluation"][
                "oracle_eer_threshold"
            ]
        )

        eer = float(
            threshold_result["evaluation"][
                "oracle_eer"
            ]
        )
    else:
        eer_threshold = int(
            threshold_result["development"][
                "eer_threshold"
            ]
        )

        eer = float(
            threshold_result["development"][
                "eer"
            ]
        )

    return CurveData(
        enrollment_count=enrollment_count,
        thresholds=dataframe[
            "threshold"
        ].to_numpy(dtype=np.int16),
        false_accept_rates=dataframe[
            f"{prefix}_far"
        ].to_numpy(dtype=np.float64),
        false_reject_rates=dataframe[
            f"{prefix}_frr"
        ].to_numpy(dtype=np.float64),
        true_accept_rates=dataframe[
            f"{prefix}_tar"
        ].to_numpy(dtype=np.float64),
        eer_threshold=eer_threshold,
        eer=eer,
    )


def load_distance_distribution(
    *,
    enrollment_count: int,
    trial_path: Path,
    selected_threshold: int,
) -> DistanceDistributionData:
    if not trial_path.is_file():
        raise VisualizationBuildError(
            f"Hamming trial file does not exist: "
            f"{trial_path}"
        )

    dataframe = pd.read_parquet(
        trial_path,
        columns=[
            "trial_type",
            "hamming_distance",
        ],
    )

    genuine = dataframe.loc[
        dataframe["trial_type"] == "genuine",
        "hamming_distance",
    ].to_numpy(dtype=np.int16)

    impostor = dataframe.loc[
        dataframe["trial_type"] == "impostor",
        "hamming_distance",
    ].to_numpy(dtype=np.int16)

    if len(genuine) == 0:
        raise VisualizationBuildError(
            f"No genuine trials in {trial_path}"
        )

    if len(impostor) == 0:
        raise VisualizationBuildError(
            f"No impostor trials in {trial_path}"
        )

    return DistanceDistributionData(
        enrollment_count=enrollment_count,
        genuine_distances=genuine,
        impostor_distances=impostor,
        selected_threshold=(
            selected_threshold
        ),
    )


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(
            args.config
        )

        trial_config = config[
            "hamming_trials"
        ]

        visualization_config = config[
            "visualization"
        ]

        trial_root = Path(
            trial_config["output_dir"]
        ).expanduser().resolve()

        threshold_root = (
            trial_root / "threshold_evaluation"
        )

        threshold_summary_path = (
            threshold_root
            / "threshold_evaluation_summary.json"
        )

        output_root = Path(
            visualization_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value in visualization_config[
                "enrollment_counts"
            ]
        ]

        curve_group = str(
            visualization_config.get(
                "curve_group",
                "evaluation",
            )
        )

        distribution_group = str(
            visualization_config.get(
                "distribution_group",
                "evaluation",
            )
        )

        template_length = int(
            visualization_config.get(
                "template_length",
                128,
            )
        )

        det_log_scale = bool(
            visualization_config.get(
                "det_log_scale",
                True,
            )
        )

        if curve_group not in {
            "development",
            "evaluation",
        }:
            raise VisualizationBuildError(
                "curve_group must be development or evaluation"
            )

        if distribution_group not in {
            "development",
            "evaluation",
        }:
            raise VisualizationBuildError(
                "distribution_group must be "
                "development or evaluation"
            )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        threshold_results = (
            load_threshold_summary(
                threshold_summary_path
            )
        )

        curves: list[CurveData] = []
        distributions: list[
            DistanceDistributionData
        ] = []

        figure_records: list[
            dict[str, Any]
        ] = []

        print(
            "Enrollment counts:",
            enrollment_counts,
        )
        print(
            "Curve group:",
            curve_group,
        )
        print(
            "Distribution group:",
            distribution_group,
        )

        for enrollment_count in enrollment_counts:
            if (
                enrollment_count
                not in threshold_results
            ):
                raise VisualizationBuildError(
                    f"Threshold result missing for "
                    f"enrollment={enrollment_count}"
                )

            threshold_result = (
                threshold_results[
                    enrollment_count
                ]
            )

            sweep_path = (
                threshold_root
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}_"
                    "threshold_sweep.csv"
                )
            )

            curve = load_curve_data(
                enrollment_count=(
                    enrollment_count
                ),
                sweep_path=sweep_path,
                threshold_result=(
                    threshold_result
                ),
                group=curve_group,
            )

            selected_threshold = int(
                threshold_result[
                    "selected_threshold"
                ]
            )

            trial_path = (
                trial_root
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}"
                )
                / f"{distribution_group}.parquet"
            )

            distribution = (
                load_distance_distribution(
                    enrollment_count=(
                        enrollment_count
                    ),
                    trial_path=trial_path,
                    selected_threshold=(
                        selected_threshold
                    ),
                )
            )

            curves.append(curve)
            distributions.append(
                distribution
            )

            enrollment_output_dir = (
                output_root
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}"
                )
            )

            plot_roc_curve(
                [curve],
                enrollment_output_dir
                / "roc_curve",
                title=(
                    f"ROC Curve — Enrollment "
                    f"{enrollment_count}"
                ),
            )

            plot_det_curve(
                [curve],
                enrollment_output_dir
                / "det_curve",
                title=(
                    f"DET Curve — Enrollment "
                    f"{enrollment_count}"
                ),
                use_log_scale=(
                    det_log_scale
                ),
            )

            plot_distance_distribution(
                distribution,
                enrollment_output_dir
                / "hamming_distribution",
                template_length=(
                    template_length
                ),
                title=(
                    f"Hamming Distance Distribution "
                    f"— Enrollment "
                    f"{enrollment_count}"
                ),
            )

            figure_records.append(
                {
                    "enrollment_count": (
                        enrollment_count
                    ),
                    "selected_threshold": (
                        selected_threshold
                    ),
                    "curve_group": (
                        curve_group
                    ),
                    "curve_eer": curve.eer,
                    "genuine_trial_count": (
                        len(
                            distribution
                            .genuine_distances
                        )
                    ),
                    "impostor_trial_count": (
                        len(
                            distribution
                            .impostor_distances
                        )
                    ),
                    "output_dir": str(
                        enrollment_output_dir
                    ),
                }
            )

            print(
                f"Enrollment {enrollment_count}: "
                f"EER={curve.eer:.6f}, "
                f"threshold={selected_threshold}"
            )

        plot_roc_curve(
            curves,
            output_root
            / "roc_all_enrollments",
            title=(
                "ROC Curves for Enrollment Sizes"
            ),
        )

        plot_det_curve(
            curves,
            output_root
            / "det_all_enrollments",
            title=(
                "DET Curves for Enrollment Sizes"
            ),
            use_log_scale=det_log_scale,
        )

        plot_distribution_comparison(
            distributions,
            output_root
            / (
                "hamming_distribution_"
                "all_enrollments"
            ),
            template_length=template_length,
            title=(
                "Genuine and Impostor "
                "Hamming-Distance Distributions"
            ),
        )

        summary_payload = {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "curve_group": curve_group,
            "distribution_group": (
                distribution_group
            ),
            "template_length": (
                template_length
            ),
            "det_log_scale": (
                det_log_scale
            ),
            "figures": figure_records,
        }

        summary_path = (
            output_root
            / "figure_summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary_payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            "Saved figures:",
            output_root,
        )

        return 0

    except KeyError as exc:
        print(
            f"Missing configuration key: {exc}",
            file=sys.stderr,
        )
        return 2

    except (
        VisualizationBuildError,
        BiometricPlotError,
    ) as exc:
        print(
            f"Visualization failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Visualization interrupted",
            file=sys.stderr,
        )
        return 130

    except Exception as exc:
        print(
            f"Unexpected error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 99


if __name__ == "__main__":
    sys.exit(main())