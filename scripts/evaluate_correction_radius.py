from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


class CorrectionRadiusEvaluationError(RuntimeError):
    """Raised when correction-radius evaluation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ideal bounded-distance reconstruction rates "
            "for candidate BCH correction capabilities."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/embedding_facenet512.yaml"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise CorrectionRadiusEvaluationError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise CorrectionRadiusEvaluationError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise CorrectionRadiusEvaluationError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise CorrectionRadiusEvaluationError(
            "Configuration root must be a mapping"
        )

    return loaded


def load_distances(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise CorrectionRadiusEvaluationError(
            f"Hamming trial file does not exist: {path}"
        )

    dataframe = pd.read_parquet(
        path,
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
        raise CorrectionRadiusEvaluationError(
            f"No genuine trials found in {path}"
        )

    if len(impostor) == 0:
        raise CorrectionRadiusEvaluationError(
            f"No impostor trials found in {path}"
        )

    return genuine, impostor


def wilson_interval(
    success_count: int,
    total_count: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """
    Calculate a 95% Wilson score interval for a binomial proportion.
    """

    if total_count <= 0:
        raise CorrectionRadiusEvaluationError(
            "total_count must be positive"
        )

    probability = success_count / total_count
    z_squared = z * z

    denominator = (
        1.0
        + z_squared / total_count
    )

    center = (
        probability
        + z_squared / (2.0 * total_count)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                probability
                * (1.0 - probability)
                / total_count
            )
            + (
                z_squared
                / (4.0 * total_count**2)
            )
        )
        / denominator
    )

    return (
        float(max(0.0, center - margin)),
        float(min(1.0, center + margin)),
    )


def evaluate_radius(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    *,
    correction_capability: int,
    template_length: int,
) -> dict[str, Any]:
    if not 0 <= correction_capability <= template_length:
        raise CorrectionRadiusEvaluationError(
            f"Invalid correction capability: "
            f"{correction_capability}"
        )

    genuine_success_count = int(
        (
            genuine_distances
            <= correction_capability
        ).sum()
    )

    genuine_failure_count = int(
        len(genuine_distances)
        - genuine_success_count
    )

    false_reconstruction_count = int(
        (
            impostor_distances
            <= correction_capability
        ).sum()
    )

    true_rejection_count = int(
        len(impostor_distances)
        - false_reconstruction_count
    )

    grr = (
        genuine_success_count
        / len(genuine_distances)
    )

    genuine_failure_rate = (
        genuine_failure_count
        / len(genuine_distances)
    )

    false_reconstruction_rate = (
        false_reconstruction_count
        / len(impostor_distances)
    )

    true_rejection_rate = (
        true_rejection_count
        / len(impostor_distances)
    )

    grr_lower, grr_upper = wilson_interval(
        genuine_success_count,
        len(genuine_distances),
    )

    false_lower, false_upper = wilson_interval(
        false_reconstruction_count,
        len(impostor_distances),
    )

    return {
        "correction_capability": (
            correction_capability
        ),
        "genuine_trial_count": int(
            len(genuine_distances)
        ),
        "impostor_trial_count": int(
            len(impostor_distances)
        ),
        "genuine_success_count": (
            genuine_success_count
        ),
        "genuine_failure_count": (
            genuine_failure_count
        ),
        "false_reconstruction_count": (
            false_reconstruction_count
        ),
        "true_rejection_count": (
            true_rejection_count
        ),
        "genuine_reconstruction_rate": (
            float(grr)
        ),
        "genuine_reconstruction_rate_ci_lower": (
            grr_lower
        ),
        "genuine_reconstruction_rate_ci_upper": (
            grr_upper
        ),
        "genuine_failure_rate": float(
            genuine_failure_rate
        ),
        "false_reconstruction_rate": float(
            false_reconstruction_rate
        ),
        "false_reconstruction_rate_ci_lower": (
            false_lower
        ),
        "false_reconstruction_rate_ci_upper": (
            false_upper
        ),
        "true_rejection_rate": float(
            true_rejection_rate
        ),
    }


def save_radius_plot(
    dataframe: pd.DataFrame,
    output_base: Path,
    *,
    title: str,
) -> None:
    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(7.4, 5.4)
    )

    axis.plot(
        dataframe[
            "correction_capability"
        ],
        dataframe[
            "genuine_reconstruction_rate"
        ],
        marker="o",
        linewidth=2,
        label="Genuine Reconstruction Rate",
    )

    axis.plot(
        dataframe[
            "correction_capability"
        ],
        dataframe[
            "false_reconstruction_rate"
        ],
        marker="s",
        linewidth=2,
        label="False Reconstruction Rate",
    )

    axis.set_xlabel(
        "Correction Capability t"
    )

    axis.set_ylabel(
        "Rate"
    )

    axis.set_title(title)

    axis.set_ylim(
        -0.02,
        1.02,
    )

    axis.grid(
        True,
        linestyle=":",
        linewidth=0.7,
    )

    axis.legend()

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(figure)


def save_log_false_rate_plot(
    dataframe: pd.DataFrame,
    output_base: Path,
    *,
    title: str,
) -> None:
    """
    Plot the false reconstruction rate using a logarithmic y-axis.
    """

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_count = int(
        dataframe[
            "impostor_trial_count"
        ].iloc[0]
    )

    # Zero empirical rates cannot be displayed on a logarithmic axis.
    # Use half the inverse trial count only for plotting.
    plotting_floor = (
        0.5 / trial_count
    )

    false_rates = np.maximum(
        dataframe[
            "false_reconstruction_rate"
        ].to_numpy(dtype=np.float64),
        plotting_floor,
    )

    figure, axis = plt.subplots(
        figsize=(7.4, 5.4)
    )

    axis.plot(
        dataframe[
            "correction_capability"
        ],
        false_rates,
        marker="o",
        linewidth=2,
    )

    axis.set_yscale("log")

    axis.set_xlabel(
        "Correction Capability t"
    )

    axis.set_ylabel(
        "False Reconstruction Rate"
    )

    axis.set_title(title)

    axis.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.7,
    )

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(
            args.config
        )

        trial_config = config[
            "hamming_trials"
        ]

        radius_config = config[
            "correction_radius"
        ]

        trial_root = Path(
            trial_config["output_dir"]
        ).expanduser().resolve()

        output_root = Path(
            radius_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value in radius_config[
                "enrollment_counts"
            ]
        ]

        correction_capabilities = sorted(
            {
                int(value)
                for value in radius_config[
                    "correction_capabilities"
                ]
            }
        )

        experiment_groups = [
            str(value)
            for value in radius_config[
                "experiment_groups"
            ]
        ]

        template_length = int(
            radius_config.get(
                "template_length",
                128,
            )
        )

        if not correction_capabilities:
            raise CorrectionRadiusEvaluationError(
                "No correction capabilities configured"
            )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_records: list[
            dict[str, Any]
        ] = []

        print(
            "Enrollment counts:",
            enrollment_counts,
        )
        print(
            "Correction capabilities:",
            correction_capabilities,
        )
        print(
            "Groups:",
            experiment_groups,
        )
        print(
            "Template length:",
            template_length,
        )

        for enrollment_count in enrollment_counts:
            enrollment_output_dir = (
                output_root
                / f"enrollment_{enrollment_count:02d}"
            )

            enrollment_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for experiment_group in experiment_groups:
                trial_path = (
                    trial_root
                    / f"enrollment_{enrollment_count:02d}"
                    / f"{experiment_group}.parquet"
                )

                genuine, impostor = load_distances(
                    trial_path
                )

                group_records: list[
                    dict[str, Any]
                ] = []

                for capability in correction_capabilities:
                    metrics = evaluate_radius(
                        genuine,
                        impostor,
                        correction_capability=(
                            capability
                        ),
                        template_length=(
                            template_length
                        ),
                    )

                    record = {
                        "enrollment_count": (
                            enrollment_count
                        ),
                        "experiment_group": (
                            experiment_group
                        ),
                        "template_length": (
                            template_length
                        ),
                        **metrics,
                    }

                    group_records.append(
                        record
                    )

                    all_records.append(
                        record
                    )

                group_dataframe = (
                    pd.DataFrame.from_records(
                        group_records
                    )
                )

                csv_path = (
                    enrollment_output_dir
                    / (
                        f"{experiment_group}_"
                        "correction_radius.csv"
                    )
                )

                if (
                    csv_path.exists()
                    and not args.overwrite
                ):
                    raise CorrectionRadiusEvaluationError(
                        f"Output already exists: "
                        f"{csv_path}. "
                        "Use --overwrite."
                    )

                group_dataframe.to_csv(
                    csv_path,
                    index=False,
                )

                save_radius_plot(
                    group_dataframe,
                    enrollment_output_dir
                    / (
                        f"{experiment_group}_"
                        "grr_false_reconstruction"
                    ),
                    title=(
                        "Reconstruction Rates "
                        f"— Enrollment "
                        f"{enrollment_count}, "
                        f"{experiment_group.title()}"
                    ),
                )

                save_log_false_rate_plot(
                    group_dataframe,
                    enrollment_output_dir
                    / (
                        f"{experiment_group}_"
                        "false_reconstruction_log"
                    ),
                    title=(
                        "False Reconstruction Rate "
                        f"— Enrollment "
                        f"{enrollment_count}, "
                        f"{experiment_group.title()}"
                    ),
                )

                print()
                print(
                    "Enrollment:",
                    enrollment_count,
                    "Group:",
                    experiment_group,
                )

                print(
                    group_dataframe[
                        [
                            "correction_capability",
                            "genuine_reconstruction_rate",
                            "false_reconstruction_rate",
                            "genuine_success_count",
                            "false_reconstruction_count",
                        ]
                    ].to_string(index=False)
                )

        result_dataframe = (
            pd.DataFrame.from_records(
                all_records
            )
        )

        combined_csv_path = (
            output_root
            / "correction_radius_all.csv"
        )

        result_dataframe.to_csv(
            combined_csv_path,
            index=False,
        )

        result_records = []

        for record in all_records:
            result_records.append(
                {
                    key: (
                        value.item()
                        if isinstance(
                            value,
                            np.generic,
                        )
                        else value
                    )
                    for key, value
                    in record.items()
                }
            )

        summary_payload = {
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "analysis_type": (
                "ideal_bounded_distance_proxy"
            ),
            "interpretation": (
                "A trial is counted as reconstructable "
                "when its Hamming distance is no greater "
                "than correction capability t. This is not "
                "yet an actual BCH decoding experiment."
            ),
            "template_length": template_length,
            "correction_capabilities": (
                correction_capabilities
            ),
            "results": result_records,
        }

        summary_path = (
            output_root
            / "correction_radius_summary.json"
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
            "Saved combined CSV:",
            combined_csv_path,
        )
        print(
            "Saved summary:",
            summary_path,
        )

        return 0

    except KeyError as exc:
        print(
            f"Missing configuration key: {exc}",
            file=sys.stderr,
        )
        return 2

    except CorrectionRadiusEvaluationError as exc:
        print(
            f"Correction-radius evaluation failed: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Correction-radius evaluation interrupted",
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