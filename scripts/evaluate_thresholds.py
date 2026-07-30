from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fuzzy_did.evaluation import (
    ThresholdEvaluationError,
    evaluate_fixed_threshold,
    sweep_thresholds,
)


class ThresholdBuildError(RuntimeError):
    """Raised when threshold evaluation cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Hamming thresholds using development trials "
            "and evaluate fixed thresholds on evaluation trials."
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise ThresholdBuildError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise ThresholdBuildError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ThresholdBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ThresholdBuildError(
            "Configuration root must be a mapping"
        )

    return loaded


def read_trial_distances(
    path: Path,
) -> tuple[pd.Series, pd.Series]:
    if not path.is_file():
        raise ThresholdBuildError(
            f"Trial file does not exist: {path}"
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
    ]

    impostor = dataframe.loc[
        dataframe["trial_type"] == "impostor",
        "hamming_distance",
    ]

    if genuine.empty:
        raise ThresholdBuildError(
            f"No genuine trials in {path}"
        )

    if impostor.empty:
        raise ThresholdBuildError(
            f"No impostor trials in {path}"
        )

    return genuine, impostor


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(args.config)

        feature_config = config[
            "feature_selection"
        ]

        trial_config = config[
            "hamming_trials"
        ]

        trial_root = Path(
            trial_config["output_dir"]
        ).expanduser().resolve()

        output_root = (
            trial_root / "threshold_evaluation"
        )

        enrollment_counts = [
            int(value)
            for value in trial_config[
                "enrollment_counts"
            ]
        ]

        template_length = int(
            feature_config.get(
                "top_k",
                128,
            )
        )

        if output_root.exists():
            existing_files = list(
                output_root.glob("*")
            )

            if (
                existing_files
                and not args.overwrite
            ):
                raise ThresholdBuildError(
                    f"Output directory is not empty: "
                    f"{output_root}. "
                    "Use --overwrite."
                )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        summaries: list[dict[str, Any]] = []

        print(
            "Enrollment counts:",
            enrollment_counts,
        )
        print(
            "Template length:",
            template_length,
        )
        print(
            "Threshold policy: distance <= threshold"
            " means accept"
        )

        for enrollment_count in enrollment_counts:
            enrollment_dir = (
                trial_root
                / f"enrollment_{enrollment_count:02d}"
            )

            development_path = (
                enrollment_dir
                / "development.parquet"
            )

            evaluation_path = (
                enrollment_dir
                / "evaluation.parquet"
            )

            (
                development_genuine,
                development_impostor,
            ) = read_trial_distances(
                development_path
            )

            development_sweep = (
                sweep_thresholds(
                    development_genuine.to_numpy(),
                    development_impostor.to_numpy(),
                    template_length=template_length,
                )
            )

            selected_threshold = (
                development_sweep.eer_threshold
            )

            development_fixed = (
                evaluate_fixed_threshold(
                    development_genuine.to_numpy(),
                    development_impostor.to_numpy(),
                    threshold=selected_threshold,
                    template_length=template_length,
                )
            )

            (
                evaluation_genuine,
                evaluation_impostor,
            ) = read_trial_distances(
                evaluation_path
            )

            # Evaluation EER은 분석용으로만 계산.
            # 실제 최종 평가 threshold는 development에서 고정.
            evaluation_sweep = (
                sweep_thresholds(
                    evaluation_genuine.to_numpy(),
                    evaluation_impostor.to_numpy(),
                    template_length=template_length,
                )
            )

            evaluation_fixed = (
                evaluate_fixed_threshold(
                    evaluation_genuine.to_numpy(),
                    evaluation_impostor.to_numpy(),
                    threshold=selected_threshold,
                    template_length=template_length,
                )
            )

            sweep_dataframe = pd.DataFrame(
                {
                    "threshold": (
                        development_sweep.thresholds
                    ),
                    "development_far": (
                        development_sweep
                        .false_accept_rates
                    ),
                    "development_frr": (
                        development_sweep
                        .false_reject_rates
                    ),
                    "development_tar": (
                        development_sweep
                        .true_accept_rates
                    ),
                    "development_trr": (
                        development_sweep
                        .true_reject_rates
                    ),
                    "development_ber": (
                        development_sweep
                        .balanced_error_rates
                    ),
                    "evaluation_far": (
                        evaluation_sweep
                        .false_accept_rates
                    ),
                    "evaluation_frr": (
                        evaluation_sweep
                        .false_reject_rates
                    ),
                    "evaluation_tar": (
                        evaluation_sweep
                        .true_accept_rates
                    ),
                    "evaluation_trr": (
                        evaluation_sweep
                        .true_reject_rates
                    ),
                    "evaluation_ber": (
                        evaluation_sweep
                        .balanced_error_rates
                    ),
                }
            )

            sweep_path = (
                output_root
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}_"
                    "threshold_sweep.csv"
                )
            )

            sweep_dataframe.to_csv(
                sweep_path,
                index=False,
            )

            summary = {
                "enrollment_count": (
                    enrollment_count
                ),
                "template_length": (
                    template_length
                ),
                "acceptance_rule": (
                    "hamming_distance <= threshold"
                ),
                "selected_threshold_source": (
                    "development_eer"
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "development": {
                    "eer_threshold": (
                        development_sweep
                        .eer_threshold
                    ),
                    "eer": (
                        development_sweep.eer
                    ),
                    "eer_far": (
                        development_sweep.eer_far
                    ),
                    "eer_frr": (
                        development_sweep.eer_frr
                    ),
                    "fixed_threshold_metrics": (
                        asdict(
                            development_fixed
                        )
                    ),
                },
                "evaluation": {
                    "fixed_threshold": (
                        selected_threshold
                    ),
                    "fixed_threshold_metrics": (
                        asdict(
                            evaluation_fixed
                        )
                    ),
                    "oracle_eer_threshold": (
                        evaluation_sweep
                        .eer_threshold
                    ),
                    "oracle_eer": (
                        evaluation_sweep.eer
                    ),
                    "oracle_eer_note": (
                        "Analysis only; not used to "
                        "select the final threshold."
                    ),
                },
                "threshold_sweep_path": str(
                    sweep_path.resolve()
                ),
            }

            summaries.append(summary)

            print()
            print(
                "Enrollment:",
                enrollment_count,
            )
            print(
                " Development EER threshold:",
                selected_threshold,
            )
            print(
                " Development EER:",
                f"{development_sweep.eer:.6f}",
            )
            print(
                " Evaluation FAR:",
                f"{evaluation_fixed.false_accept_rate:.6f}",
            )
            print(
                " Evaluation FRR:",
                f"{evaluation_fixed.false_reject_rate:.6f}",
            )
            print(
                " Evaluation BER:",
                f"{evaluation_fixed.balanced_error_rate:.6f}",
            )
            print(
                " Evaluation TAR:",
                f"{evaluation_fixed.true_accept_rate:.6f}",
            )

        summary_payload = {
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "trial_root": str(
                trial_root
            ),
            "threshold_selection": {
                "source_group": "development",
                "criterion": (
                    "minimum absolute FAR-FRR gap; "
                    "balanced error used as tie-break"
                ),
                "acceptance_rule": (
                    "hamming_distance <= threshold"
                ),
            },
            "results": summaries,
        }

        summary_path = (
            output_root
            / "threshold_evaluation_summary.json"
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

    except (
        ThresholdBuildError,
        ThresholdEvaluationError,
    ) as exc:
        print(
            f"Threshold evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Threshold evaluation interrupted",
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