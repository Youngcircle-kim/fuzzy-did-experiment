from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fuzzy_did.normalization import (
    RobustNormalizationError,
    RobustNormalizationSet,
    RobustScalerConfig,
    build_robust_normalization_set,
)


class NormalizationBuildError(RuntimeError):
    """Raised when robust-normalization artifacts cannot be built."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build background-fitted robust normalization artifacts "
            "for subject-specific selected Facenet512 dimensions."
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


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise NormalizationBuildError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise NormalizationBuildError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise NormalizationBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise NormalizationBuildError(
            "Configuration root must be a mapping"
        )

    return loaded


def atomic_save_npz(
    output_path: Path,
    **arrays: Any,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}_",
        suffix=".npz",
        dir=output_path.parent,
    )

    os.close(descriptor)

    temporary_path = Path(
        temporary_name
    )

    try:
        np.savez_compressed(
            temporary_path,
            **arrays,
        )

        temporary_path.replace(
            output_path
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )


def save_normalization_set(
    normalization_set: RobustNormalizationSet,
    output_path: Path,
    scaler_config: RobustScalerConfig,
) -> None:
    state = normalization_set.scaler_state

    atomic_save_npz(
        output_path,
        enrollment_count=np.asarray(
            [normalization_set.enrollment_count],
            dtype=np.int16,
        ),
        top_k=np.asarray(
            [normalization_set.top_k],
            dtype=np.int16,
        ),
        lower_quantile=np.asarray(
            [scaler_config.lower_quantile],
            dtype=np.float32,
        ),
        upper_quantile=np.asarray(
            [scaler_config.upper_quantile],
            dtype=np.float32,
        ),
        scale_floor=np.asarray(
            [scaler_config.scale_floor],
            dtype=np.float32,
        ),
        identity_ids=(
            normalization_set.identity_ids
        ),
        experiment_groups=(
            normalization_set.experiment_groups
        ),
        selected_dimensions=(
            normalization_set.selected_dimensions
        ),
        raw_subject_centers=(
            normalization_set.raw_subject_centers
        ),
        normalized_subject_centers=(
            normalization_set.normalized_subject_centers
        ),
        raw_selected_centers=(
            normalization_set.raw_selected_centers
        ),
        normalized_selected_centers=(
            normalization_set.normalized_selected_centers
        ),
        global_center=state.center,
        global_scale=state.scale,
        global_raw_scale=state.raw_scale,
        global_q1=state.q1,
        global_q3=state.q3,
        floored_dimensions=(
            state.floored_dimensions
        ),
    )


def summarize_normalization(
    normalization_set: RobustNormalizationSet,
    output_path: Path,
) -> dict[str, Any]:
    state = normalization_set.scaler_state

    normalized = (
        normalization_set
        .normalized_selected_centers
    )

    groups = (
        normalization_set.experiment_groups
        .astype(str)
    )

    group_counts = {
        group: int((groups == group).sum())
        for group in sorted(set(groups))
    }

    return {
        "output_path": str(
            output_path.resolve()
        ),
        "enrollment_count": int(
            normalization_set.enrollment_count
        ),
        "identity_count": int(
            normalization_set.identity_count
        ),
        "embedding_dimension": int(
            normalization_set.embedding_dimension
        ),
        "top_k": int(
            normalization_set.top_k
        ),
        "experiment_group_counts": (
            group_counts
        ),
        "floored_dimension_count": int(
            state.floored_dimensions.sum()
        ),
        "raw_scale_min": float(
            state.raw_scale.min()
        ),
        "raw_scale_max": float(
            state.raw_scale.max()
        ),
        "scale_min": float(
            state.scale.min()
        ),
        "scale_max": float(
            state.scale.max()
        ),
        "normalized_selected_min": float(
            normalized.min()
        ),
        "normalized_selected_max": float(
            normalized.max()
        ),
        "normalized_selected_mean": float(
            normalized.mean()
        ),
        "normalized_selected_std": float(
            normalized.std()
        ),
        "all_finite": bool(
            np.isfinite(normalized).all()
        ),
    }


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(
            args.config
        )

        feature_config = config[
            "feature_selection"
        ]
        normalization_config = config[
            "normalization"
        ]

        feature_dir = Path(
            feature_config["output_dir"]
        ).expanduser().resolve()

        output_dir = Path(
            normalization_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value in normalization_config[
                "enrollment_counts"
            ]
        ]

        top_k = int(
            feature_config.get(
                "top_k",
                128,
            )
        )

        stability_weight = float(
            feature_config.get(
                "stability_weight",
                0.5,
            )
        )

        alpha_label = int(
            round(stability_weight * 100)
        )

        scaler_config = RobustScalerConfig(
            lower_quantile=float(
                normalization_config.get(
                    "lower_quantile",
                    25.0,
                )
            ),
            upper_quantile=float(
                normalization_config.get(
                    "upper_quantile",
                    75.0,
                )
            ),
            scale_floor=float(
                normalization_config.get(
                    "scale_floor",
                    1e-6,
                )
            ),
        )

        scaler_config.validate()

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summaries: list[
            dict[str, Any]
        ] = []

        print(
            "Enrollment counts:",
            enrollment_counts,
        )
        print("Top-K:", top_k)
        print(
            "Quantile range:",
            (
                scaler_config.lower_quantile,
                scaler_config.upper_quantile,
            ),
        )
        print(
            "Scale floor:",
            scaler_config.scale_floor,
        )

        for enrollment_count in enrollment_counts:
            feature_path = (
                feature_dir
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}_"
                    f"top{top_k}_"
                    f"alpha{alpha_label:03d}.npz"
                )
            )

            if not feature_path.is_file():
                raise NormalizationBuildError(
                    f"Feature-selection artifact does not exist: "
                    f"{feature_path}"
                )

            output_path = (
                output_dir
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}_"
                    f"top{top_k}_robust.npz"
                )
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                raise NormalizationBuildError(
                    f"Output already exists: {output_path}. "
                    "Use --overwrite."
                )

            print()
            print(
                f"Building enrollment_count="
                f"{enrollment_count}"
            )

            with np.load(
                feature_path,
                allow_pickle=False,
            ) as data:
                identity_ids = (
                    data["identity_ids"]
                    .astype(np.str_)
                )
                experiment_groups = (
                    data["experiment_groups"]
                    .astype(np.str_)
                )
                subject_centers = (
                    data["subject_centers"]
                    .astype(np.float32)
                )
                selected_dimensions = (
                    data["selected_dimensions"]
                    .astype(np.int16)
                )

            normalization_set = (
                build_robust_normalization_set(
                    identity_ids=identity_ids,
                    experiment_groups=(
                        experiment_groups
                    ),
                    subject_centers=(
                        subject_centers
                    ),
                    selected_dimensions=(
                        selected_dimensions
                    ),
                    enrollment_count=(
                        enrollment_count
                    ),
                    top_k=top_k,
                    config=scaler_config,
                )
            )

            save_normalization_set(
                normalization_set=(
                    normalization_set
                ),
                output_path=output_path,
                scaler_config=scaler_config,
            )

            summary = summarize_normalization(
                normalization_set=(
                    normalization_set
                ),
                output_path=output_path,
            )

            summaries.append(summary)

            print(
                "  normalized selected shape:",
                normalization_set
                .normalized_selected_centers
                .shape,
            )
            print(
                "  floored dimensions:",
                summary[
                    "floored_dimension_count"
                ],
            )
            print(
                "  mean/std:",
                (
                    summary[
                        "normalized_selected_mean"
                    ],
                    summary[
                        "normalized_selected_std"
                    ],
                ),
            )
            print("  saved:", output_path)

        summary_payload = {
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "feature_dir": str(
                feature_dir
            ),
            "output_dir": str(
                output_dir
            ),
            "configuration": {
                "lower_quantile": (
                    scaler_config.lower_quantile
                ),
                "upper_quantile": (
                    scaler_config.upper_quantile
                ),
                "scale_floor": (
                    scaler_config.scale_floor
                ),
                "fit_group": "background",
            },
            "normalizations": summaries,
        }

        summary_path = (
            output_dir
            / "robust_normalization_summary.json"
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
        NormalizationBuildError,
        RobustNormalizationError,
    ) as exc:
        print(
            f"Normalization generation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Normalization generation interrupted",
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