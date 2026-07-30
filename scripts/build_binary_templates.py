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

from fuzzy_did.binarization import (
    BinarizationError,
    BinaryTemplateSet,
    MedianBinarizerConfig,
    build_binary_template_set,
)


class BinaryTemplateBuildError(RuntimeError):
    """Raised when binary-template artifacts cannot be generated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build median-binarized enrollment templates from "
            "robust-normalized selected Facenet512 dimensions."
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
        raise BinaryTemplateBuildError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise BinaryTemplateBuildError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise BinaryTemplateBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise BinaryTemplateBuildError(
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


def save_binary_template_set(
    template_set: BinaryTemplateSet,
    output_path: Path,
    config: MedianBinarizerConfig,
) -> None:
    atomic_save_npz(
        output_path,
        enrollment_count=np.asarray(
            [template_set.enrollment_count],
            dtype=np.int16,
        ),
        top_k=np.asarray(
            [template_set.top_k],
            dtype=np.int16,
        ),
        threshold=np.asarray(
            [template_set.threshold],
            dtype=np.float32,
        ),
        positive_when_greater=np.asarray(
            [config.positive_when_greater],
            dtype=np.bool_,
        ),
        bitorder=np.asarray(
            [template_set.bitorder],
            dtype=np.str_,
        ),
        identity_ids=template_set.identity_ids,
        experiment_groups=(
            template_set.experiment_groups
        ),
        selected_dimensions=(
            template_set.selected_dimensions
        ),
        normalized_selected_values=(
            template_set.normalized_selected_values
        ),
        binary_templates=(
            template_set.binary_templates
        ),
        packed_binary_templates=(
            template_set.packed_binary_templates
        ),
        one_counts=template_set.one_counts,
    )


def calculate_group_statistics(
    template_set: BinaryTemplateSet,
) -> dict[str, dict[str, Any]]:
    groups = (
        template_set.experiment_groups
        .astype(str)
    )

    statistics: dict[str, dict[str, Any]] = {}

    for group in sorted(set(groups)):
        mask = groups == group

        group_templates = (
            template_set.binary_templates[
                mask
            ]
        )

        group_one_counts = (
            template_set.one_counts[
                mask
            ]
        )

        statistics[group] = {
            "identity_count": int(mask.sum()),
            "global_one_ratio": float(
                group_templates.mean()
            ),
            "one_count_min": int(
                group_one_counts.min()
            ),
            "one_count_max": int(
                group_one_counts.max()
            ),
            "one_count_mean": float(
                group_one_counts.mean()
            ),
            "one_count_std": float(
                group_one_counts.std()
            ),
        }

    return statistics


def summarize_binary_templates(
    template_set: BinaryTemplateSet,
    output_path: Path,
) -> dict[str, Any]:
    one_counts = template_set.one_counts

    per_bit_one_ratios = (
        template_set.binary_templates
        .mean(axis=0)
    )

    constant_zero_bits = int(
        (per_bit_one_ratios == 0.0).sum()
    )

    constant_one_bits = int(
        (per_bit_one_ratios == 1.0).sum()
    )

    near_balanced_bits = int(
        (
            (per_bit_one_ratios >= 0.4)
            & (per_bit_one_ratios <= 0.6)
        ).sum()
    )

    unique_template_count = int(
        np.unique(
            template_set.packed_binary_templates,
            axis=0,
        ).shape[0]
    )

    return {
        "output_path": str(
            output_path.resolve()
        ),
        "enrollment_count": int(
            template_set.enrollment_count
        ),
        "identity_count": int(
            template_set.identity_count
        ),
        "template_length_bits": int(
            template_set.template_length
        ),
        "packed_length_bytes": int(
            template_set.packed_length_bytes
        ),
        "threshold": float(
            template_set.threshold
        ),
        "bitorder": template_set.bitorder,
        "global_one_ratio": float(
            template_set.global_one_ratio
        ),
        "one_count_min": int(
            one_counts.min()
        ),
        "one_count_max": int(
            one_counts.max()
        ),
        "one_count_mean": float(
            one_counts.mean()
        ),
        "one_count_std": float(
            one_counts.std()
        ),
        "constant_zero_bit_count": (
            constant_zero_bits
        ),
        "constant_one_bit_count": (
            constant_one_bits
        ),
        "near_balanced_bit_count": (
            near_balanced_bits
        ),
        "unique_template_count": (
            unique_template_count
        ),
        "duplicate_template_count": int(
            template_set.identity_count
            - unique_template_count
        ),
        "group_statistics": (
            calculate_group_statistics(
                template_set
            )
        ),
        "all_binary": bool(
            np.isin(
                template_set.binary_templates,
                [0, 1],
            ).all()
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

        binarization_config = config[
            "binarization"
        ]

        normalization_dir = Path(
            normalization_config["output_dir"]
        ).expanduser().resolve()

        output_dir = Path(
            binarization_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value in binarization_config[
                "enrollment_counts"
            ]
        ]

        top_k = int(
            feature_config.get(
                "top_k",
                128,
            )
        )

        binary_config = MedianBinarizerConfig(
            threshold=float(
                binarization_config.get(
                    "threshold",
                    0.0,
                )
            ),
            positive_when_greater=bool(
                binarization_config.get(
                    "positive_when_greater",
                    True,
                )
            ),
            bitorder=str(
                binarization_config.get(
                    "bitorder",
                    "big",
                )
            ),
        )

        binary_config.validate()

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
            "Threshold:",
            binary_config.threshold,
        )
        print(
            "Positive rule:",
            (
                "value > threshold"
                if binary_config.positive_when_greater
                else "value >= threshold"
            ),
        )
        print(
            "Bit order:",
            binary_config.bitorder,
        )

        for enrollment_count in enrollment_counts:
            normalization_path = (
                normalization_dir
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}_"
                    f"top{top_k}_robust.npz"
                )
            )

            if not normalization_path.is_file():
                raise BinaryTemplateBuildError(
                    "Normalization artifact does not exist: "
                    f"{normalization_path}"
                )

            output_path = (
                output_dir
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}_"
                    f"top{top_k}_binary.npz"
                )
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                raise BinaryTemplateBuildError(
                    f"Output already exists: {output_path}. "
                    "Use --overwrite."
                )

            print()
            print(
                f"Building enrollment_count="
                f"{enrollment_count}"
            )

            with np.load(
                normalization_path,
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

                selected_dimensions = (
                    data["selected_dimensions"]
                    .astype(np.int16)
                )

                normalized_selected = (
                    data[
                        "normalized_selected_centers"
                    ]
                    .astype(np.float32)
                )

            template_set = build_binary_template_set(
                identity_ids=identity_ids,
                experiment_groups=(
                    experiment_groups
                ),
                selected_dimensions=(
                    selected_dimensions
                ),
                normalized_selected_values=(
                    normalized_selected
                ),
                enrollment_count=(
                    enrollment_count
                ),
                top_k=top_k,
                config=binary_config,
            )

            save_binary_template_set(
                template_set=template_set,
                output_path=output_path,
                config=binary_config,
            )

            summary = summarize_binary_templates(
                template_set=template_set,
                output_path=output_path,
            )

            summaries.append(summary)

            print(
                "  binary shape:",
                template_set.binary_templates.shape,
            )
            print(
                "  packed shape:",
                template_set
                .packed_binary_templates
                .shape,
            )
            print(
                "  global one ratio:",
                f"{summary['global_one_ratio']:.6f}",
            )
            print(
                "  one-count mean/std:",
                (
                    summary["one_count_mean"],
                    summary["one_count_std"],
                ),
            )
            print(
                "  unique templates:",
                summary["unique_template_count"],
            )
            print(
                "  saved:",
                output_path,
            )

        summary_payload = {
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "normalization_dir": str(
                normalization_dir
            ),
            "output_dir": str(
                output_dir
            ),
            "configuration": {
                "threshold": (
                    binary_config.threshold
                ),
                "positive_when_greater": (
                    binary_config
                    .positive_when_greater
                ),
                "bitorder": (
                    binary_config.bitorder
                ),
                "threshold_interpretation": (
                    "zero in robust-normalized space "
                    "equals the background median in "
                    "the original embedding space"
                ),
            },
            "binary_templates": summaries,
        }

        summary_path = (
            output_dir
            / "binary_template_summary.json"
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
        BinaryTemplateBuildError,
        BinarizationError,
    ) as exc:
        print(
            f"Binary-template generation failed: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Binary-template generation interrupted",
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