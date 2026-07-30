from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fuzzy_did.data import (
    EmbeddingRepository,
    EmbeddingRepositoryError,
)
from fuzzy_did.features import (
    FeatureSelectionConfig,
    FeatureSelectionError,
    FeatureSelectionSet,
    build_feature_selection_set,
)


class FeatureSelectionBuildError(RuntimeError):
    """Raised when feature-selection artifact generation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build subject-specific feature selections from "
            "identity-level Facenet512 embedding caches."
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

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--stability-weight",
        type=float,
        default=None,
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise FeatureSelectionBuildError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise FeatureSelectionBuildError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise FeatureSelectionBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise FeatureSelectionBuildError(
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


def save_feature_selection_set(
    selection_set: FeatureSelectionSet,
    output_path: Path,
    config: FeatureSelectionConfig,
) -> None:
    atomic_save_npz(
        output_path,
        enrollment_count=np.asarray(
            [selection_set.enrollment_count],
            dtype=np.int16,
        ),
        top_k=np.asarray(
            [selection_set.top_k],
            dtype=np.int16,
        ),
        stability_weight=np.asarray(
            [selection_set.stability_weight],
            dtype=np.float32,
        ),
        epsilon=np.asarray(
            [config.epsilon],
            dtype=np.float32,
        ),
        mad_scale=np.asarray(
            [config.mad_scale],
            dtype=np.float32,
        ),
        use_leave_one_out_background=np.asarray(
            [config.use_leave_one_out_background],
            dtype=np.bool_,
        ),
        identity_ids=selection_set.identity_ids,
        experiment_groups=(
            selection_set.experiment_groups
        ),
        subject_centers=(
            selection_set.subject_centers
        ),
        subject_dispersions=(
            selection_set.subject_dispersions
        ),
        reference_centers=(
            selection_set.reference_centers
        ),
        reference_dispersions=(
            selection_set.reference_dispersions
        ),
        stability_raw_scores=(
            selection_set.stability_raw_scores
        ),
        discrimination_raw_scores=(
            selection_set.discrimination_raw_scores
        ),
        stability_scores=(
            selection_set.stability_scores
        ),
        discrimination_scores=(
            selection_set.discrimination_scores
        ),
        combined_scores=(
            selection_set.combined_scores
        ),
        selected_dimensions=(
            selection_set.selected_dimensions
        ),
        selected_scores=(
            selection_set.selected_scores
        ),
        global_background_center=(
            selection_set.global_background_center
        ),
        global_background_dispersion=(
            selection_set.global_background_dispersion
        ),
        background_identity_ids=(
            selection_set.background_identity_ids
        ),
    )


def calculate_selection_overlap(
    selected_dimensions: np.ndarray,
) -> dict[str, float]:
    """
    Calculate simple identity-level selection diversity statistics.
    """

    identity_count, top_k = (
        selected_dimensions.shape
    )

    dimension_counts = np.bincount(
        selected_dimensions.reshape(-1),
        minlength=512,
    )

    selected_dimension_count = int(
        (dimension_counts > 0).sum()
    )

    maximum_selection_frequency = int(
        dimension_counts.max()
    )

    mean_selection_frequency = float(
        dimension_counts[
            dimension_counts > 0
        ].mean()
    )

    # Mean pairwise Jaccard similarity.
    jaccard_values: list[float] = []

    selection_sets = [
        set(row.astype(int).tolist())
        for row in selected_dimensions
    ]

    for first_index in range(
        identity_count
    ):
        first = selection_sets[first_index]

        for second_index in range(
            first_index + 1,
            identity_count,
        ):
            second = selection_sets[second_index]

            union_size = len(
                first | second
            )

            if union_size == 0:
                continue

            jaccard_values.append(
                len(first & second)
                / union_size
            )

    return {
        "unique_selected_dimension_count": (
            selected_dimension_count
        ),
        "maximum_dimension_selection_frequency": (
            maximum_selection_frequency
        ),
        "mean_selected_dimension_frequency": (
            mean_selection_frequency
        ),
        "mean_pairwise_jaccard": (
            float(np.mean(jaccard_values))
            if jaccard_values
            else 0.0
        ),
        "top_k": top_k,
    }


def summarize_feature_selection(
    selection_set: FeatureSelectionSet,
    output_path: Path,
) -> dict[str, Any]:
    group_counts: dict[str, int] = {}

    unique_groups, counts = np.unique(
        selection_set.experiment_groups,
        return_counts=True,
    )

    for group, count in zip(
        unique_groups,
        counts,
        strict=True,
    ):
        group_counts[str(group)] = int(count)

    overlap = calculate_selection_overlap(
        selection_set.selected_dimensions
    )

    return {
        "output_path": str(
            output_path.resolve()
        ),
        "enrollment_count": int(
            selection_set.enrollment_count
        ),
        "identity_count": int(
            selection_set.identity_count
        ),
        "embedding_dimension": int(
            selection_set.embedding_dimension
        ),
        "top_k": int(
            selection_set.top_k
        ),
        "stability_weight": float(
            selection_set.stability_weight
        ),
        "experiment_group_counts": (
            group_counts
        ),
        "selected_dimension_shape": list(
            selection_set
            .selected_dimensions.shape
        ),
        "selected_score_min": float(
            selection_set.selected_scores.min()
        ),
        "selected_score_max": float(
            selection_set.selected_scores.max()
        ),
        "selected_score_mean": float(
            selection_set.selected_scores.mean()
        ),
        "subject_dispersion_mean": float(
            selection_set.subject_dispersions.mean()
        ),
        "background_dispersion_mean": float(
            selection_set
            .global_background_dispersion
            .mean()
        ),
        "all_finite": bool(
            np.isfinite(
                selection_set.combined_scores
            ).all()
        ),
        **overlap,
    }


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(
            args.config
        )

        data_config = config["data"]
        extraction_config = config.get(
            "extraction",
            {},
        )
        feature_config = config[
            "feature_selection"
        ]

        cache_dir = Path(
            data_config["cache_dir"]
        ).expanduser().resolve()

        output_dir = Path(
            feature_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value in feature_config[
                "enrollment_counts"
            ]
        ]

        top_k = (
            args.top_k
            if args.top_k is not None
            else int(
                feature_config.get(
                    "top_k",
                    128,
                )
            )
        )

        stability_weight = (
            args.stability_weight
            if args.stability_weight
            is not None
            else float(
                feature_config.get(
                    "stability_weight",
                    0.5,
                )
            )
        )

        selection_config = FeatureSelectionConfig(
            top_k=top_k,
            stability_weight=(
                stability_weight
            ),
            epsilon=float(
                feature_config.get(
                    "epsilon",
                    1e-6,
                )
            ),
            mad_scale=float(
                feature_config.get(
                    "mad_scale",
                    1.4826,
                )
            ),
            use_leave_one_out_background=bool(
                feature_config.get(
                    "use_leave_one_out_background",
                    True,
                )
            ),
        )

        expected_dimension = int(
            extraction_config.get(
                "expected_embedding_dimension",
                512,
            )
        )

        repository = EmbeddingRepository(
            cache_root=cache_dir,
            expected_embedding_dimension=(
                expected_dimension
            ),
        )

        if len(repository) != 540:
            raise FeatureSelectionBuildError(
                f"Expected 540 identity caches, "
                f"found {len(repository)}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        alpha_label = int(
            round(
                selection_config
                .stability_weight
                * 100
            )
        )

        summaries: list[
            dict[str, Any]
        ] = []

        print(
            "Cache identities:",
            len(repository),
        )
        print(
            "Enrollment counts:",
            enrollment_counts,
        )
        print(
            "Top-K:",
            selection_config.top_k,
        )
        print(
            "Stability weight:",
            selection_config.stability_weight,
        )
        print(
            "Leave-one-out background:",
            selection_config
            .use_leave_one_out_background,
        )

        for enrollment_count in enrollment_counts:
            filename = (
                f"enrollment_"
                f"{enrollment_count:02d}_"
                f"top{selection_config.top_k}_"
                f"alpha{alpha_label:03d}.npz"
            )

            output_path = (
                output_dir / filename
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                raise FeatureSelectionBuildError(
                    f"Output already exists: "
                    f"{output_path}. "
                    "Use --overwrite."
                )

            print()
            print(
                "Building enrollment_count=",
                enrollment_count,
            )

            selection_set = (
                build_feature_selection_set(
                    repository=repository,
                    enrollment_count=(
                        enrollment_count
                    ),
                    config=selection_config,
                )
            )

            save_feature_selection_set(
                selection_set=selection_set,
                output_path=output_path,
                config=selection_config,
            )

            summary = (
                summarize_feature_selection(
                    selection_set=selection_set,
                    output_path=output_path,
                )
            )

            summaries.append(
                summary
            )

            print(
                "  selected dimensions:",
                selection_set
                .selected_dimensions.shape,
            )
            print(
                "  mean selected score:",
                f"{summary['selected_score_mean']:.6f}",
            )
            print(
                "  mean pairwise Jaccard:",
                f"{summary['mean_pairwise_jaccard']:.6f}",
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
            "cache_dir": str(
                cache_dir
            ),
            "output_dir": str(
                output_dir
            ),
            "identity_count": len(
                repository
            ),
            "configuration": {
                "top_k": (
                    selection_config.top_k
                ),
                "stability_weight": (
                    selection_config
                    .stability_weight
                ),
                "discrimination_weight": (
                    1.0
                    - selection_config
                    .stability_weight
                ),
                "epsilon": (
                    selection_config.epsilon
                ),
                "mad_scale": (
                    selection_config.mad_scale
                ),
                "use_leave_one_out_background": (
                    selection_config
                    .use_leave_one_out_background
                ),
            },
            "feature_selections": summaries,
        }

        summary_path = (
            output_dir
            / "feature_selection_summary.json"
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
        FeatureSelectionBuildError,
        FeatureSelectionError,
        EmbeddingRepositoryError,
    ) as exc:
        print(
            f"Feature-selection generation failed: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Feature-selection generation interrupted",
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