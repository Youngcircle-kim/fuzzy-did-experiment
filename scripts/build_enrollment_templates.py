from __future__ import annotations

import argparse
import json
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
from fuzzy_did.templates import (
    EnrollmentTemplateError,
    EnrollmentTemplateSet,
    build_enrollment_template_set,
)


class TemplateBuildError(RuntimeError):
    """Raised when enrollment template generation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build identity-level representative enrollment "
            "embeddings from Facenet512 caches."
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

    parser.add_argument(
        "--aggregation",
        choices=[
            "mean",
            "median",
        ],
        default=None,
    )

    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help=(
            "Override the configuration and apply "
            "L2 normalization after aggregation."
        ),
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise TemplateBuildError(
            f"Configuration does not exist: "
            f"{resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise TemplateBuildError(
            f"Failed to read configuration: "
            f"{resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise TemplateBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise TemplateBuildError(
            "Configuration root must be a mapping"
        )

    return loaded


def save_template_set(
    template_set: EnrollmentTemplateSet,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        enrollment_count=np.asarray(
            [template_set.enrollment_count],
            dtype=np.int16,
        ),
        aggregation=np.asarray(
            [template_set.aggregation],
            dtype=np.str_,
        ),
        l2_normalized=np.asarray(
            [template_set.l2_normalized],
            dtype=np.bool_,
        ),
        identity_ids=(
            template_set.identity_ids
        ),
        experiment_groups=(
            template_set.experiment_groups
        ),
        representative_embeddings=(
            template_set
            .representative_embeddings
        ),
        enrollment_counts=(
            template_set.enrollment_counts
        ),
        candidate_ranks=(
            template_set.candidate_ranks
        ),
    )


def summarize_template_set(
    template_set: EnrollmentTemplateSet,
    output_path: Path,
) -> dict[str, Any]:
    embeddings = (
        template_set.representative_embeddings
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    group_counts: dict[str, int] = {}

    unique_groups, counts = np.unique(
        template_set.experiment_groups,
        return_counts=True,
    )

    for group, count in zip(
        unique_groups,
        counts,
        strict=True,
    ):
        group_counts[str(group)] = int(count)

    return {
        "output_path": str(
            output_path.resolve()
        ),
        "enrollment_count": (
            template_set.enrollment_count
        ),
        "aggregation": (
            template_set.aggregation
        ),
        "l2_normalized": (
            template_set.l2_normalized
        ),
        "identity_count": (
            template_set.identity_count
        ),
        "embedding_dimension": (
            template_set.embedding_dimension
        ),
        "experiment_group_counts": (
            group_counts
        ),
        "embedding_value_min": float(
            embeddings.min()
        ),
        "embedding_value_max": float(
            embeddings.max()
        ),
        "embedding_value_mean": float(
            embeddings.mean()
        ),
        "l2_norm_min": float(
            norms.min()
        ),
        "l2_norm_max": float(
            norms.max()
        ),
        "l2_norm_mean": float(
            norms.mean()
        ),
        "all_finite": bool(
            np.isfinite(embeddings).all()
        ),
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
        template_config = config[
            "templates"
        ]

        cache_dir = Path(
            data_config["cache_dir"]
        ).expanduser().resolve()

        output_dir = Path(
            template_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value
            in template_config[
                "enrollment_counts"
            ]
        ]

        aggregation = (
            args.aggregation
            if args.aggregation is not None
            else str(
                template_config.get(
                    "aggregation",
                    "median",
                )
            )
        )

        l2_normalize = bool(
            template_config.get(
                "l2_normalize",
                False,
            )
        )

        if args.l2_normalize:
            l2_normalize = True

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

        print(
            "Cache identities:",
            len(repository),
        )
        print(
            "Enrollment counts:",
            enrollment_counts,
        )
        print(
            "Aggregation:",
            aggregation,
        )
        print(
            "L2 normalize:",
            l2_normalize,
        )

        if len(repository) != 540:
            raise TemplateBuildError(
                f"Expected 540 identity cache files, "
                f"found {len(repository)}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        template_summaries: list[
            dict[str, Any]
        ] = []

        normalization_suffix = (
            "l2"
            if l2_normalize
            else "raw"
        )

        for enrollment_count in enrollment_counts:
            filename = (
                f"enrollment_"
                f"{enrollment_count:02d}_"
                f"{aggregation}_"
                f"{normalization_suffix}.npz"
            )

            output_path = (
                output_dir / filename
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                raise TemplateBuildError(
                    f"Output already exists: "
                    f"{output_path}. "
                    "Use --overwrite."
                )

            print(
                f"Building enrollment_count="
                f"{enrollment_count}"
            )

            template_set = (
                build_enrollment_template_set(
                    repository=repository,
                    enrollment_count=(
                        enrollment_count
                    ),
                    aggregation=aggregation,
                    l2_normalize=(
                        l2_normalize
                    ),
                )
            )

            save_template_set(
                template_set=template_set,
                output_path=output_path,
            )

            summary = summarize_template_set(
                template_set=template_set,
                output_path=output_path,
            )

            template_summaries.append(
                summary
            )

            print(
                f"  saved={output_path}"
            )
            print(
                f"  shape="
                f"{template_set.representative_embeddings.shape}"
            )
            print(
                f"  mean norm="
                f"{summary['l2_norm_mean']:.6f}"
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
            "identity_count": len(
                repository
            ),
            "aggregation": aggregation,
            "l2_normalize": (
                l2_normalize
            ),
            "templates": (
                template_summaries
            ),
        }

        summary_path = (
            output_dir
            / "enrollment_template_summary.json"
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
        TemplateBuildError,
        EmbeddingRepositoryError,
        EnrollmentTemplateError,
    ) as exc:
        print(
            f"Template generation failed: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Template generation interrupted",
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