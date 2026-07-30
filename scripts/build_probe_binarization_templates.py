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
from tqdm import tqdm

from fuzzy_did.binarization import (
    MedianBinarizerConfig,
)
from fuzzy_did.data import (
    EmbeddingRepository,
    EmbeddingRepositoryError,
)
from fuzzy_did.normalization import (
    RobustScalerState,
)
from fuzzy_did.probes import (
    ProbeBinaryTemplateSet,
    ProbeTemplateError,
    build_probe_binary_template_set,
)


class ProbeTemplateBuildError(RuntimeError):
    """Raised when probe-template artifact generation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build identity-level probe binary templates using "
            "enrollment-specific robust scalers and selected dimensions."
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
        "--resume",
        action="store_true",
        help="Skip valid identity-level probe caches.",
    )

    parser.add_argument(
        "--max-identities",
        type=int,
        default=None,
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise ProbeTemplateBuildError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise ProbeTemplateBuildError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ProbeTemplateBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ProbeTemplateBuildError(
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
    temporary_path = Path(temporary_name)

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


def save_probe_template_set(
    template_set: ProbeBinaryTemplateSet,
    output_path: Path,
    *,
    bitorder: str,
) -> None:
    atomic_save_npz(
        output_path,
        identity_id=np.asarray(
            [template_set.identity_id],
            dtype=np.str_,
        ),
        experiment_group=np.asarray(
            [template_set.experiment_group],
            dtype=np.str_,
        ),
        enrollment_count=np.asarray(
            [template_set.enrollment_count],
            dtype=np.int16,
        ),
        bitorder=np.asarray(
            [bitorder],
            dtype=np.str_,
        ),
        image_ids=template_set.image_ids,
        relative_paths=template_set.relative_paths,
        selected_dimensions=(
            template_set.selected_dimensions
        ),
        binary_templates=(
            template_set.binary_templates
        ),
        packed_binary_templates=(
            template_set.packed_binary_templates
        ),
        one_counts=template_set.one_counts,
    )


def is_valid_probe_cache(
    path: Path,
    *,
    identity_id: str,
    expected_template_length: int,
    expected_bitorder: str,
) -> bool:
    if not path.is_file():
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            cached_identity = str(
                data["identity_id"][0]
            )

            binary = data[
                "binary_templates"
            ]

            packed = data[
                "packed_binary_templates"
            ]

            bitorder = str(
                data["bitorder"][0]
            )

            image_ids = data[
                "image_ids"
            ]

    except Exception:
        return False

    if cached_identity != identity_id:
        return False

    if bitorder != expected_bitorder:
        return False

    if binary.ndim != 2:
        return False

    if binary.shape[1] != expected_template_length:
        return False

    if packed.shape != (
        binary.shape[0],
        (expected_template_length + 7) // 8,
    ):
        return False

    if len(image_ids) != binary.shape[0]:
        return False

    if not np.isin(binary, [0, 1]).all():
        return False

    restored = np.unpackbits(
        packed,
        axis=1,
        count=expected_template_length,
        bitorder=bitorder,
    )

    return bool(
        np.array_equal(
            restored,
            binary,
        )
    )


def load_enrollment_artifacts(
    *,
    enrollment_count: int,
    top_k: int,
    normalization_dir: Path,
    binary_dir: Path,
) -> dict[str, Any]:
    normalization_path = (
        normalization_dir
        / (
            f"enrollment_"
            f"{enrollment_count:02d}_"
            f"top{top_k}_robust.npz"
        )
    )

    binary_path = (
        binary_dir
        / (
            f"enrollment_"
            f"{enrollment_count:02d}_"
            f"top{top_k}_binary.npz"
        )
    )

    if not normalization_path.is_file():
        raise ProbeTemplateBuildError(
            f"Normalization artifact missing: {normalization_path}"
        )

    if not binary_path.is_file():
        raise ProbeTemplateBuildError(
            f"Enrollment binary artifact missing: {binary_path}"
        )

    with np.load(
        normalization_path,
        allow_pickle=False,
    ) as data:
        normalization_identity_ids = (
            data["identity_ids"]
            .astype(str)
        )

        normalization_groups = (
            data["experiment_groups"]
            .astype(str)
        )

        selected_dimensions = (
            data["selected_dimensions"]
            .astype(np.int16)
        )

        scaler_state = RobustScalerState(
            center=data[
                "global_center"
            ].astype(np.float32),
            scale=data[
                "global_scale"
            ].astype(np.float32),
            raw_scale=data[
                "global_raw_scale"
            ].astype(np.float32),
            q1=data[
                "global_q1"
            ].astype(np.float32),
            q3=data[
                "global_q3"
            ].astype(np.float32),
            floored_dimensions=data[
                "floored_dimensions"
            ].astype(np.bool_),
        )

    with np.load(
        binary_path,
        allow_pickle=False,
    ) as data:
        binary_identity_ids = (
            data["identity_ids"]
            .astype(str)
        )

        bitorder = str(
            data["bitorder"][0]
        )

        threshold = float(
            data["threshold"][0]
        )

        positive_when_greater = bool(
            data["positive_when_greater"][0]
        )

    if not np.array_equal(
        normalization_identity_ids,
        binary_identity_ids,
    ):
        raise ProbeTemplateBuildError(
            "Identity order differs between normalization "
            "and enrollment binary artifacts"
        )

    if len(np.unique(normalization_identity_ids)) != len(
        normalization_identity_ids
    ):
        raise ProbeTemplateBuildError(
            "Duplicate identity IDs in enrollment artifacts"
        )

    return {
        "identity_ids": normalization_identity_ids,
        "experiment_groups": normalization_groups,
        "selected_dimensions": selected_dimensions,
        "scaler_state": scaler_state,
        "binarizer_config": MedianBinarizerConfig(
            threshold=threshold,
            positive_when_greater=positive_when_greater,
            bitorder=bitorder,
        ),
    }


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(args.config)

        data_config = config["data"]
        feature_config = config["feature_selection"]
        normalization_config = config["normalization"]
        binarization_config = config["binarization"]
        probe_config = config["probe_templates"]

        if not bool(
            probe_config.get(
                "exclude_all_enrollment_candidates",
                True,
            )
        ):
            raise ProbeTemplateBuildError(
                "This implementation requires all enrollment "
                "candidates to be excluded from probe evaluation"
            )

        cache_dir = Path(
            data_config["cache_dir"]
        ).expanduser().resolve()

        normalization_dir = Path(
            normalization_config["output_dir"]
        ).expanduser().resolve()

        binary_dir = Path(
            binarization_config["output_dir"]
        ).expanduser().resolve()

        output_root = Path(
            probe_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value in probe_config[
                "enrollment_counts"
            ]
        ]

        top_k = int(
            feature_config.get(
                "top_k",
                128,
            )
        )

        repository = EmbeddingRepository(
            cache_root=cache_dir,
            expected_embedding_dimension=512,
        )

        identity_ids = repository.identity_ids()

        if args.max_identities is not None:
            if args.max_identities <= 0:
                raise ProbeTemplateBuildError(
                    "max-identities must be positive"
                )

            identity_ids = identity_ids[
                :args.max_identities
            ]

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_summaries: list[dict[str, Any]] = []

        print("Cache identities:", len(repository))
        print("Identities to process:", len(identity_ids))
        print("Enrollment counts:", enrollment_counts)
        print("Top-K:", top_k)
        print(
            "Probe policy: exclude all candidate ranks 1-10"
        )

        for enrollment_count in enrollment_counts:
            print()
            print(
                f"Building enrollment_count={enrollment_count}"
            )

            artifacts = load_enrollment_artifacts(
                enrollment_count=enrollment_count,
                top_k=top_k,
                normalization_dir=normalization_dir,
                binary_dir=binary_dir,
            )

            artifact_identity_ids = artifacts[
                "identity_ids"
            ]

            identity_to_index = {
                identity_id: index
                for index, identity_id in enumerate(
                    artifact_identity_ids
                )
            }

            output_dir = (
                output_root
                / f"enrollment_{enrollment_count:02d}"
            )

            identity_output_dir = (
                output_dir / "identities"
            )

            identity_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            manifest_records: list[dict[str, Any]] = []

            completed_count = 0
            skipped_count = 0
            total_probe_count = 0

            for identity_id in tqdm(
                identity_ids,
                desc=f"Enrollment {enrollment_count}",
                unit="identity",
            ):
                if identity_id not in identity_to_index:
                    raise ProbeTemplateBuildError(
                        f"Identity missing from enrollment artifacts: "
                        f"{identity_id}"
                    )

                artifact_index = identity_to_index[
                    identity_id
                ]

                output_path = (
                    identity_output_dir
                    / f"{identity_id}.npz"
                )

                bitorder = artifacts[
                    "binarizer_config"
                ].bitorder

                if (
                    args.resume
                    and is_valid_probe_cache(
                        output_path,
                        identity_id=identity_id,
                        expected_template_length=top_k,
                        expected_bitorder=bitorder,
                    )
                ):
                    with np.load(
                        output_path,
                        allow_pickle=False,
                    ) as data:
                        cached_probe_count = int(
                            data["binary_templates"].shape[0]
                        )

                    skipped_count += 1
                    total_probe_count += cached_probe_count

                    manifest_records.append(
                        {
                            "identity_id": identity_id,
                            "status": "skipped",
                            "probe_count": cached_probe_count,
                            "output_path": str(output_path),
                        }
                    )

                    continue

                if (
                    output_path.exists()
                    and not args.overwrite
                    and not args.resume
                ):
                    raise ProbeTemplateBuildError(
                        f"Output already exists: {output_path}. "
                        "Use --overwrite or --resume."
                    )

                cache = repository.load(
                    identity_id
                )

                expected_group = str(
                    artifacts[
                        "experiment_groups"
                    ][artifact_index]
                )

                if cache.experiment_group != expected_group:
                    raise ProbeTemplateBuildError(
                        f"{identity_id}: experiment-group mismatch"
                    )

                selected_dimensions = artifacts[
                    "selected_dimensions"
                ][artifact_index]

                template_set = (
                    build_probe_binary_template_set(
                        cache=cache,
                        enrollment_count=enrollment_count,
                        selected_dimensions=selected_dimensions,
                        scaler_state=artifacts[
                            "scaler_state"
                        ],
                        binarizer_config=artifacts[
                            "binarizer_config"
                        ],
                    )
                )

                save_probe_template_set(
                    template_set,
                    output_path,
                    bitorder=bitorder,
                )

                completed_count += 1
                total_probe_count += (
                    template_set.probe_count
                )

                manifest_records.append(
                    {
                        "identity_id": identity_id,
                        "experiment_group": (
                            template_set.experiment_group
                        ),
                        "status": "complete",
                        "probe_count": (
                            template_set.probe_count
                        ),
                        "template_length": (
                            template_set.template_length
                        ),
                        "packed_length_bytes": (
                            template_set.packed_length_bytes
                        ),
                        "one_ratio": float(
                            template_set.binary_templates.mean()
                        ),
                        "output_path": str(output_path),
                    }
                )

            group_probe_counts: dict[str, int] = {}

            for record in manifest_records:
                if "experiment_group" not in record:
                    continue

                group = str(
                    record["experiment_group"]
                )

                group_probe_counts[group] = (
                    group_probe_counts.get(group, 0)
                    + int(record["probe_count"])
                )

            manifest_payload = {
                "created_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "enrollment_count": enrollment_count,
                "top_k": top_k,
                "identity_count": len(
                    manifest_records
                ),
                "completed_identity_count": (
                    completed_count
                ),
                "skipped_identity_count": (
                    skipped_count
                ),
                "total_probe_count": (
                    total_probe_count
                ),
                "probe_policy": (
                    "sample_role=probe and "
                    "enrollment_candidate_rank<0"
                ),
                "experiment_group_probe_counts": (
                    group_probe_counts
                ),
                "records": manifest_records,
            }

            manifest_path = (
                output_dir / "manifest.json"
            )

            with manifest_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    manifest_payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            summary = {
                "enrollment_count": (
                    enrollment_count
                ),
                "identity_count": len(
                    manifest_records
                ),
                "completed_identity_count": (
                    completed_count
                ),
                "skipped_identity_count": (
                    skipped_count
                ),
                "total_probe_count": (
                    total_probe_count
                ),
                "manifest_path": str(
                    manifest_path
                ),
            }

            all_summaries.append(summary)

            print(
                "  completed identities:",
                completed_count,
            )
            print(
                "  skipped identities:",
                skipped_count,
            )
            print(
                "  total probes:",
                total_probe_count,
            )
            print(
                "  manifest:",
                manifest_path,
            )

        summary_payload = {
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "cache_dir": str(cache_dir),
            "output_root": str(output_root),
            "probe_policy": {
                "sample_role": "probe",
                "candidate_rank_requirement": (
                    "enrollment_candidate_rank < 0"
                ),
                "excluded_failed_embeddings": 16,
            },
            "probe_templates": all_summaries,
        }

        summary_path = (
            output_root
            / "probe_binary_template_summary.json"
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
        print("Saved summary:", summary_path)

        return 0

    except KeyError as exc:
        print(
            f"Missing configuration key: {exc}",
            file=sys.stderr,
        )
        return 2

    except (
        ProbeTemplateBuildError,
        ProbeTemplateError,
        EmbeddingRepositoryError,
    ) as exc:
        print(
            f"Probe-template generation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Probe-template generation interrupted",
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