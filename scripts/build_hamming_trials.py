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
from tqdm import tqdm

from fuzzy_did.binarization import (
    MedianBinarizerConfig,
)
from fuzzy_did.data import (
    EmbeddingRepository,
    EmbeddingRepositoryError,
    IdentityEmbeddingCache,
)
from fuzzy_did.evaluation import (
    HammingTrialError,
    hamming_distance_batch,
    transform_probe_embeddings_for_claimant,
)
from fuzzy_did.normalization import (
    RobustScalerState,
)


class HammingTrialBuildError(RuntimeError):
    """Raised when Hamming trial artifacts cannot be built."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build genuine and sampled-impostor Hamming trials "
            "for subject-specific binary face templates."
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
        "--max-claimants",
        type=int,
        default=None,
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise HammingTrialBuildError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise HammingTrialBuildError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise HammingTrialBuildError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise HammingTrialBuildError(
            "Configuration root must be a mapping"
        )

    return loaded


def true_probe_mask(
    cache: IdentityEmbeddingCache,
) -> npt.NDArray[np.bool_]:
    """
    Use only non-candidate probe images.
    """

    return (
        (
            cache.enrollment_candidate_ranks
            < 0
        )
        & (
            cache.sample_roles.astype(str)
            == "probe"
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
        raise HammingTrialBuildError(
            f"Missing normalization artifact: "
            f"{normalization_path}"
        )

    if not binary_path.is_file():
        raise HammingTrialBuildError(
            f"Missing binary artifact: {binary_path}"
        )

    with np.load(
        normalization_path,
        allow_pickle=False,
    ) as data:
        normalization_identity_ids = (
            data["identity_ids"]
            .astype(str)
        )

        experiment_groups = (
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

        enrollment_templates = (
            data["binary_templates"]
            .astype(np.uint8)
        )

        threshold = float(
            data["threshold"][0]
        )

        positive_when_greater = bool(
            data["positive_when_greater"][0]
        )

        bitorder = str(
            data["bitorder"][0]
        )

    if not np.array_equal(
        normalization_identity_ids,
        binary_identity_ids,
    ):
        raise HammingTrialBuildError(
            "Identity order differs between normalization "
            "and binary enrollment artifacts"
        )

    return {
        "identity_ids": normalization_identity_ids,
        "experiment_groups": experiment_groups,
        "selected_dimensions": selected_dimensions,
        "enrollment_templates": enrollment_templates,
        "scaler_state": scaler_state,
        "binarizer_config": MedianBinarizerConfig(
            threshold=threshold,
            positive_when_greater=positive_when_greater,
            bitorder=bitorder,
        ),
    }


def build_group_probe_pool(
    *,
    repository: EmbeddingRepository,
    group_identity_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Load valid probe embeddings for every identity in one group.
    """

    pool: dict[str, dict[str, Any]] = {}

    for identity_id in tqdm(
        group_identity_ids,
        desc="Loading probe pool",
        unit="identity",
        leave=False,
    ):
        cache = repository.load(
            identity_id
        )

        mask = true_probe_mask(cache)

        if not mask.any():
            raise HammingTrialBuildError(
                f"{identity_id}: no valid probes"
            )

        pool[identity_id] = {
            "embeddings": (
                cache.embeddings[
                    mask
                ].astype(np.float32)
            ),
            "image_ids": (
                cache.image_ids[
                    mask
                ].astype(str)
            ),
        }

    return pool


def sample_impostor_probes(
    *,
    claimant_identity_id: str,
    group_identity_ids: list[str],
    probe_pool: dict[str, dict[str, Any]],
    trial_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample impostor probes uniformly by identity, then by probe image.

    Sampling is with replacement. This keeps exactly the configured
    number of impostor trials per claimant.
    """

    impostor_identity_ids = [
        identity_id
        for identity_id in group_identity_ids
        if identity_id != claimant_identity_id
    ]

    if not impostor_identity_ids:
        raise HammingTrialBuildError(
            f"{claimant_identity_id}: no impostor identities available"
        )

    sampled_identity_positions = rng.integers(
        low=0,
        high=len(impostor_identity_ids),
        size=trial_count,
    )

    sampled_probe_embeddings: list[np.ndarray] = []
    sampled_probe_identity_ids: list[str] = []
    sampled_probe_image_ids: list[str] = []

    for identity_position in sampled_identity_positions:
        probe_identity_id = impostor_identity_ids[
            int(identity_position)
        ]

        identity_pool = probe_pool[
            probe_identity_id
        ]

        probe_position = int(
            rng.integers(
                low=0,
                high=len(
                    identity_pool["embeddings"]
                ),
            )
        )

        sampled_probe_embeddings.append(
            identity_pool["embeddings"][
                probe_position
            ]
        )

        sampled_probe_identity_ids.append(
            probe_identity_id
        )

        sampled_probe_image_ids.append(
            str(
                identity_pool["image_ids"][
                    probe_position
                ]
            )
        )

    return (
        np.stack(
            sampled_probe_embeddings,
            axis=0,
        ).astype(np.float32),
        np.asarray(
            sampled_probe_identity_ids,
            dtype=np.str_,
        ),
        np.asarray(
            sampled_probe_image_ids,
            dtype=np.str_,
        ),
    )


def append_trial_records(
    records: list[dict[str, Any]],
    *,
    trial_type: str,
    experiment_group: str,
    claimant_identity_id: str,
    probe_identity_ids: np.ndarray,
    probe_image_ids: np.ndarray,
    distances: np.ndarray,
    normalized_distances: np.ndarray,
    is_match: bool,
) -> None:
    for (
        probe_identity_id,
        probe_image_id,
        distance,
        normalized_distance,
    ) in zip(
        probe_identity_ids,
        probe_image_ids,
        distances,
        normalized_distances,
        strict=True,
    ):
        records.append(
            {
                "trial_type": trial_type,
                "experiment_group": (
                    experiment_group
                ),
                "claimant_identity_id": (
                    claimant_identity_id
                ),
                "probe_identity_id": str(
                    probe_identity_id
                ),
                "probe_image_id": str(
                    probe_image_id
                ),
                "hamming_distance": int(
                    distance
                ),
                "normalized_hamming_distance": float(
                    normalized_distance
                ),
                "is_match": bool(
                    is_match
                ),
            }
        )


def build_trials_for_group(
    *,
    repository: EmbeddingRepository,
    group_name: str,
    group_identity_ids: list[str],
    artifact_identity_to_index: dict[str, int],
    artifacts: dict[str, Any],
    impostor_trials_per_claimant: int,
    random_seed: int,
    max_claimants: int | None,
) -> pd.DataFrame:
    claimant_ids = list(
        group_identity_ids
    )

    if max_claimants is not None:
        claimant_ids = claimant_ids[
            :max_claimants
        ]

    probe_pool = build_group_probe_pool(
        repository=repository,
        group_identity_ids=group_identity_ids,
    )

    records: list[dict[str, Any]] = []

    for claimant_identity_id in tqdm(
        claimant_ids,
        desc=f"{group_name} claimants",
        unit="identity",
    ):
        artifact_index = (
            artifact_identity_to_index[
                claimant_identity_id
            ]
        )

        enrollment_template = artifacts[
            "enrollment_templates"
        ][artifact_index]

        selected_dimensions = artifacts[
            "selected_dimensions"
        ][artifact_index]

        # Genuine trials
        genuine_pool = probe_pool[
            claimant_identity_id
        ]

        genuine_binary = (
            transform_probe_embeddings_for_claimant(
                probe_embeddings=(
                    genuine_pool["embeddings"]
                ),
                claimant_selected_dimensions=(
                    selected_dimensions
                ),
                scaler_state=artifacts[
                    "scaler_state"
                ],
                binarizer_config=artifacts[
                    "binarizer_config"
                ],
            )
        )

        genuine_result = (
            hamming_distance_batch(
                enrollment_template,
                genuine_binary,
            )
        )

        genuine_probe_count = len(
            genuine_pool["embeddings"]
        )

        append_trial_records(
            records,
            trial_type="genuine",
            experiment_group=group_name,
            claimant_identity_id=(
                claimant_identity_id
            ),
            probe_identity_ids=np.full(
                genuine_probe_count,
                claimant_identity_id,
                dtype=np.str_,
            ),
            probe_image_ids=(
                genuine_pool["image_ids"]
            ),
            distances=(
                genuine_result.hamming_distances
            ),
            normalized_distances=(
                genuine_result.normalized_distances
            ),
            is_match=True,
        )

        # Deterministic claimant-specific RNG
        claimant_seed = (
            random_seed
            + int(
                claimant_identity_id
                .replace("n", "")
            )
            * 1009
        ) % (2**32)

        rng = np.random.default_rng(
            claimant_seed
        )

        (
            impostor_embeddings,
            impostor_identity_ids,
            impostor_image_ids,
        ) = sample_impostor_probes(
            claimant_identity_id=(
                claimant_identity_id
            ),
            group_identity_ids=(
                group_identity_ids
            ),
            probe_pool=probe_pool,
            trial_count=(
                impostor_trials_per_claimant
            ),
            rng=rng,
        )

        impostor_binary = (
            transform_probe_embeddings_for_claimant(
                probe_embeddings=(
                    impostor_embeddings
                ),
                claimant_selected_dimensions=(
                    selected_dimensions
                ),
                scaler_state=artifacts[
                    "scaler_state"
                ],
                binarizer_config=artifacts[
                    "binarizer_config"
                ],
            )
        )

        impostor_result = (
            hamming_distance_batch(
                enrollment_template,
                impostor_binary,
            )
        )

        append_trial_records(
            records,
            trial_type="impostor",
            experiment_group=group_name,
            claimant_identity_id=(
                claimant_identity_id
            ),
            probe_identity_ids=(
                impostor_identity_ids
            ),
            probe_image_ids=(
                impostor_image_ids
            ),
            distances=(
                impostor_result.hamming_distances
            ),
            normalized_distances=(
                impostor_result.normalized_distances
            ),
            is_match=False,
        )

    dataframe = pd.DataFrame.from_records(
        records
    )

    if dataframe.empty:
        raise HammingTrialBuildError(
            f"No trials generated for group {group_name}"
        )

    return dataframe


def summarize_trials(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    genuine = dataframe[
        dataframe["trial_type"]
        == "genuine"
    ]

    impostor = dataframe[
        dataframe["trial_type"]
        == "impostor"
    ]

    return {
        "trial_count": int(
            len(dataframe)
        ),
        "genuine_trial_count": int(
            len(genuine)
        ),
        "impostor_trial_count": int(
            len(impostor)
        ),
        "genuine_distance_mean": float(
            genuine[
                "hamming_distance"
            ].mean()
        ),
        "genuine_distance_std": float(
            genuine[
                "hamming_distance"
            ].std()
        ),
        "genuine_distance_min": int(
            genuine[
                "hamming_distance"
            ].min()
        ),
        "genuine_distance_max": int(
            genuine[
                "hamming_distance"
            ].max()
        ),
        "impostor_distance_mean": float(
            impostor[
                "hamming_distance"
            ].mean()
        ),
        "impostor_distance_std": float(
            impostor[
                "hamming_distance"
            ].std()
        ),
        "impostor_distance_min": int(
            impostor[
                "hamming_distance"
            ].min()
        ),
        "impostor_distance_max": int(
            impostor[
                "hamming_distance"
            ].max()
        ),
        "distance_margin": float(
            impostor[
                "hamming_distance"
            ].mean()
            - genuine[
                "hamming_distance"
            ].mean()
        ),
    }


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(
            args.config
        )

        data_config = config["data"]
        feature_config = config[
            "feature_selection"
        ]
        normalization_config = config[
            "normalization"
        ]
        binarization_config = config[
            "binarization"
        ]
        trial_config = config[
            "hamming_trials"
        ]

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
            trial_config["output_dir"]
        ).expanduser().resolve()

        enrollment_counts = [
            int(value)
            for value in trial_config[
                "enrollment_counts"
            ]
        ]

        experiment_groups = [
            str(value)
            for value in trial_config[
                "experiment_groups"
            ]
        ]

        top_k = int(
            feature_config.get(
                "top_k",
                128,
            )
        )

        impostor_trials_per_claimant = int(
            trial_config.get(
                "impostor_trials_per_claimant",
                1000,
            )
        )

        random_seed = int(
            trial_config.get(
                "random_seed",
                42,
            )
        )

        if impostor_trials_per_claimant <= 0:
            raise HammingTrialBuildError(
                "impostor_trials_per_claimant must be positive"
            )

        repository = EmbeddingRepository(
            cache_root=cache_dir,
            expected_embedding_dimension=512,
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_summaries: list[dict[str, Any]] = []

        print("Cache identities:", len(repository))
        print("Enrollment counts:", enrollment_counts)
        print("Groups:", experiment_groups)
        print(
            "Impostor trials per claimant:",
            impostor_trials_per_claimant,
        )
        print("Random seed:", random_seed)

        for enrollment_count in enrollment_counts:
            print()
            print(
                f"Enrollment count={enrollment_count}"
            )

            artifacts = load_enrollment_artifacts(
                enrollment_count=enrollment_count,
                top_k=top_k,
                normalization_dir=(
                    normalization_dir
                ),
                binary_dir=binary_dir,
            )

            artifact_identity_to_index = {
                identity_id: index
                for index, identity_id in enumerate(
                    artifacts["identity_ids"]
                )
            }

            enrollment_output_dir = (
                output_root
                / f"enrollment_{enrollment_count:02d}"
            )

            enrollment_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            group_summaries: dict[
                str,
                dict[str, Any],
            ] = {}

            for group_name in experiment_groups:
                group_identity_ids = [
                    identity_id
                    for identity_id, group
                    in zip(
                        artifacts["identity_ids"],
                        artifacts[
                            "experiment_groups"
                        ],
                        strict=True,
                    )
                    if group == group_name
                ]

                if not group_identity_ids:
                    raise HammingTrialBuildError(
                        f"No identities found for "
                        f"group {group_name}"
                    )

                output_path = (
                    enrollment_output_dir
                    / f"{group_name}.parquet"
                )

                if (
                    output_path.exists()
                    and not args.overwrite
                ):
                    raise HammingTrialBuildError(
                        f"Output already exists: "
                        f"{output_path}. "
                        "Use --overwrite."
                    )

                dataframe = (
                    build_trials_for_group(
                        repository=repository,
                        group_name=group_name,
                        group_identity_ids=(
                            group_identity_ids
                        ),
                        artifact_identity_to_index=(
                            artifact_identity_to_index
                        ),
                        artifacts=artifacts,
                        impostor_trials_per_claimant=(
                            impostor_trials_per_claimant
                        ),
                        random_seed=random_seed,
                        max_claimants=(
                            args.max_claimants
                        ),
                    )
                )

                dataframe.to_parquet(
                    output_path,
                    index=False,
                    engine="pyarrow",
                    compression="snappy",
                )

                group_summary = summarize_trials(
                    dataframe
                )

                group_summary[
                    "output_path"
                ] = str(
                    output_path.resolve()
                )

                group_summary[
                    "claimant_identity_count"
                ] = int(
                    dataframe[
                        "claimant_identity_id"
                    ].nunique()
                )

                group_summaries[
                    group_name
                ] = group_summary

                print()
                print(" Group:", group_name)
                print(
                    "  claimants:",
                    group_summary[
                        "claimant_identity_count"
                    ],
                )
                print(
                    "  genuine:",
                    group_summary[
                        "genuine_trial_count"
                    ],
                )
                print(
                    "  impostor:",
                    group_summary[
                        "impostor_trial_count"
                    ],
                )
                print(
                    "  genuine mean:",
                    f"{group_summary['genuine_distance_mean']:.4f}",
                )
                print(
                    "  impostor mean:",
                    f"{group_summary['impostor_distance_mean']:.4f}",
                )
                print(
                    "  margin:",
                    f"{group_summary['distance_margin']:.4f}",
                )

            enrollment_summary = {
                "enrollment_count": (
                    enrollment_count
                ),
                "template_length": top_k,
                "groups": group_summaries,
            }

            with (
                enrollment_output_dir
                / "summary.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    enrollment_summary,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            all_summaries.append(
                enrollment_summary
            )

        summary_payload = {
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "template_length": top_k,
            "impostor_sampling": {
                "trials_per_claimant": (
                    impostor_trials_per_claimant
                ),
                "sampling_unit": (
                    "identity uniformly, then "
                    "probe uniformly"
                ),
                "with_replacement": True,
                "random_seed": random_seed,
            },
            "hamming_trials": all_summaries,
        }

        summary_path = (
            output_root
            / "hamming_trial_summary.json"
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
        HammingTrialBuildError,
        HammingTrialError,
        EmbeddingRepositoryError,
    ) as exc:
        print(
            f"Hamming-trial generation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Hamming-trial generation interrupted",
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