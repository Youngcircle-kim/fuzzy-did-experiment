from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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


IDENTITY_PATTERN = re.compile(
    r"^n\d{6}$"
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


def parse_identity_from_image_id(
    image_id: str,
) -> str:
    """
    Parse a VGGFace2 identity from a cached image identifier.

    Expected example:
        train__n000002__0001_01
    """

    value = str(image_id)

    parts = value.split("__")

    if len(parts) < 3:
        raise HammingTrialBuildError(
            f"Unexpected probe image ID format: {value}"
        )

    identity_id = parts[1]

    if IDENTITY_PATTERN.fullmatch(
        identity_id
    ) is None:
        raise HammingTrialBuildError(
            "Could not parse a valid VGGFace2 identity "
            f"from probe image ID: {value}"
        )

    return identity_id


def validate_identity_id(
    identity_id: str,
    *,
    field_name: str,
) -> str:
    value = str(identity_id)

    if IDENTITY_PATTERN.fullmatch(
        value
    ) is None:
        raise HammingTrialBuildError(
            f"Invalid {field_name}: {value}"
        )

    return value


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

    if selected_dimensions.ndim != 2:
        raise HammingTrialBuildError(
            "selected_dimensions must have shape "
            f"[identity, dimension], got "
            f"{selected_dimensions.shape}"
        )

    if enrollment_templates.ndim != 2:
        raise HammingTrialBuildError(
            "binary_templates must have shape "
            f"[identity, bit], got "
            f"{enrollment_templates.shape}"
        )

    identity_count = len(
        normalization_identity_ids
    )

    if len(experiment_groups) != identity_count:
        raise HammingTrialBuildError(
            "experiment_groups length differs from "
            "identity count"
        )

    if selected_dimensions.shape[0] != identity_count:
        raise HammingTrialBuildError(
            "selected_dimensions identity count differs "
            "from artifact identity count"
        )

    if enrollment_templates.shape[0] != identity_count:
        raise HammingTrialBuildError(
            "binary template identity count differs from "
            "artifact identity count"
        )

    if enrollment_templates.shape[1] != top_k:
        raise HammingTrialBuildError(
            "Binary template length differs from configured top_k: "
            f"template_length={enrollment_templates.shape[1]}, "
            f"top_k={top_k}"
        )

    for identity_id in normalization_identity_ids:
        validate_identity_id(
            identity_id,
            field_name="artifact identity ID",
        )

    return {
        "identity_ids": normalization_identity_ids,
        "experiment_groups": experiment_groups,
        "selected_dimensions": selected_dimensions,
        "enrollment_templates": enrollment_templates,
        "scaler_state": scaler_state,
        "binarizer_config": MedianBinarizerConfig(
            threshold=threshold,
            positive_when_greater=(
                positive_when_greater
            ),
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
        validate_identity_id(
            identity_id,
            field_name="group identity ID",
        )

        cache = repository.load(
            identity_id
        )

        mask = true_probe_mask(
            cache
        )

        if not mask.any():
            raise HammingTrialBuildError(
                f"{identity_id}: no valid probes"
            )

        embeddings = (
            cache.embeddings[
                mask
            ].astype(
                np.float32,
                copy=False,
            )
        )

        image_ids = np.asarray(
            cache.image_ids[
                mask
            ],
            dtype=object,
        )

        if embeddings.ndim != 2:
            raise HammingTrialBuildError(
                f"{identity_id}: invalid embedding shape "
                f"{embeddings.shape}"
            )

        if len(embeddings) != len(image_ids):
            raise HammingTrialBuildError(
                f"{identity_id}: embedding/image ID count "
                "mismatch"
            )

        for image_id in image_ids:
            parsed_identity = (
                parse_identity_from_image_id(
                    str(image_id)
                )
            )

            if parsed_identity != identity_id:
                raise HammingTrialBuildError(
                    "Probe image identity mismatch: "
                    f"cache_identity={identity_id}, "
                    f"parsed_identity={parsed_identity}, "
                    f"image_id={image_id}"
                )

        pool[identity_id] = {
            "embeddings": embeddings,
            "image_ids": image_ids,
        }

    return pool


def sample_impostor_probes(
    *,
    claimant_identity_id: str,
    group_identity_ids: list[str],
    probe_pool: dict[str, dict[str, Any]],
    trial_count: int,
    rng: np.random.Generator,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
]:
    """
    Sample impostor probes uniformly by identity, then by probe image.

    Sampling is with replacement. This keeps exactly the configured
    number of impostor trials per claimant.
    """

    validate_identity_id(
        claimant_identity_id,
        field_name="claimant identity ID",
    )

    if trial_count <= 0:
        raise HammingTrialBuildError(
            "Impostor trial count must be positive"
        )

    impostor_identity_ids = [
        identity_id
        for identity_id in group_identity_ids
        if identity_id != claimant_identity_id
    ]

    if not impostor_identity_ids:
        raise HammingTrialBuildError(
            f"{claimant_identity_id}: "
            "no impostor identities available"
        )

    sampled_identity_positions = rng.integers(
        low=0,
        high=len(
            impostor_identity_ids
        ),
        size=trial_count,
    )

    sampled_probe_embeddings: list[
        npt.NDArray[np.float32]
    ] = []

    sampled_probe_identity_ids: list[str] = []
    sampled_probe_image_ids: list[str] = []

    for identity_position in (
        sampled_identity_positions
    ):
        probe_identity_id = (
            impostor_identity_ids[
                int(identity_position)
            ]
        )

        identity_pool = probe_pool[
            probe_identity_id
        ]

        probe_count = len(
            identity_pool["embeddings"]
        )

        if probe_count <= 0:
            raise HammingTrialBuildError(
                f"{probe_identity_id}: "
                "empty impostor probe pool"
            )

        probe_position = int(
            rng.integers(
                low=0,
                high=probe_count,
            )
        )

        probe_embedding = (
            identity_pool["embeddings"][
                probe_position
            ]
        )

        probe_image_id = str(
            identity_pool["image_ids"][
                probe_position
            ]
        )

        parsed_identity_id = (
            parse_identity_from_image_id(
                probe_image_id
            )
        )

        if (
            parsed_identity_id
            != probe_identity_id
        ):
            raise HammingTrialBuildError(
                "Sampled impostor identity mismatch: "
                f"expected={probe_identity_id}, "
                f"parsed={parsed_identity_id}, "
                f"image_id={probe_image_id}"
            )

        sampled_probe_embeddings.append(
            probe_embedding
        )

        sampled_probe_identity_ids.append(
            probe_identity_id
        )

        sampled_probe_image_ids.append(
            probe_image_id
        )

    if not sampled_probe_embeddings:
        raise HammingTrialBuildError(
            f"{claimant_identity_id}: "
            "no impostor probes sampled"
        )

    return (
        np.stack(
            sampled_probe_embeddings,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        np.asarray(
            sampled_probe_identity_ids,
            dtype=object,
        ),
        np.asarray(
            sampled_probe_image_ids,
            dtype=object,
        ),
    )


def append_trial_records(
    records: list[dict[str, Any]],
    *,
    trial_type: str,
    experiment_group: str,
    claimant_identity_id: str,
    probe_identity_ids: npt.ArrayLike,
    probe_image_ids: npt.ArrayLike,
    distances: npt.ArrayLike,
    normalized_distances: npt.ArrayLike,
    is_match: bool,
) -> None:
    """
    Append validated Hamming-trial rows.

    String metadata is converted to Python objects to prevent
    fixed-width NumPy string truncation.
    """

    if trial_type not in {
        "genuine",
        "impostor",
    }:
        raise HammingTrialBuildError(
            f"Unsupported trial type: {trial_type}"
        )

    claimant_identity_id = (
        validate_identity_id(
            claimant_identity_id,
            field_name="claimant identity ID",
        )
    )

    probe_identity_array = np.asarray(
        probe_identity_ids,
        dtype=object,
    )

    probe_image_array = np.asarray(
        probe_image_ids,
        dtype=object,
    )

    distance_array = np.asarray(
        distances,
        dtype=np.int16,
    )

    normalized_array = np.asarray(
        normalized_distances,
        dtype=np.float32,
    )

    arrays = {
        "probe_identity_ids": (
            probe_identity_array
        ),
        "probe_image_ids": (
            probe_image_array
        ),
        "distances": distance_array,
        "normalized_distances": (
            normalized_array
        ),
    }

    for name, array in arrays.items():
        if array.ndim != 1:
            raise HammingTrialBuildError(
                f"{name} must be one-dimensional, "
                f"got shape={array.shape}"
            )

    lengths = {
        name: len(array)
        for name, array in arrays.items()
    }

    unique_lengths = set(
        lengths.values()
    )

    if len(unique_lengths) != 1:
        raise HammingTrialBuildError(
            "Trial array lengths differ: "
            f"{lengths}"
        )

    trial_count = len(
        probe_identity_array
    )

    if trial_count == 0:
        raise HammingTrialBuildError(
            f"{trial_type}: no trial rows supplied"
        )

    if not np.isfinite(
        normalized_array
    ).all():
        invalid_count = int(
            (
                ~np.isfinite(
                    normalized_array
                )
            ).sum()
        )

        raise HammingTrialBuildError(
            "Normalized Hamming distances contain "
            f"non-finite values: {invalid_count}"
        )

    if (distance_array < 0).any():
        raise HammingTrialBuildError(
            "Hamming distances contain negative values"
        )

    if (
        (normalized_array < 0.0).any()
        or (normalized_array > 1.0).any()
    ):
        raise HammingTrialBuildError(
            "Normalized Hamming distances must be "
            "within [0, 1]"
        )

    expected_is_match = (
        trial_type == "genuine"
    )

    if bool(is_match) != expected_is_match:
        raise HammingTrialBuildError(
            "is_match is inconsistent with trial type: "
            f"trial_type={trial_type}, "
            f"is_match={is_match}"
        )

    for (
        probe_identity_value,
        probe_image_value,
        distance,
        normalized_distance,
    ) in zip(
        probe_identity_array,
        probe_image_array,
        distance_array,
        normalized_array,
        strict=True,
    ):
        probe_identity_id = (
            validate_identity_id(
                str(
                    probe_identity_value
                ),
                field_name="probe identity ID",
            )
        )

        probe_image_id = str(
            probe_image_value
        )

        parsed_probe_identity_id = (
            parse_identity_from_image_id(
                probe_image_id
            )
        )

        if (
            parsed_probe_identity_id
            != probe_identity_id
        ):
            raise HammingTrialBuildError(
                "Probe identity does not match probe image ID: "
                f"probe_identity_id={probe_identity_id}, "
                f"parsed_identity_id="
                f"{parsed_probe_identity_id}, "
                f"probe_image_id={probe_image_id}"
            )

        if (
            trial_type == "genuine"
            and probe_identity_id
            != claimant_identity_id
        ):
            raise HammingTrialBuildError(
                "Genuine trial claimant/probe mismatch: "
                f"claimant={claimant_identity_id}, "
                f"probe={probe_identity_id}, "
                f"image={probe_image_id}"
            )

        if (
            trial_type == "impostor"
            and probe_identity_id
            == claimant_identity_id
        ):
            raise HammingTrialBuildError(
                "Impostor trial uses claimant identity: "
                f"claimant={claimant_identity_id}, "
                f"probe={probe_identity_id}, "
                f"image={probe_image_id}"
            )

        records.append(
            {
                "trial_type": str(
                    trial_type
                ),
                "experiment_group": str(
                    experiment_group
                ),
                "claimant_identity_id": str(
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


def normalize_and_validate_trial_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize DataFrame dtypes and validate all trial metadata before
    saving the Parquet artifact.
    """

    required_columns = {
        "trial_type",
        "experiment_group",
        "claimant_identity_id",
        "probe_identity_id",
        "probe_image_id",
        "hamming_distance",
        "normalized_hamming_distance",
        "is_match",
    }

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        raise HammingTrialBuildError(
            "Missing trial columns: "
            f"{sorted(missing_columns)}"
        )

    normalized = dataframe.copy()

    string_columns = [
        "trial_type",
        "experiment_group",
        "claimant_identity_id",
        "probe_identity_id",
        "probe_image_id",
    ]

    for column in string_columns:
        normalized[column] = pd.Series(
            normalized[column],
            dtype="string",
            index=normalized.index,
        )

    normalized[
        "hamming_distance"
    ] = pd.to_numeric(
        normalized[
            "hamming_distance"
        ],
        errors="raise",
    ).astype(np.int16)

    normalized[
        "normalized_hamming_distance"
    ] = pd.to_numeric(
        normalized[
            "normalized_hamming_distance"
        ],
        errors="raise",
    ).astype(np.float32)

    normalized[
        "is_match"
    ] = normalized[
        "is_match"
    ].astype(bool)

    null_counts = (
        normalized[
            string_columns
        ]
        .isna()
        .sum()
    )

    if int(
        null_counts.sum()
    ) > 0:
        raise HammingTrialBuildError(
            "Trial metadata contains null values: "
            f"{null_counts.to_dict()}"
        )

    valid_trial_types = {
        "genuine",
        "impostor",
    }

    observed_trial_types = set(
        normalized[
            "trial_type"
        ].astype(str)
    )

    invalid_trial_types = (
        observed_trial_types
        - valid_trial_types
    )

    if invalid_trial_types:
        raise HammingTrialBuildError(
            "Invalid trial types detected: "
            f"{sorted(invalid_trial_types)}"
        )

    invalid_claimant_mask = (
        ~normalized[
            "claimant_identity_id"
        ].str.fullmatch(
            r"n\d{6}",
            na=False,
        )
    )

    if invalid_claimant_mask.any():
        examples = normalized.loc[
            invalid_claimant_mask,
            [
                "trial_type",
                "claimant_identity_id",
                "probe_identity_id",
                "probe_image_id",
            ],
        ].head(10)

        raise HammingTrialBuildError(
            "Invalid claimant identity IDs detected:\n"
            f"{examples.to_string(index=False)}"
        )

    invalid_probe_mask = (
        ~normalized[
            "probe_identity_id"
        ].str.fullmatch(
            r"n\d{6}",
            na=False,
        )
    )

    if invalid_probe_mask.any():
        examples = normalized.loc[
            invalid_probe_mask,
            [
                "trial_type",
                "claimant_identity_id",
                "probe_identity_id",
                "probe_image_id",
            ],
        ].head(10)

        raise HammingTrialBuildError(
            "Invalid probe identity IDs detected:\n"
            f"{examples.to_string(index=False)}"
        )

    parsed_probe_identity_ids = (
        normalized[
            "probe_image_id"
        ]
        .astype(str)
        .map(
            parse_identity_from_image_id
        )
        .astype("string")
    )

    image_identity_mismatch = (
        parsed_probe_identity_ids
        != normalized[
            "probe_identity_id"
        ]
    )

    if image_identity_mismatch.any():
        examples = normalized.loc[
            image_identity_mismatch,
            [
                "trial_type",
                "claimant_identity_id",
                "probe_identity_id",
                "probe_image_id",
            ],
        ].head(10).copy()

        examples[
            "parsed_probe_identity_id"
        ] = parsed_probe_identity_ids[
            image_identity_mismatch
        ].head(10).to_numpy()

        raise HammingTrialBuildError(
            "Probe identity differs from probe image identity:\n"
            f"{examples.to_string(index=False)}"
        )

    genuine_mask = (
        normalized[
            "trial_type"
        ]
        == "genuine"
    )

    impostor_mask = (
        normalized[
            "trial_type"
        ]
        == "impostor"
    )

    genuine_identity_mismatch = (
        genuine_mask
        & (
            normalized[
                "claimant_identity_id"
            ]
            != normalized[
                "probe_identity_id"
            ]
        )
    )

    if genuine_identity_mismatch.any():
        examples = normalized.loc[
            genuine_identity_mismatch,
            [
                "claimant_identity_id",
                "probe_identity_id",
                "probe_image_id",
            ],
        ].head(10)

        raise HammingTrialBuildError(
            "Genuine trials contain claimant/probe "
            "identity mismatches:\n"
            f"{examples.to_string(index=False)}"
        )

    impostor_identity_match = (
        impostor_mask
        & (
            normalized[
                "claimant_identity_id"
            ]
            == normalized[
                "probe_identity_id"
            ]
        )
    )

    if impostor_identity_match.any():
        examples = normalized.loc[
            impostor_identity_match,
            [
                "claimant_identity_id",
                "probe_identity_id",
                "probe_image_id",
            ],
        ].head(10)

        raise HammingTrialBuildError(
            "Impostor trials contain claimant/probe "
            "identity matches:\n"
            f"{examples.to_string(index=False)}"
        )

    expected_match_values = (
        genuine_mask
        .to_numpy(
            dtype=np.bool_,
        )
    )

    actual_match_values = (
        normalized[
            "is_match"
        ]
        .to_numpy(
            dtype=np.bool_,
        )
    )

    if not np.array_equal(
        expected_match_values,
        actual_match_values,
    ):
        mismatch_count = int(
            np.count_nonzero(
                expected_match_values
                != actual_match_values
            )
        )

        raise HammingTrialBuildError(
            "is_match values are inconsistent with "
            f"trial_type: mismatch_count={mismatch_count}"
        )

    distance_values = normalized[
        "hamming_distance"
    ].to_numpy(
        dtype=np.int16,
    )

    normalized_distance_values = normalized[
        "normalized_hamming_distance"
    ].to_numpy(
        dtype=np.float32,
    )

    if (distance_values < 0).any():
        raise HammingTrialBuildError(
            "Hamming distances contain negative values"
        )

    if not np.isfinite(
        normalized_distance_values
    ).all():
        invalid_count = int(
            (
                ~np.isfinite(
                    normalized_distance_values
                )
            ).sum()
        )

        raise HammingTrialBuildError(
            "Normalized distances contain non-finite "
            f"values: invalid_count={invalid_count}"
        )

    if (
        (
            normalized_distance_values
            < 0.0
        ).any()
        or (
            normalized_distance_values
            > 1.0
        ).any()
    ):
        raise HammingTrialBuildError(
            "Normalized Hamming distances must be "
            "within [0, 1]"
        )

    return normalized


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
        if max_claimants <= 0:
            raise HammingTrialBuildError(
                "max_claimants must be positive"
            )

        claimant_ids = claimant_ids[
            :max_claimants
        ]

    probe_pool = build_group_probe_pool(
        repository=repository,
        group_identity_ids=(
            group_identity_ids
        ),
    )

    records: list[
        dict[str, Any]
    ] = []

    for claimant_identity_id in tqdm(
        claimant_ids,
        desc=(
            f"{group_name} claimants"
        ),
        unit="identity",
    ):
        validate_identity_id(
            claimant_identity_id,
            field_name="claimant identity ID",
        )

        if (
            claimant_identity_id
            not in artifact_identity_to_index
        ):
            raise HammingTrialBuildError(
                "Claimant identity missing from "
                f"enrollment artifacts: "
                f"{claimant_identity_id}"
            )

        artifact_index = (
            artifact_identity_to_index[
                claimant_identity_id
            ]
        )

        enrollment_template = np.asarray(
            artifacts[
                "enrollment_templates"
            ][artifact_index],
            dtype=np.uint8,
        )

        selected_dimensions = np.asarray(
            artifacts[
                "selected_dimensions"
            ][artifact_index],
            dtype=np.int16,
        )

        genuine_pool = probe_pool[
            claimant_identity_id
        ]

        genuine_binary = (
            transform_probe_embeddings_for_claimant(
                probe_embeddings=(
                    genuine_pool[
                        "embeddings"
                    ]
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
            genuine_pool[
                "embeddings"
            ]
        )

        if (
            genuine_result.trial_count
            != genuine_probe_count
        ):
            raise HammingTrialBuildError(
                "Genuine result count differs from "
                "genuine probe count: "
                f"result={genuine_result.trial_count}, "
                f"probes={genuine_probe_count}"
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
                dtype=object,
            ),
            probe_image_ids=np.asarray(
                genuine_pool[
                    "image_ids"
                ],
                dtype=object,
            ),
            distances=(
                genuine_result
                .hamming_distances
            ),
            normalized_distances=(
                genuine_result
                .normalized_distances
            ),
            is_match=True,
        )

        claimant_numeric_id = int(
            claimant_identity_id[
                1:
            ]
        )

        claimant_seed = (
            random_seed
            + claimant_numeric_id
            * 1009
        ) % (
            2**32
        )

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

        if (
            impostor_result.trial_count
            != impostor_trials_per_claimant
        ):
            raise HammingTrialBuildError(
                "Impostor result count differs from "
                "configured trial count: "
                f"result={impostor_result.trial_count}, "
                f"configured="
                f"{impostor_trials_per_claimant}"
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
                impostor_result
                .hamming_distances
            ),
            normalized_distances=(
                impostor_result
                .normalized_distances
            ),
            is_match=False,
        )

    dataframe = pd.DataFrame.from_records(
        records
    )

    if dataframe.empty:
        raise HammingTrialBuildError(
            f"No trials generated for group "
            f"{group_name}"
        )

    return (
        normalize_and_validate_trial_dataframe(
            dataframe
        )
    )


def summarize_trials(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    genuine = dataframe[
        dataframe[
            "trial_type"
        ]
        == "genuine"
    ]

    impostor = dataframe[
        dataframe[
            "trial_type"
        ]
        == "impostor"
    ]

    if genuine.empty:
        raise HammingTrialBuildError(
            "No genuine trials available for summary"
        )

    if impostor.empty:
        raise HammingTrialBuildError(
            "No impostor trials available for summary"
        )

    return {
        "trial_count": int(
            len(
                dataframe
            )
        ),
        "genuine_trial_count": int(
            len(
                genuine
            )
        ),
        "impostor_trial_count": int(
            len(
                impostor
            )
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

        data_config = config[
            "data"
        ]

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
            data_config[
                "cache_dir"
            ]
        ).expanduser().resolve()

        normalization_dir = Path(
            normalization_config[
                "output_dir"
            ]
        ).expanduser().resolve()

        binary_dir = Path(
            binarization_config[
                "output_dir"
            ]
        ).expanduser().resolve()

        output_root = Path(
            trial_config[
                "output_dir"
            ]
        ).expanduser().resolve()

        enrollment_counts = [
            int(
                value
            )
            for value in trial_config[
                "enrollment_counts"
            ]
        ]

        experiment_groups = [
            str(
                value
            )
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

        if not enrollment_counts:
            raise HammingTrialBuildError(
                "No enrollment counts configured"
            )

        if any(
            value <= 0
            for value in enrollment_counts
        ):
            raise HammingTrialBuildError(
                "Enrollment counts must be positive"
            )

        if not experiment_groups:
            raise HammingTrialBuildError(
                "No experiment groups configured"
            )

        if top_k <= 0:
            raise HammingTrialBuildError(
                "feature_selection.top_k must be positive"
            )

        if (
            impostor_trials_per_claimant
            <= 0
        ):
            raise HammingTrialBuildError(
                "impostor_trials_per_claimant "
                "must be positive"
            )

        repository = EmbeddingRepository(
            cache_root=cache_dir,
            expected_embedding_dimension=512,
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_summaries: list[
            dict[str, Any]
        ] = []

        print(
            "Cache identities:",
            len(
                repository
            ),
        )

        print(
            "Enrollment counts:",
            enrollment_counts,
        )

        print(
            "Groups:",
            experiment_groups,
        )

        print(
            "Impostor trials per claimant:",
            impostor_trials_per_claimant,
        )

        print(
            "Random seed:",
            random_seed,
        )

        for enrollment_count in (
            enrollment_counts
        ):
            print()

            print(
                "Enrollment count="
                f"{enrollment_count}"
            )

            artifacts = (
                load_enrollment_artifacts(
                    enrollment_count=(
                        enrollment_count
                    ),
                    top_k=top_k,
                    normalization_dir=(
                        normalization_dir
                    ),
                    binary_dir=(
                        binary_dir
                    ),
                )
            )

            artifact_identity_to_index = {
                str(
                    identity_id
                ): index
                for index, identity_id
                in enumerate(
                    artifacts[
                        "identity_ids"
                    ]
                )
            }

            enrollment_output_dir = (
                output_root
                / (
                    f"enrollment_"
                    f"{enrollment_count:02d}"
                )
            )

            enrollment_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            group_summaries: dict[
                str,
                dict[str, Any],
            ] = {}

            for group_name in (
                experiment_groups
            ):
                group_identity_ids = [
                    str(
                        identity_id
                    )
                    for identity_id, group
                    in zip(
                        artifacts[
                            "identity_ids"
                        ],
                        artifacts[
                            "experiment_groups"
                        ],
                        strict=True,
                    )
                    if str(
                        group
                    ) == group_name
                ]

                if not group_identity_ids:
                    raise HammingTrialBuildError(
                        "No identities found for "
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
                        "Output already exists: "
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
                        random_seed=(
                            random_seed
                        ),
                        max_claimants=(
                            args.max_claimants
                        ),
                    )
                )

                dataframe = (
                    normalize_and_validate_trial_dataframe(
                        dataframe
                    )
                )

                dataframe.to_parquet(
                    output_path,
                    index=False,
                    engine="pyarrow",
                    compression="snappy",
                )

                saved_dataframe = (
                    pd.read_parquet(
                        output_path,
                        columns=[
                            "trial_type",
                            "claimant_identity_id",
                            "probe_identity_id",
                            "probe_image_id",
                        ],
                    )
                )

                normalize_and_validate_trial_dataframe(
                    dataframe
                )

                if (
                    len(
                        saved_dataframe
                    )
                    != len(
                        dataframe
                    )
                ):
                    raise HammingTrialBuildError(
                        "Saved Parquet row count differs "
                        "from generated DataFrame: "
                        f"generated={len(dataframe)}, "
                        f"saved={len(saved_dataframe)}"
                    )

                group_summary = (
                    summarize_trials(
                        dataframe
                    )
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

                print(
                    " Group:",
                    group_name,
                )

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

            enrollment_summary_path = (
                enrollment_output_dir
                / "summary.json"
            )

            with enrollment_summary_path.open(
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
            "hamming_trials": (
                all_summaries
            ),
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
        HammingTrialBuildError,
        HammingTrialError,
        EmbeddingRepositoryError,
    ) as exc:
        print(
            "Hamming-trial generation failed: "
            f"{exc}",
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
            "Unexpected error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        return 99


if __name__ == "__main__":
    sys.exit(
        main()
    )