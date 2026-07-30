from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from fuzzy_did.binarization import MedianBinarizerConfig
from fuzzy_did.data import (
    EmbeddingRepository,
    EmbeddingRepositoryError,
)
from fuzzy_did.evaluation import (
    transform_probe_embeddings_for_claimant,
)
from fuzzy_did.fuzzy_extractor import (
    BCHCodeOffsetFuzzyExtractor,
    FuzzyExtractorError,
    hash_message,
)
from fuzzy_did.normalization import RobustScalerState


class ActualFuzzyEvaluationError(RuntimeError):
    """Raised when actual fuzzy-extractor evaluation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate actual BCH code-offset fuzzy extraction "
            "using claimant-conditioned binary face templates."
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
        "--enrollment-counts",
        type=int,
        nargs="+",
        default=None,
        help="Override configured enrollment counts.",
    )

    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=None,
        help="Override configured BCH correction capabilities.",
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["development", "evaluation"],
        default=None,
    )

    parser.add_argument(
        "--max-claimants",
        type=int,
        default=None,
        help="Limit claimants for a smoke test.",
    )

    parser.add_argument(
        "--no-trial-details",
        action="store_true",
        help="Do not save trial-level Parquet files.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise ActualFuzzyEvaluationError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise ActualFuzzyEvaluationError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ActualFuzzyEvaluationError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ActualFuzzyEvaluationError(
            "Configuration root must be a mapping"
        )

    return loaded


def deterministic_message(
    *,
    identity_id: str,
    enrollment_count: int,
    correction_capability: int,
    message_length: int,
    random_seed: int,
) -> np.ndarray:
    """
    Generate a deterministic experimental BCH message.

    A deployed system must generate this message from a CSPRNG.
    Determinism here is only for reproducible experiments.
    """

    if message_length <= 0:
        raise ActualFuzzyEvaluationError(
            "message_length must be positive"
        )

    context = (
        f"fuzzy-did|seed={random_seed}|"
        f"identity={identity_id}|"
        f"enrollment={enrollment_count}|"
        f"t={correction_capability}"
    ).encode("utf-8")

    required_bytes = (
        message_length + 7
    ) // 8

    generated = bytearray()
    counter = 0

    while len(generated) < required_bytes:
        digest = hashlib.sha256(
            context
            + counter.to_bytes(
                4,
                byteorder="big",
                signed=False,
            )
        ).digest()

        generated.extend(digest)
        counter += 1

    bits = np.unpackbits(
        np.frombuffer(
            bytes(generated[:required_bytes]),
            dtype=np.uint8,
        ),
        bitorder="big",
    )

    return bits[:message_length].astype(
        np.uint8,
        copy=True,
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
            f"enrollment_{enrollment_count:02d}_"
            f"top{top_k}_robust.npz"
        )
    )

    binary_path = (
        binary_dir
        / (
            f"enrollment_{enrollment_count:02d}_"
            f"top{top_k}_binary.npz"
        )
    )

    if not normalization_path.is_file():
        raise ActualFuzzyEvaluationError(
            f"Missing normalization artifact: {normalization_path}"
        )

    if not binary_path.is_file():
        raise ActualFuzzyEvaluationError(
            f"Missing binary artifact: {binary_path}"
        )

    with np.load(
        normalization_path,
        allow_pickle=False,
    ) as data:
        normalization_identity_ids = (
            data["identity_ids"].astype(str)
        )

        experiment_groups = (
            data["experiment_groups"].astype(str)
        )

        selected_dimensions = (
            data["selected_dimensions"].astype(np.int16)
        )

        scaler_state = RobustScalerState(
            center=data["global_center"].astype(np.float32),
            scale=data["global_scale"].astype(np.float32),
            raw_scale=data[
                "global_raw_scale"
            ].astype(np.float32),
            q1=data["global_q1"].astype(np.float32),
            q3=data["global_q3"].astype(np.float32),
            floored_dimensions=data[
                "floored_dimensions"
            ].astype(np.bool_),
        )

    with np.load(
        binary_path,
        allow_pickle=False,
    ) as data:
        binary_identity_ids = (
            data["identity_ids"].astype(str)
        )

        enrollment_templates = (
            data["binary_templates"].astype(np.uint8)
        )

        binarizer_config = MedianBinarizerConfig(
            threshold=float(data["threshold"][0]),
            positive_when_greater=bool(
                data["positive_when_greater"][0]
            ),
            bitorder=str(data["bitorder"][0]),
        )

    if not np.array_equal(
        normalization_identity_ids,
        binary_identity_ids,
    ):
        raise ActualFuzzyEvaluationError(
            "Identity order differs between normalization "
            "and enrollment binary artifacts"
        )

    if enrollment_templates.shape != (
        len(normalization_identity_ids),
        top_k,
    ):
        raise ActualFuzzyEvaluationError(
            "Unexpected enrollment-template shape: "
            f"{enrollment_templates.shape}"
        )

    return {
        "identity_ids": normalization_identity_ids,
        "experiment_groups": experiment_groups,
        "selected_dimensions": selected_dimensions,
        "enrollment_templates": enrollment_templates,
        "scaler_state": scaler_state,
        "binarizer_config": binarizer_config,
    }


def load_group_probe_pool(
    *,
    repository: EmbeddingRepository,
    identity_ids: list[str],
) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}

    for identity_id in identity_ids:
        cache = repository.load(identity_id)

        probe_mask = (
            (cache.enrollment_candidate_ranks < 0)
            & (
                cache.sample_roles.astype(str)
                == "probe"
            )
        )

        if not probe_mask.any():
            raise ActualFuzzyEvaluationError(
                f"{identity_id}: no valid probes"
            )

        image_ids = cache.image_ids[
            probe_mask
        ].astype(str)

        embeddings = cache.embeddings[
            probe_mask
        ].astype(np.float32)

        image_to_index = {
            image_id: index
            for index, image_id in enumerate(image_ids)
        }

        pool[identity_id] = {
            "image_ids": image_ids,
            "embeddings": embeddings,
            "image_to_index": image_to_index,
        }

    return pool


def gather_trial_embeddings(
    trial_frame: pd.DataFrame,
    *,
    probe_pool: dict[str, dict[str, Any]],
) -> np.ndarray:
    embeddings: list[np.ndarray] = []

    for probe_identity_id, probe_image_id in zip(
        trial_frame["probe_identity_id"].astype(str),
        trial_frame["probe_image_id"].astype(str),
        strict=True,
    ):
        if probe_identity_id not in probe_pool:
            raise ActualFuzzyEvaluationError(
                f"Probe identity missing from pool: "
                f"{probe_identity_id}"
            )

        identity_pool = probe_pool[
            probe_identity_id
        ]

        image_to_index = identity_pool[
            "image_to_index"
        ]

        if probe_image_id not in image_to_index:
            raise ActualFuzzyEvaluationError(
                f"Probe image missing from cache: "
                f"{probe_image_id}"
            )

        image_index = image_to_index[
            probe_image_id
        ]

        embeddings.append(
            identity_pool["embeddings"][
                image_index
            ]
        )

    if not embeddings:
        return np.empty(
            (0, 512),
            dtype=np.float32,
        )

    return np.stack(
        embeddings,
        axis=0,
    ).astype(np.float32)


def decode_batch(
    *,
    extractor: BCHCodeOffsetFuzzyExtractor,
    probe_templates: np.ndarray,
    helper_data: np.ndarray,
    expected_message: np.ndarray,
    expected_key_digest: bytes,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """
    Decode probe templates in batches.

    Message equality is checked directly. Digest equality is also
    calculated for successfully decoded rows.
    """

    if probe_templates.ndim != 2:
        raise ActualFuzzyEvaluationError(
            "probe_templates must have shape [N, n]"
        )

    if probe_templates.shape[1] != extractor.parameters.n:
        raise ActualFuzzyEvaluationError(
            "Probe BCH template length mismatch"
        )

    if batch_size <= 0:
        raise ActualFuzzyEvaluationError(
            "batch_size must be positive"
        )

    trial_count = len(probe_templates)

    decode_succeeded = np.zeros(
        trial_count,
        dtype=np.bool_,
    )

    message_matched = np.zeros(
        trial_count,
        dtype=np.bool_,
    )

    key_matched = np.zeros(
        trial_count,
        dtype=np.bool_,
    )

    decoder_error_count = np.full(
        trial_count,
        -1,
        dtype=np.int16,
    )

    for start in range(
        0,
        trial_count,
        batch_size,
    ):
        end = min(
            start + batch_size,
            trial_count,
        )

        batch_templates = probe_templates[
            start:end
        ]

        noisy_codewords = np.bitwise_xor(
            batch_templates,
            helper_data[None, :],
        ).astype(np.uint8)

        field_codewords = extractor.code.field(
            noisy_codewords
        )

        try:
            decoded_messages, error_counts = (
                extractor.code.decode(
                    field_codewords,
                    errors=True,
                )
            )
        except Exception:
            # Batch failure fallback: decode rows individually.
            for local_index in range(
                len(batch_templates)
            ):
                global_index = (
                    start + local_index
                )

                result = extractor.rep(
                    batch_templates[local_index],
                    helper_data,
                    expected_key_digest=(
                        expected_key_digest
                    ),
                )

                decode_succeeded[
                    global_index
                ] = result.decode_succeeded

                key_matched[
                    global_index
                ] = result.key_matched

                if (
                    result.decoder_error_count
                    is not None
                ):
                    decoder_error_count[
                        global_index
                    ] = (
                        result.decoder_error_count
                    )

                if (
                    result.recovered_message
                    is not None
                ):
                    message_matched[
                        global_index
                    ] = np.array_equal(
                        result.recovered_message,
                        expected_message,
                    )

            continue

        decoded_array = np.asarray(
            decoded_messages,
            dtype=np.uint8,
        )

        error_array = np.asarray(
            error_counts
        ).reshape(-1).astype(np.int16)

        valid_decode = error_array >= 0

        decoder_error_count[
            start:end
        ] = error_array

        decode_succeeded[
            start:end
        ] = valid_decode

        if decoded_array.ndim == 1:
            decoded_array = (
                decoded_array[None, :]
            )

        exact_message_match = (
            valid_decode
            & np.all(
                decoded_array
                == expected_message[None, :],
                axis=1,
            )
        )

        message_matched[
            start:end
        ] = exact_message_match

        # Digest verification is performed only for decoded messages.
        for local_index in np.flatnonzero(
            valid_decode
        ):
            recovered_digest = hash_message(
                decoded_array[local_index],
                digest_algorithm=(
                    extractor.digest_algorithm
                ),
            )

            key_matched[
                start + int(local_index)
            ] = (
                recovered_digest
                == expected_key_digest
            )

    return {
        "decode_succeeded": decode_succeeded,
        "message_matched": message_matched,
        "key_matched": key_matched,
        "decoder_error_count": (
            decoder_error_count
        ),
    }


def evaluate_claimant_trials(
    *,
    claimant_frame: pd.DataFrame,
    claimant_identity_id: str,
    claimant_embedding_index: int,
    artifacts: dict[str, Any],
    probe_pool: dict[str, dict[str, Any]],
    extractor: BCHCodeOffsetFuzzyExtractor,
    enrollment_count: int,
    correction_capability: int,
    random_seed: int,
    codeword_length: int,
    decode_batch_size: int,
) -> pd.DataFrame:
    selected_dimensions = artifacts[
        "selected_dimensions"
    ][claimant_embedding_index][
        :codeword_length
    ]

    enrollment_template = artifacts[
        "enrollment_templates"
    ][claimant_embedding_index][
        :codeword_length
    ]

    message = deterministic_message(
        identity_id=claimant_identity_id,
        enrollment_count=enrollment_count,
        correction_capability=(
            correction_capability
        ),
        message_length=(
            extractor.parameters.k
        ),
        random_seed=random_seed,
    )

    enrollment_record = extractor.gen(
        enrollment_template,
        message=message,
    )

    probe_embeddings = gather_trial_embeddings(
        claimant_frame,
        probe_pool=probe_pool,
    )

    probe_templates_128 = (
        transform_probe_embeddings_for_claimant(
            probe_embeddings=probe_embeddings,
            claimant_selected_dimensions=(
                artifacts[
                    "selected_dimensions"
                ][claimant_embedding_index]
            ),
            scaler_state=artifacts[
                "scaler_state"
            ],
            binarizer_config=artifacts[
                "binarizer_config"
            ],
        )
    )

    probe_templates = (
        probe_templates_128[
            :,
            :codeword_length,
        ]
    )

    hamming_distances = np.count_nonzero(
        probe_templates
        != enrollment_template[None, :],
        axis=1,
    ).astype(np.int16)

    within_radius = (
        hamming_distances
        <= extractor.parameters.t
    )

    decoding = decode_batch(
        extractor=extractor,
        probe_templates=probe_templates,
        helper_data=(
            enrollment_record.helper_data
        ),
        expected_message=message,
        expected_key_digest=(
            enrollment_record.key_digest
        ),
        batch_size=decode_batch_size,
    )

    result = claimant_frame.copy()

    result["hamming_distance_127"] = (
        hamming_distances
    )

    result["within_correction_radius"] = (
        within_radius
    )

    result["decode_succeeded"] = decoding[
        "decode_succeeded"
    ]

    result["decoder_error_count"] = decoding[
        "decoder_error_count"
    ]

    result["message_matched"] = decoding[
        "message_matched"
    ]

    result["key_matched"] = decoding[
        "key_matched"
    ]

    result["miscorrection"] = (
        result["decode_succeeded"]
        & ~result["key_matched"]
    )

    result["bch_n"] = extractor.parameters.n
    result["bch_k"] = extractor.parameters.k
    result["bch_d"] = extractor.parameters.d
    result["bch_t"] = extractor.parameters.t

    return result


def summarize_actual_results(
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

    genuine_count = len(genuine)
    impostor_count = len(impostor)

    genuine_key_matches = int(
        genuine["key_matched"].sum()
    )

    impostor_key_matches = int(
        impostor["key_matched"].sum()
    )

    genuine_decode_successes = int(
        genuine["decode_succeeded"].sum()
    )

    impostor_decode_successes = int(
        impostor["decode_succeeded"].sum()
    )

    genuine_miscorrections = int(
        genuine["miscorrection"].sum()
    )

    impostor_miscorrections = int(
        impostor["miscorrection"].sum()
    )

    proxy_genuine_successes = int(
        genuine[
            "within_correction_radius"
        ].sum()
    )

    proxy_impostor_successes = int(
        impostor[
            "within_correction_radius"
        ].sum()
    )

    return {
        "trial_count": int(len(dataframe)),
        "genuine_trial_count": genuine_count,
        "impostor_trial_count": impostor_count,

        "actual_genuine_key_match_count": (
            genuine_key_matches
        ),
        "actual_genuine_reconstruction_rate": (
            genuine_key_matches
            / genuine_count
        ),

        "actual_false_key_match_count": (
            impostor_key_matches
        ),
        "actual_false_key_reconstruction_rate": (
            impostor_key_matches
            / impostor_count
        ),

        "genuine_decode_success_count": (
            genuine_decode_successes
        ),
        "genuine_decode_success_rate": (
            genuine_decode_successes
            / genuine_count
        ),

        "impostor_decode_success_count": (
            impostor_decode_successes
        ),
        "impostor_decode_success_rate": (
            impostor_decode_successes
            / impostor_count
        ),

        "genuine_miscorrection_count": (
            genuine_miscorrections
        ),
        "genuine_miscorrection_rate": (
            genuine_miscorrections
            / genuine_count
        ),

        "impostor_miscorrection_count": (
            impostor_miscorrections
        ),
        "impostor_miscorrection_rate": (
            impostor_miscorrections
            / impostor_count
        ),

        "proxy_genuine_success_count": (
            proxy_genuine_successes
        ),
        "proxy_genuine_reconstruction_rate": (
            proxy_genuine_successes
            / genuine_count
        ),

        "proxy_false_reconstruction_count": (
            proxy_impostor_successes
        ),
        "proxy_false_reconstruction_rate": (
            proxy_impostor_successes
            / impostor_count
        ),

        "actual_proxy_genuine_rate_difference": (
            genuine_key_matches
            / genuine_count
            - proxy_genuine_successes
            / genuine_count
        ),

        "actual_proxy_false_rate_difference": (
            impostor_key_matches
            / impostor_count
            - proxy_impostor_successes
            / impostor_count
        ),
    }


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(args.config)

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
        hamming_config = config[
            "hamming_trials"
        ]
        fuzzy_config = config[
            "fuzzy_extractor"
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

        hamming_root = Path(
            hamming_config["output_dir"]
        ).expanduser().resolve()

        output_root = (
            Path(
                fuzzy_config["output_dir"]
            )
            .expanduser()
            .resolve()
            / "actual_evaluation"
        )

        configured_enrollment_counts = [
            int(value)
            for value in fuzzy_config[
                "enrollment_counts"
            ]
        ]

        enrollment_counts = (
            args.enrollment_counts
            if args.enrollment_counts
            is not None
            else configured_enrollment_counts
        )

        configured_t_values = [
            int(value)
            for value in fuzzy_config[
                "evaluation_correction_capabilities"
            ]
        ]

        t_values = (
            args.t_values
            if args.t_values is not None
            else configured_t_values
        )

        configured_groups = [
            str(value)
            for value in fuzzy_config[
                "experiment_groups"
            ]
        ]

        groups = (
            args.groups
            if args.groups is not None
            else configured_groups
        )

        codeword_length = int(
            fuzzy_config[
                "bch_codeword_length"
            ]
        )

        digest_algorithm = str(
            fuzzy_config.get(
                "digest_algorithm",
                "sha256",
            )
        )

        random_seed = int(
            fuzzy_config.get(
                "random_seed",
                42,
            )
        )

        decode_batch_size = int(
            fuzzy_config.get(
                "decode_batch_size",
                2048,
            )
        )

        save_trial_details = bool(
            fuzzy_config.get(
                "save_trial_details",
                True,
            )
        )

        if args.no_trial_details:
            save_trial_details = False

        top_k = int(
            feature_config.get(
                "top_k",
                128,
            )
        )

        if codeword_length > top_k:
            raise ActualFuzzyEvaluationError(
                "BCH codeword length exceeds selected "
                "template length"
            )

        repository = EmbeddingRepository(
            cache_root=cache_dir,
            expected_embedding_dimension=512,
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_records: list[
            dict[str, Any]
        ] = []

        print("Enrollment counts:", enrollment_counts)
        print("Correction capabilities:", t_values)
        print("Groups:", groups)
        print("BCH codeword length:", codeword_length)
        print("Decode batch size:", decode_batch_size)

        for enrollment_count in enrollment_counts:
            artifacts = load_enrollment_artifacts(
                enrollment_count=(
                    enrollment_count
                ),
                top_k=top_k,
                normalization_dir=(
                    normalization_dir
                ),
                binary_dir=binary_dir,
            )

            identity_to_index = {
                identity_id: index
                for index, identity_id
                in enumerate(
                    artifacts["identity_ids"]
                )
            }

            for group_name in groups:
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

                if args.max_claimants is not None:
                    group_identity_ids = (
                        group_identity_ids[
                            :args.max_claimants
                        ]
                    )

                # Include all possible probe identities because
                # existing impostor trials may reference them.
                full_group_identity_ids = [
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

                probe_pool = load_group_probe_pool(
                    repository=repository,
                    identity_ids=(
                        full_group_identity_ids
                    ),
                )

                trial_path = (
                    hamming_root
                    / (
                        f"enrollment_"
                        f"{enrollment_count:02d}"
                    )
                    / f"{group_name}.parquet"
                )

                if not trial_path.is_file():
                    raise ActualFuzzyEvaluationError(
                        f"Hamming trial file missing: "
                        f"{trial_path}"
                    )

                base_trials = pd.read_parquet(
                    trial_path
                )

                base_trials = base_trials[
                    base_trials[
                        "claimant_identity_id"
                    ].isin(group_identity_ids)
                ].copy()

                for correction_capability in t_values:
                    extractor = (
                        BCHCodeOffsetFuzzyExtractor(
                            codeword_length=(
                                codeword_length
                            ),
                            correction_capability=(
                                correction_capability
                            ),
                            digest_algorithm=(
                                digest_algorithm
                            ),
                        )
                    )

                    print()
                    print(
                        f"Enrollment={enrollment_count}, "
                        f"Group={group_name}, "
                        f"BCH=({extractor.parameters.n}, "
                        f"{extractor.parameters.k}, "
                        f"{extractor.parameters.d}), "
                        f"t={extractor.parameters.t}"
                    )

                    claimant_results: list[
                        pd.DataFrame
                    ] = []

                    for claimant_identity_id in tqdm(
                        group_identity_ids,
                        desc=(
                            f"E{enrollment_count} "
                            f"{group_name} "
                            f"t={correction_capability}"
                        ),
                        unit="claimant",
                    ):
                        claimant_frame = base_trials[
                            base_trials[
                                "claimant_identity_id"
                            ]
                            == claimant_identity_id
                        ].copy()

                        if claimant_frame.empty:
                            raise ActualFuzzyEvaluationError(
                                "No trial rows for claimant: "
                                f"{claimant_identity_id}"
                            )

                        claimant_result = (
                            evaluate_claimant_trials(
                                claimant_frame=(
                                    claimant_frame
                                ),
                                claimant_identity_id=(
                                    claimant_identity_id
                                ),
                                claimant_embedding_index=(
                                    identity_to_index[
                                        claimant_identity_id
                                    ]
                                ),
                                artifacts=artifacts,
                                probe_pool=probe_pool,
                                extractor=extractor,
                                enrollment_count=(
                                    enrollment_count
                                ),
                                correction_capability=(
                                    correction_capability
                                ),
                                random_seed=random_seed,
                                codeword_length=(
                                    codeword_length
                                ),
                                decode_batch_size=(
                                    decode_batch_size
                                ),
                            )
                        )

                        claimant_results.append(
                            claimant_result
                        )

                    result_dataframe = pd.concat(
                        claimant_results,
                        ignore_index=True,
                    )

                    summary = summarize_actual_results(
                        result_dataframe
                    )

                    summary_record = {
                        "enrollment_count": (
                            enrollment_count
                        ),
                        "experiment_group": (
                            group_name
                        ),
                        "requested_t": (
                            correction_capability
                        ),
                        "bch_n": (
                            extractor.parameters.n
                        ),
                        "bch_k": (
                            extractor.parameters.k
                        ),
                        "bch_d": (
                            extractor.parameters.d
                        ),
                        "bch_t": (
                            extractor.parameters.t
                        ),
                        "message_entropy_upper_bound_bits": (
                            extractor.parameters.k
                        ),
                        "digest_algorithm": (
                            digest_algorithm
                        ),
                        "claimant_count": len(
                            group_identity_ids
                        ),
                        **summary,
                    }

                    summary_records.append(
                        summary_record
                    )

                    result_dir = (
                        output_root
                        / (
                            f"enrollment_"
                            f"{enrollment_count:02d}"
                        )
                        / group_name
                    )

                    result_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    if save_trial_details:
                        detail_path = (
                            result_dir
                            / (
                                f"bch_t"
                                f"{correction_capability:02d}_"
                                "trials.parquet"
                            )
                        )

                        if (
                            detail_path.exists()
                            and not args.overwrite
                        ):
                            raise ActualFuzzyEvaluationError(
                                f"Output exists: {detail_path}"
                            )

                        result_dataframe.to_parquet(
                            detail_path,
                            index=False,
                            engine="pyarrow",
                            compression="snappy",
                        )

                        summary_record[
                            "trial_detail_path"
                        ] = str(
                            detail_path.resolve()
                        )

                    print(
                        "  Actual GRR:",
                        f"{summary['actual_genuine_reconstruction_rate']:.6f}",
                    )
                    print(
                        "  Actual false key reconstruction:",
                        f"{summary['actual_false_key_reconstruction_rate']:.6f}",
                    )
                    print(
                        "  Genuine miscorrection:",
                        f"{summary['genuine_miscorrection_rate']:.6f}",
                    )
                    print(
                        "  Impostor miscorrection:",
                        f"{summary['impostor_miscorrection_rate']:.6f}",
                    )
                    print(
                        "  Proxy GRR:",
                        f"{summary['proxy_genuine_reconstruction_rate']:.6f}",
                    )
                    print(
                        "  Proxy false reconstruction:",
                        f"{summary['proxy_false_reconstruction_rate']:.6f}",
                    )

        summary_dataframe = (
            pd.DataFrame.from_records(
                summary_records
            )
        )

        csv_path = (
            output_root
            / "actual_fuzzy_extractor_results.csv"
        )

        summary_dataframe.to_csv(
            csv_path,
            index=False,
        )

        payload = {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "dataset_name": "vggface2",
            "model_name": "Facenet512",
            "construction": (
                "BCH code-offset fuzzy extractor"
            ),
            "source_template_length": top_k,
            "bch_template_length": (
                codeword_length
            ),
            "dimension_selection_policy": (
                "first 127 dimensions from each "
                "identity-specific ranked Top-128 selection"
            ),
            "experimental_message_policy": (
                "deterministic SHA-256 expansion for "
                "reproducibility; deployment must use a CSPRNG"
            ),
            "success_condition": (
                "BCH decoding succeeds and recovered "
                "message/key digest matches enrollment"
            ),
            "results": summary_records,
        }

        json_path = (
            output_root
            / "actual_fuzzy_extractor_results.json"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("Saved CSV:", csv_path)
        print("Saved JSON:", json_path)

        return 0

    except KeyError as exc:
        print(
            f"Missing configuration key: {exc}",
            file=sys.stderr,
        )
        return 2

    except (
        ActualFuzzyEvaluationError,
        EmbeddingRepositoryError,
        FuzzyExtractorError,
    ) as exc:
        print(
            f"Actual fuzzy-extractor evaluation failed: "
            f"{exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Actual fuzzy-extractor evaluation interrupted",
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