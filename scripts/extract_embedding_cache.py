from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from fuzzy_did.face import (
    DeepFaceExtractor,
    FaceExtractionError,
)
from fuzzy_did.storage import (
    EmbeddingCache,
)


class EmbeddingExtractionError(RuntimeError):
    """Raised when embedding-cache generation cannot continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract identity-level DeepFace embedding caches "
            "from one dataset shard."
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
        "--shard-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--identity-id",
        type=str,
        default=None,
        help=(
            "Process only one identity. "
            "Useful for an identity-level smoke test."
        ),
    )

    parser.add_argument(
        "--max-identities",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip identities with a valid complete cache.",
    )

    parser.add_argument(
        "--retry-partial",
        action="store_true",
        help="Retry identities previously recorded as partial.",
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise EmbeddingExtractionError(
            f"Configuration does not exist: {resolved}"
        )

    with resolved.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        raise EmbeddingExtractionError(
            "Configuration root must be a mapping."
        )

    return loaded


def select_identity_frames(
    dataframe: pd.DataFrame,
    identity_id: str | None,
    max_identities: int | None,
) -> list[tuple[str, pd.DataFrame]]:
    grouped = [
        (
            str(group_identity),
            group_frame.sort_values(
                by="image_id",
                kind="stable",
            ).copy(),
        )
        for group_identity, group_frame
        in dataframe.groupby(
            "identity_id",
            sort=True,
        )
    ]

    if identity_id is not None:
        grouped = [
            item
            for item in grouped
            if item[0] == identity_id
        ]

        if not grouped:
            raise EmbeddingExtractionError(
                f"Identity is not present in shard: {identity_id}"
            )

    if max_identities is not None:
        if max_identities <= 0:
            raise EmbeddingExtractionError(
                "max-identities must be positive."
            )

        grouped = grouped[
            :max_identities
        ]

    return grouped


def read_candidate_rank(
    value: Any,
) -> int:
    if value is None or pd.isna(value):
        return -1

    return int(value)


def extract_identity(
    *,
    identity_id: str,
    identity_frame: pd.DataFrame,
    extractor: DeepFaceExtractor,
    cache: EmbeddingCache,
    shard_index: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    successful_image_ids: list[str] = []
    successful_image_paths: list[str] = []
    successful_relative_paths: list[str] = []
    successful_experiment_groups: list[str] = []
    successful_sample_roles: list[str] = []
    successful_candidate_ranks: list[int] = []
    successful_embeddings: list[np.ndarray] = []
    successful_confidences: list[float] = []

    failures: list[dict[str, Any]] = []

    for row in tqdm(
        identity_frame.itertuples(
            index=False
        ),
        total=len(identity_frame),
        desc=identity_id,
        unit="image",
        leave=False,
    ):
        try:
            result = extractor.extract(
                row.image_path
            )

        except FaceExtractionError as exc:
            failures.append(
                {
                    "identity_id": identity_id,
                    "image_id": str(
                        row.image_id
                    ),
                    "image_path": str(
                        row.image_path
                    ),
                    "relative_path": str(
                        row.relative_path
                    ),
                    "error_type": type(
                        exc
                    ).__name__,
                    "error_message": str(
                        exc
                    ),
                }
            )
            continue

        successful_image_ids.append(
            str(row.image_id)
        )
        successful_image_paths.append(
            str(row.image_path)
        )
        successful_relative_paths.append(
            str(row.relative_path)
        )
        successful_experiment_groups.append(
            str(row.experiment_group)
        )
        successful_sample_roles.append(
            str(row.sample_role)
        )
        successful_candidate_ranks.append(
            read_candidate_rank(
                row.enrollment_candidate_rank
            )
        )
        successful_embeddings.append(
            result.embedding
        )
        successful_confidences.append(
            np.nan
            if result.face_confidence is None
            else float(
                result.face_confidence
            )
        )

    requested_count = len(
        identity_frame
    )
    success_count = len(
        successful_embeddings
    )
    failure_count = len(
        failures
    )

    cache_path: Path | None = None
    embedding_dimension: int | None = None

    if successful_embeddings:
        dimensions = {
            int(embedding.shape[0])
            for embedding in successful_embeddings
        }

        if len(dimensions) != 1:
            raise EmbeddingExtractionError(
                f"Inconsistent embedding dimensions for "
                f"{identity_id}: {sorted(dimensions)}"
            )

        embedding_matrix = np.stack(
            successful_embeddings,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        confidence_array = np.asarray(
            successful_confidences,
            dtype=np.float32,
        )

        embedding_dimension = int(
            embedding_matrix.shape[1]
        )

        cache_path = cache.save_identity(
            identity_id=identity_id,
            image_ids=successful_image_ids,
            image_paths=successful_image_paths,
            relative_paths=successful_relative_paths,
            experiment_groups=(
                successful_experiment_groups
            ),
            sample_roles=(
                successful_sample_roles
            ),
            candidate_ranks=(
                successful_candidate_ranks
            ),
            embeddings=embedding_matrix,
            face_confidences=(
                confidence_array
            ),
        )

    cache.save_failures(
        identity_id=identity_id,
        failures=failures,
    )

    if success_count == requested_count:
        status = "complete"
    elif success_count > 0:
        status = "partial"
    else:
        status = "failed"

    elapsed_seconds = (
        time.perf_counter() - started_at
    )

    status_record = {
        "status": status,
        "shard_index": shard_index,
        "requested_image_count": (
            requested_count
        ),
        "success_image_count": (
            success_count
        ),
        "failure_image_count": (
            failure_count
        ),
        "embedding_dimension": (
            embedding_dimension
        ),
        "cache_path": (
            str(cache_path)
            if cache_path is not None
            else None
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    cache.save_status(
        identity_id=identity_id,
        payload=status_record,
    )

    cache.append_manifest(
        shard_index=shard_index,
        payload={
            "identity_id": identity_id,
            **status_record,
        },
    )

    return {
        "identity_id": identity_id,
        **status_record,
    }


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(
            args.config
        )

        data_config = config["data"]
        sharding_config = config[
            "sharding"
        ]
        face_config = config[
            "face_model"
        ]
        extraction_config = config.get(
            "extraction",
            {},
        )

        num_shards = int(
            sharding_config["num_shards"]
        )

        if not (
            0
            <= args.shard_index
            < num_shards
        ):
            raise EmbeddingExtractionError(
                f"shard-index must be between 0 and "
                f"{num_shards - 1}"
            )

        shard_path = (
            Path(data_config["shard_dir"])
            / (
                f"shard_"
                f"{args.shard_index:03d}"
                f".parquet"
            )
        ).expanduser().resolve()

        if not shard_path.is_file():
            raise EmbeddingExtractionError(
                f"Shard does not exist: {shard_path}"
            )

        cache_root = Path(
            data_config["cache_dir"]
        ).expanduser().resolve()

        dataframe = pd.read_parquet(
            shard_path
        )

        required_columns = {
            "identity_id",
            "image_id",
            "image_path",
            "relative_path",
            "experiment_group",
            "sample_role",
            "enrollment_candidate_rank",
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise EmbeddingExtractionError(
                "Shard is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        identity_frames = (
            select_identity_frames(
                dataframe=dataframe,
                identity_id=args.identity_id,
                max_identities=(
                    args.max_identities
                ),
            )
        )

        print(
            "CUDA_VISIBLE_DEVICES:",
            os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                "<not set>",
            ),
        )
        print(
            "Shard:",
            args.shard_index,
        )
        print(
            "Identities to process:",
            len(identity_frames),
        )

        expected_dimension = int(
            extraction_config.get(
                "expected_embedding_dimension",
                512,
            )
        )

        extractor = DeepFaceExtractor(
            model_name=str(
                face_config["model_name"]
            ),
            detector_backend=str(
                face_config[
                    "detector_backend"
                ]
            ),
            normalization=str(
                face_config["normalization"]
            ),
            align=bool(
                face_config["align"]
            ),
            enforce_detection=bool(
                face_config[
                    "enforce_detection"
                ]
            ),
            max_faces=int(
                face_config.get(
                    "max_faces",
                    1,
                )
            ),
            expected_dimension=(
                expected_dimension
            ),
            memory_growth=bool(
                extraction_config.get(
                    "memory_growth",
                    True,
                )
            ),
        )

        cache = EmbeddingCache(
            cache_root=cache_root
        )

        completed_count = 0
        partial_count = 0
        failed_count = 0
        skipped_count = 0

        for identity_id, identity_frame in tqdm(
            identity_frames,
            desc=f"Shard {args.shard_index}",
            unit="identity",
        ):
            expected_image_ids = (
                identity_frame[
                    "image_id"
                ]
                .astype(str)
                .tolist()
            )

            if args.resume:
                if cache.is_complete(
                    identity_id=identity_id,
                    expected_image_ids=(
                        expected_image_ids
                    ),
                    expected_dimension=(
                        expected_dimension
                    ),
                ):
                    skipped_count += 1
                    continue

                previous_status = (
                    cache.read_status(
                        identity_id
                    )
                )

                if (
                    previous_status is not None
                    and previous_status.get(
                        "status"
                    ) == "partial"
                    and not args.retry_partial
                ):
                    skipped_count += 1
                    continue

            record = extract_identity(
                identity_id=identity_id,
                identity_frame=(
                    identity_frame
                ),
                extractor=extractor,
                cache=cache,
                shard_index=(
                    args.shard_index
                ),
            )

            if record["status"] == "complete":
                completed_count += 1
            elif record["status"] == "partial":
                partial_count += 1
            else:
                failed_count += 1

            print(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

        print()
        print(
            "Extraction finished:",
            {
                "completed": completed_count,
                "partial": partial_count,
                "failed": failed_count,
                "skipped": skipped_count,
            },
        )

        return 0

    except KeyError as exc:
        print(
            f"Missing configuration key: {exc}",
            file=sys.stderr,
        )
        return 2

    except (
        EmbeddingExtractionError,
        FaceExtractionError,
    ) as exc:
        print(
            f"Embedding extraction failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Embedding extraction interrupted.",
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