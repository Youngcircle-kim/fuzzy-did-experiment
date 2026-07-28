from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LOGGER = logging.getLogger("build_embedding_shards")


class ShardBuildError(RuntimeError):
    """Raised when embedding shards cannot be generated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build identity-balanced embedding shards from a dataset split."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/embedding_facenet512.yaml"
        ),
        help="Path to the embedding experiment configuration.",
    )

    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="Override the input split Parquet path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the shard output directory.",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Override the number of shards.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files.",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise ShardBuildError(
            f"configuration file does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise ShardBuildError(
            f"failed to read configuration file: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ShardBuildError(
            f"invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ShardBuildError(
            f"configuration root must be a mapping: {resolved}"
        )

    return loaded


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_split(dataframe: pd.DataFrame) -> None:
    required_columns = {
        "identity_id",
        "image_id",
        "image_path",
        "experiment_group",
        "sample_role",
        "enrollment_candidate_rank",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ShardBuildError(
            "split dataframe is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ShardBuildError("split dataframe is empty")

    if dataframe["identity_id"].isna().any():
        raise ShardBuildError(
            "identity_id contains missing values"
        )

    if dataframe["image_id"].duplicated().any():
        duplicate_count = int(
            dataframe["image_id"]
            .duplicated(keep=False)
            .sum()
        )

        raise ShardBuildError(
            f"duplicate image IDs detected: {duplicate_count}"
        )

    group_counts = (
        dataframe.groupby("identity_id")[
            "experiment_group"
        ]
        .nunique()
    )

    if (group_counts != 1).any():
        raise ShardBuildError(
            "some identities belong to multiple experiment groups"
        )


def build_identity_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per identity.

    The image count is used as the primary balancing weight.
    """

    identity_statistics = (
        dataframe.groupby(
            [
                "identity_id",
                "experiment_group",
            ],
            as_index=False,
        )
        .agg(
            image_count=("image_id", "size"),
            enrollment_count=(
                "sample_role",
                lambda values: int(
                    (values == "enrollment").sum()
                ),
            ),
            candidate_count=(
                "enrollment_candidate_rank",
                lambda values: int(values.notna().sum()),
            ),
        )
    )

    return identity_statistics


def assign_identity_balanced_shards(
    identity_statistics: pd.DataFrame,
    num_shards: int,
) -> dict[str, int]:
    """
    Assign identities using greedy load balancing.

    Identities with more images are assigned first. Each identity is placed
    into the shard with the smallest current image count.

    Identity boundaries are never broken.
    """

    if num_shards <= 0:
        raise ShardBuildError(
            "num_shards must be positive"
        )

    if len(identity_statistics) < num_shards:
        raise ShardBuildError(
            "the number of identities must be at least num_shards"
        )

    ordered = identity_statistics.sort_values(
        by=[
            "image_count",
            "identity_id",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    )

    shard_image_loads = [0] * num_shards
    shard_identity_loads = [0] * num_shards

    assignments: dict[str, int] = {}

    for row in ordered.itertuples(index=False):
        # Primary criterion: total image count.
        # Secondary criterion: identity count.
        # Final criterion: shard index, ensuring deterministic behavior.
        target_shard = min(
            range(num_shards),
            key=lambda shard_index: (
                shard_image_loads[shard_index],
                shard_identity_loads[shard_index],
                shard_index,
            ),
        )

        identity_id = str(row.identity_id)
        image_count = int(row.image_count)

        assignments[identity_id] = target_shard
        shard_image_loads[target_shard] += image_count
        shard_identity_loads[target_shard] += 1

    return assignments


def validate_assignments(
    dataframe: pd.DataFrame,
    num_shards: int,
) -> None:
    if dataframe["shard_index"].isna().any():
        raise ShardBuildError(
            "some rows were not assigned to a shard"
        )

    invalid_shards = dataframe[
        ~dataframe["shard_index"].between(
            0,
            num_shards - 1,
        )
    ]

    if not invalid_shards.empty:
        raise ShardBuildError(
            "invalid shard indices were assigned"
        )

    identity_shard_counts = (
        dataframe.groupby("identity_id")[
            "shard_index"
        ]
        .nunique()
    )

    if (identity_shard_counts != 1).any():
        raise ShardBuildError(
            "some identities were split across multiple shards"
        )

    actual_shards = set(
        dataframe["shard_index"]
        .astype(int)
        .unique()
        .tolist()
    )

    expected_shards = set(range(num_shards))

    if actual_shards != expected_shards:
        raise ShardBuildError(
            f"expected shard indices {sorted(expected_shards)}, "
            f"found {sorted(actual_shards)}"
        )


def build_summary(
    dataframe: pd.DataFrame,
    num_shards: int,
    split_path: Path,
) -> dict[str, Any]:
    shard_summary: dict[str, dict[str, Any]] = {}

    for shard_index in range(num_shards):
        shard_frame = dataframe[
            dataframe["shard_index"] == shard_index
        ]

        group_identity_counts = (
            shard_frame.groupby(
                "experiment_group"
            )["identity_id"]
            .nunique()
            .sort_index()
            .to_dict()
        )

        group_image_counts = (
            shard_frame[
                "experiment_group"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        shard_summary[str(shard_index)] = {
            "identity_count": int(
                shard_frame["identity_id"].nunique()
            ),
            "image_count": int(len(shard_frame)),
            "enrollment_image_count": int(
                (
                    shard_frame["sample_role"]
                    == "enrollment"
                ).sum()
            ),
            "probe_image_count": int(
                (
                    shard_frame["sample_role"]
                    == "probe"
                ).sum()
            ),
            "candidate_image_count": int(
                shard_frame[
                    "enrollment_candidate_rank"
                ]
                .notna()
                .sum()
            ),
            "experiment_group_identity_counts": {
                str(key): int(value)
                for key, value
                in group_identity_counts.items()
            },
            "experiment_group_image_counts": {
                str(key): int(value)
                for key, value
                in group_image_counts.items()
            },
        }

    image_counts = [
        shard_summary[str(index)]["image_count"]
        for index in range(num_shards)
    ]

    identity_counts = [
        shard_summary[str(index)]["identity_count"]
        for index in range(num_shards)
    ]

    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "split_path": str(split_path.resolve()),
        "num_shards": num_shards,
        "strategy": "identity_balanced_greedy",
        "total_identity_count": int(
            dataframe["identity_id"].nunique()
        ),
        "total_image_count": int(len(dataframe)),
        "minimum_shard_image_count": int(
            min(image_counts)
        ),
        "maximum_shard_image_count": int(
            max(image_counts)
        ),
        "image_count_difference": int(
            max(image_counts) - min(image_counts)
        ),
        "minimum_shard_identity_count": int(
            min(identity_counts)
        ),
        "maximum_shard_identity_count": int(
            max(identity_counts)
        ),
        "shards": shard_summary,
    }


def save_summary(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    ensure_directory(output_path.parent)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info(
        "saved shard summary: %s",
        output_path,
    )


def build_shards(
    split_path: Path,
    output_dir: Path,
    output_summary_path: Path,
    num_shards: int,
    overwrite: bool,
) -> None:
    if not split_path.is_file():
        raise ShardBuildError(
            f"split file does not exist: {split_path}"
        )

    if num_shards <= 0:
        raise ShardBuildError(
            "num_shards must be positive"
        )

    ensure_directory(output_dir)

    expected_output_paths = [
        output_dir / f"shard_{index:03d}.parquet"
        for index in range(num_shards)
    ]

    existing_paths = [
        path
        for path in expected_output_paths
        if path.exists()
    ]

    if existing_paths and not overwrite:
        existing_text = "\n".join(
            str(path)
            for path in existing_paths
        )

        raise ShardBuildError(
            "some shard files already exist. "
            "Use --overwrite to replace them:\n"
            f"{existing_text}"
        )

    LOGGER.info(
        "loading split: %s",
        split_path,
    )

    dataframe = pd.read_parquet(split_path)

    validate_split(dataframe)

    identity_statistics = build_identity_statistics(
        dataframe
    )

    LOGGER.info(
        "total identities: %s",
        f"{len(identity_statistics):,}",
    )

    LOGGER.info(
        "total images: %s",
        f"{len(dataframe):,}",
    )

    assignments = assign_identity_balanced_shards(
        identity_statistics=identity_statistics,
        num_shards=num_shards,
    )

    dataframe = dataframe.copy()

    dataframe["shard_index"] = (
        dataframe["identity_id"]
        .astype(str)
        .map(assignments)
        .astype("int16")
    )

    validate_assignments(
        dataframe=dataframe,
        num_shards=num_shards,
    )

    dataframe = dataframe.sort_values(
        by=[
            "shard_index",
            "experiment_group",
            "identity_id",
            "image_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    for shard_index in range(num_shards):
        shard_frame = dataframe[
            dataframe["shard_index"]
            == shard_index
        ].copy()

        output_path = (
            output_dir
            / f"shard_{shard_index:03d}.parquet"
        )

        shard_frame.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        LOGGER.info(
            "saved shard %d: identities=%s, images=%s, path=%s",
            shard_index,
            f"{shard_frame['identity_id'].nunique():,}",
            f"{len(shard_frame):,}",
            output_path,
        )

    summary = build_summary(
        dataframe=dataframe,
        num_shards=num_shards,
        split_path=split_path,
    )

    save_summary(
        summary=summary,
        output_path=output_summary_path,
    )

    LOGGER.info(
        "image-count difference between largest and smallest shard: %s",
        f"{summary['image_count_difference']:,}",
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_yaml(args.config)

        data_config = config["data"]
        sharding_config = config["sharding"]

        split_path = (
            args.split
            if args.split is not None
            else Path(data_config["split_path"])
        ).expanduser().resolve()

        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path(data_config["shard_dir"])
        ).expanduser().resolve()

        num_shards = (
            args.num_shards
            if args.num_shards is not None
            else int(sharding_config["num_shards"])
        )

        output_summary_path = Path(
            sharding_config["output_summary"]
        ).expanduser().resolve()

        strategy = str(
            sharding_config.get(
                "strategy",
                "identity_balanced",
            )
        )

        if strategy != "identity_balanced":
            raise ShardBuildError(
                f"unsupported sharding strategy: {strategy}"
            )

        build_shards(
            split_path=split_path,
            output_dir=output_dir,
            output_summary_path=output_summary_path,
            num_shards=num_shards,
            overwrite=args.overwrite,
        )

        return 0

    except KeyError as exc:
        LOGGER.error(
            "missing required configuration key: %s",
            exc,
        )
        return 2

    except ShardBuildError as exc:
        LOGGER.error("%s", exc)
        return 1

    except KeyboardInterrupt:
        LOGGER.warning(
            "shard generation interrupted by user"
        )
        return 130

    except Exception:
        LOGGER.exception(
            "unexpected shard-generation error"
        )
        return 99


if __name__ == "__main__":
    sys.exit(main())