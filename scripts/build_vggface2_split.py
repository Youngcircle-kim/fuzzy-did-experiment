from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger("build_vggface2_split")


class DatasetSplitError(RuntimeError):
    """Raised when the dataset split cannot be generated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build identity-level background, development, and evaluation "
            "splits for VGGFace2."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/vggface2.yaml"),
        help="Path to the VGGFace2 YAML configuration.",
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Override the input Parquet index path.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output split Parquet path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing split file.",
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
        raise DatasetSplitError(
            f"configuration file does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise DatasetSplitError(
            f"failed to read configuration file: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise DatasetSplitError(
            f"invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise DatasetSplitError(
            f"configuration root must be a mapping: {resolved}"
        )

    return loaded


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_index(dataframe: pd.DataFrame) -> None:
    required_columns = {
        "identity_id",
        "subset",
        "image_id",
        "image_path",
        "relative_path",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise DatasetSplitError(
            "input index is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise DatasetSplitError("input index is empty")

    if dataframe["image_id"].duplicated().any():
        duplicate_count = int(
            dataframe["image_id"].duplicated(keep=False).sum()
        )

        raise DatasetSplitError(
            f"duplicate image IDs detected: {duplicate_count}"
        )

    if dataframe["identity_id"].isna().any():
        raise DatasetSplitError(
            "identity_id contains missing values"
        )


def filter_eligible_identities(
    dataframe: pd.DataFrame,
    minimum_images_per_identity: int,
) -> tuple[pd.DataFrame, pd.Series]:
    identity_counts = dataframe.groupby("identity_id").size()

    eligible_identity_ids = identity_counts[
        identity_counts >= minimum_images_per_identity
    ].index

    filtered = dataframe[
        dataframe["identity_id"].isin(eligible_identity_ids)
    ].copy()

    excluded_identity_count = (
        dataframe["identity_id"].nunique()
        - len(eligible_identity_ids)
    )

    LOGGER.info(
        "eligible identities: %s",
        f"{len(eligible_identity_ids):,}",
    )

    LOGGER.info(
        "excluded identities: %s",
        f"{excluded_identity_count:,}",
    )

    return filtered, identity_counts.loc[eligible_identity_ids]


def assign_identity_groups(
    identity_ids: list[str],
    background_count: int,
    development_count: int,
    evaluation_count: int,
    seed: int,
) -> dict[str, str]:
    requested_total = (
        background_count
        + development_count
        + evaluation_count
    )

    available_total = len(identity_ids)

    if requested_total != available_total:
        raise DatasetSplitError(
            "identity-group counts must exactly match the number of eligible "
            f"identities: requested={requested_total}, "
            f"available={available_total}"
        )

    rng = np.random.default_rng(seed)

    shuffled_ids = np.array(
        sorted(identity_ids),
        dtype=object,
    )

    rng.shuffle(shuffled_ids)

    background_end = background_count
    development_end = (
        background_count
        + development_count
    )

    background_ids = shuffled_ids[:background_end]
    development_ids = shuffled_ids[
        background_end:development_end
    ]
    evaluation_ids = shuffled_ids[
        development_end:
    ]

    assignments: dict[str, str] = {}

    for identity_id in background_ids:
        assignments[str(identity_id)] = "background"

    for identity_id in development_ids:
        assignments[str(identity_id)] = "development"

    for identity_id in evaluation_ids:
        assignments[str(identity_id)] = "evaluation"

    return assignments


def stable_identity_seed(
    identity_id: str,
    global_seed: int,
) -> int:
    digest = hashlib.sha256(
        identity_id.encode("utf-8")
    ).digest()

    identity_offset = int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )

    return (global_seed + identity_offset) % (2**32)


def assign_enrollment_candidates(
    dataframe: pd.DataFrame,
    candidate_count: int,
    default_enrollment_count: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select deterministic enrollment candidates for every identity.

    For each identity:
    - Randomly select exactly `candidate_count` images without replacement.
    - Assign candidate ranks from 1 to `candidate_count`.
    - Mark ranks 1 through `default_enrollment_count` as enrollment.
    - Mark all remaining images as probe.

    Candidate selection and rank order are deterministic for a fixed
    global seed and identity ID.

    Args:
        dataframe:
            Image-level dataframe. It must contain at least:
            - identity_id
            - image_id

        candidate_count:
            Maximum number of enrollment candidates assigned per identity.
            For example, 10 supports enrollment-count experiments using
            rank prefixes 1/3/5/7/10.

        default_enrollment_count:
            Number of candidates marked as enrollment in the default split.
            This value must not exceed `candidate_count`.

        seed:
            Global random seed.

    Returns:
        A new dataframe containing:
            - enrollment_candidate_rank: nullable integer rank 1..N
            - sample_role: "enrollment" or "probe"

    Raises:
        DatasetSplitError:
            If arguments are invalid or an identity has insufficient images.
    """

    if dataframe.empty:
        raise DatasetSplitError(
            "cannot assign enrollment candidates to an empty dataframe"
        )

    required_columns = {
        "identity_id",
        "image_id",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise DatasetSplitError(
            "dataframe is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if candidate_count <= 0:
        raise DatasetSplitError(
            "candidate_count must be positive"
        )

    if default_enrollment_count <= 0:
        raise DatasetSplitError(
            "default_enrollment_count must be positive"
        )

    if default_enrollment_count > candidate_count:
        raise DatasetSplitError(
            "default_enrollment_count cannot exceed candidate_count: "
            f"default={default_enrollment_count}, "
            f"candidate={candidate_count}"
        )

    output_frames: list[pd.DataFrame] = []

    for identity_id, identity_frame in dataframe.groupby(
        "identity_id",
        sort=True,
    ):
        identity_frame = identity_frame.sort_values(
            by="image_id",
            kind="stable",
        ).copy()

        image_count = len(identity_frame)

        # 최소 한 장 이상의 probe를 남긴다.
        if image_count <= candidate_count:
            raise DatasetSplitError(
                f"identity {identity_id} has {image_count} images, "
                f"but candidate_count={candidate_count}. "
                "At least candidate_count + 1 images are required "
                "to preserve one probe image."
            )

        identity_seed = stable_identity_seed(
            identity_id=str(identity_id),
            global_seed=seed,
        )

        rng = np.random.default_rng(identity_seed)

        # identity_frame은 image_id 기준으로 정렬되어 있으므로,
        # 위치 기반 random selection도 동일 seed에서 재현 가능하다.
        candidate_positions = rng.choice(
            image_count,
            size=candidate_count,
            replace=False,
        )

        # rng.choice가 반환한 순서를 그대로 candidate rank로 사용한다.
        # 따라서 rank 1~3, 1~5, 1~7, 1~10은 동일 후보 순서의 prefix가 된다.
        candidate_indices = identity_frame.index[
            candidate_positions
        ]

        identity_frame[
            "enrollment_candidate_rank"
        ] = pd.Series(
            pd.NA,
            index=identity_frame.index,
            dtype="Int64",
        )

        identity_frame["sample_role"] = "probe"

        for rank, row_index in enumerate(
            candidate_indices,
            start=1,
        ):
            identity_frame.loc[
                row_index,
                "enrollment_candidate_rank",
            ] = rank

            if rank <= default_enrollment_count:
                identity_frame.loc[
                    row_index,
                    "sample_role",
                ] = "enrollment"

        # identity 단위 내부 검증
        actual_candidate_count = int(
            identity_frame[
                "enrollment_candidate_rank"
            ]
            .notna()
            .sum()
        )

        if actual_candidate_count != candidate_count:
            raise DatasetSplitError(
                f"candidate assignment failed for {identity_id}: "
                f"expected={candidate_count}, "
                f"actual={actual_candidate_count}"
            )

        actual_enrollment_count = int(
            (
                identity_frame["sample_role"]
                == "enrollment"
            ).sum()
        )

        if actual_enrollment_count != default_enrollment_count:
            raise DatasetSplitError(
                f"default enrollment assignment failed for {identity_id}: "
                f"expected={default_enrollment_count}, "
                f"actual={actual_enrollment_count}"
            )

        actual_ranks = sorted(
            identity_frame[
                "enrollment_candidate_rank"
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        expected_ranks = list(
            range(1, candidate_count + 1)
        )

        if actual_ranks != expected_ranks:
            raise DatasetSplitError(
                f"invalid candidate ranks for {identity_id}: "
                f"expected={expected_ranks}, "
                f"actual={actual_ranks}"
            )

        output_frames.append(identity_frame)

    if not output_frames:
        raise DatasetSplitError(
            "no identity frames were generated"
        )

    output = pd.concat(
        output_frames,
        ignore_index=True,
    )

    output["enrollment_candidate_rank"] = (
        output["enrollment_candidate_rank"]
        .astype("Int64")
    )

    output["sample_role"] = (
        output["sample_role"]
        .astype("string")
    )

    return output

def validate_split(
    dataframe: pd.DataFrame,
    expected_group_counts: dict[str, int],
    enrollment_candidate_count: int,
    default_enrollment_count: int,
) -> None:
    actual_group_counts = (
        dataframe.groupby("experiment_group")["identity_id"]
        .nunique()
        .to_dict()
    )

    for group_name, expected_count in expected_group_counts.items():
        actual_count = int(
            actual_group_counts.get(group_name, 0)
        )

        if actual_count != expected_count:
            raise DatasetSplitError(
                f"identity count mismatch for {group_name}: "
                f"expected={expected_count}, actual={actual_count}"
            )

    # 각 identity가 정확히 N개의 enrollment candidate를 가져야 함.
    candidate_counts = (
        dataframe[
            dataframe["enrollment_candidate_rank"].notna()
        ]
        .groupby("identity_id")
        .size()
    )

    invalid_candidate_counts = candidate_counts[
        candidate_counts != enrollment_candidate_count
    ]

    if not invalid_candidate_counts.empty:
        raise DatasetSplitError(
            "some identities do not have the required number of "
            "enrollment candidates:\n"
            f"{invalid_candidate_counts.head(20)}"
        )

    # 후보 rank가 identity별로 1...N인지 확인.
    expected_ranks = list(
        range(1, enrollment_candidate_count + 1)
    )

    for identity_id, identity_frame in dataframe.groupby(
        "identity_id",
        sort=False,
    ):
        actual_ranks = sorted(
            identity_frame[
                "enrollment_candidate_rank"
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        if actual_ranks != expected_ranks:
            raise DatasetSplitError(
                f"invalid enrollment candidate ranks for "
                f"{identity_id}: {actual_ranks}"
            )

    enrollment_counts = (
        dataframe[
            dataframe["sample_role"] == "enrollment"
        ]
        .groupby("identity_id")
        .size()
    )

    invalid_enrollment_counts = enrollment_counts[
        enrollment_counts != default_enrollment_count
    ]

    if not invalid_enrollment_counts.empty:
        raise DatasetSplitError(
            "some identities do not have the required number of "
            "default enrollment images:\n"
            f"{invalid_enrollment_counts.head(20)}"
        )

    probe_counts = (
        dataframe[
            dataframe["sample_role"] == "probe"
        ]
        .groupby("identity_id")
        .size()
    )

    if len(probe_counts) != dataframe["identity_id"].nunique():
        raise DatasetSplitError(
            "some identities have no probe images"
        )

    if (probe_counts <= 0).any():
        raise DatasetSplitError(
            "some identities have no probe images"
        )

    group_membership_counts = (
        dataframe.groupby("identity_id")[
            "experiment_group"
        ]
        .nunique()
    )

    if (group_membership_counts != 1).any():
        raise DatasetSplitError(
            "some identities belong to multiple experiment groups"
        )

def build_summary(
    dataframe: pd.DataFrame,
    seed: int,
    minimum_images_per_identity: int,
    enrollment_candidate_count: int,
    default_enrollment_count: int,
) -> dict[str, Any]:
    group_summary: dict[str, dict[str, Any]] = {}

    for group_name, group_frame in dataframe.groupby(
        "experiment_group",
        sort=True,
    ):
        group_identity_counts = (
            group_frame.groupby("identity_id").size()
        )

        group_summary[str(group_name)] = {
            "identity_count": int(
                group_frame["identity_id"].nunique()
            ),
            "image_count": int(len(group_frame)),
            "enrollment_candidate_image_count": int(
                group_frame[
                    "enrollment_candidate_rank"
                ]
                .notna()
                .sum()
            ),
            "default_enrollment_image_count": int(
                (
                    group_frame["sample_role"]
                    == "enrollment"
                ).sum()
            ),
            "probe_image_count": int(
                (
                    group_frame["sample_role"]
                    == "probe"
                ).sum()
            ),
            "minimum_images_per_identity": int(
                group_identity_counts.min()
            ),
            "maximum_images_per_identity": int(
                group_identity_counts.max()
            ),
            "mean_images_per_identity": float(
                group_identity_counts.mean()
            ),
        }

    source_subset_counts = (
        dataframe["subset"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "dataset_name": "vggface2",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": seed,
        "minimum_images_per_identity": (
            minimum_images_per_identity
        ),
        "enrollment_candidate_count_per_identity": (
            enrollment_candidate_count
        ),
        "default_enrollment_count_per_identity": (
            default_enrollment_count
        ),
        "total_identity_count": int(
            dataframe["identity_id"].nunique()
        ),
        "total_image_count": int(len(dataframe)),
        "source_subset_image_counts": {
            str(key): int(value)
            for key, value in source_subset_counts.items()
        },
        "experiment_groups": group_summary,
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

    LOGGER.info("saved split summary: %s", output_path)

def build_split(
    index_path: Path,
    output_split_path: Path,
    output_summary_path: Path,
    seed: int,
    minimum_images_per_identity: int,
    background_count: int,
    development_count: int,
    evaluation_count: int,
    enrollment_candidate_count: int,
    default_enrollment_count: int,
    overwrite: bool,
) -> None:
    if not index_path.is_file():
        raise DatasetSplitError(
            f"input index does not exist: {index_path}"
        )

    if output_split_path.exists() and not overwrite:
        raise DatasetSplitError(
            f"split file already exists: {output_split_path}\n"
            "Use --overwrite to replace it."
        )

    if enrollment_candidate_count <= 0:
        raise DatasetSplitError(
            "enrollment_candidate_count must be positive"
        )

    if default_enrollment_count <= 0:
        raise DatasetSplitError(
            "default_enrollment_count must be positive"
        )

    if default_enrollment_count > enrollment_candidate_count:
        raise DatasetSplitError(
            "default_enrollment_count cannot exceed "
            "enrollment_candidate_count"
        )

    # 후보 10장과 최소 1장의 probe가 필요함.
    required_minimum = enrollment_candidate_count + 1

    if minimum_images_per_identity < required_minimum:
        raise DatasetSplitError(
            "minimum_images_per_identity must be at least "
            f"{required_minimum} when enrollment_candidate_count="
            f"{enrollment_candidate_count}"
        )

    LOGGER.info("loading index: %s", index_path)

    dataframe = pd.read_parquet(index_path)

    validate_index(dataframe)

    filtered, identity_counts = filter_eligible_identities(
        dataframe=dataframe,
        minimum_images_per_identity=minimum_images_per_identity,
    )

    identity_ids = sorted(
        identity_counts.index.astype(str).tolist()
    )

    assignments = assign_identity_groups(
        identity_ids=identity_ids,
        background_count=background_count,
        development_count=development_count,
        evaluation_count=evaluation_count,
        seed=seed,
    )

    filtered["experiment_group"] = (
        filtered["identity_id"]
        .astype(str)
        .map(assignments)
    )

    if filtered["experiment_group"].isna().any():
        raise DatasetSplitError(
            "failed to assign experiment group to some identities"
        )

    split_dataframe = assign_enrollment_candidates(
        dataframe=filtered,
        candidate_count=enrollment_candidate_count,
        default_enrollment_count=default_enrollment_count,
        seed=seed,
    )

    validate_split(
        dataframe=split_dataframe,
        expected_group_counts={
            "background": background_count,
            "development": development_count,
            "evaluation": evaluation_count,
        },
        enrollment_candidate_count=(
            enrollment_candidate_count
        ),
        default_enrollment_count=(
            default_enrollment_count
        ),
    )

    preferred_columns = [
        "identity_id",
        "experiment_group",
        "sample_role",
        "enrollment_candidate_rank",
        "subset",
        "image_id",
        "image_path",
        "relative_path",
        "extension",
        "file_size_bytes",
    ]

    remaining_columns = [
        column
        for column in split_dataframe.columns
        if column not in preferred_columns
    ]

    split_dataframe = split_dataframe[
        preferred_columns + remaining_columns
    ]

    split_dataframe = split_dataframe.sort_values(
        by=[
            "experiment_group",
            "identity_id",
            "image_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    ensure_directory(output_split_path.parent)

    split_dataframe.to_parquet(
        output_split_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    LOGGER.info(
        "saved split: %s",
        output_split_path,
    )

    summary = build_summary(
        dataframe=split_dataframe,
        seed=seed,
        minimum_images_per_identity=(
            minimum_images_per_identity
        ),
        enrollment_candidate_count=(
            enrollment_candidate_count
        ),
        default_enrollment_count=(
            default_enrollment_count
        ),
    )

    save_summary(
        summary=summary,
        output_path=output_summary_path,
    )

    LOGGER.info(
        "total identities: %s",
        f"{summary['total_identity_count']:,}",
    )

    LOGGER.info(
        "total images: %s",
        f"{summary['total_image_count']:,}",
    )

    LOGGER.info(
        "enrollment candidates per identity: %d",
        enrollment_candidate_count,
    )

    LOGGER.info(
        "default enrollment images per identity: %d",
        default_enrollment_count,
    )

    for group_name, group_data in summary[
        "experiment_groups"
    ].items():
        LOGGER.info(
            "%s: identities=%s, images=%s, "
            "candidates=%s, default_enrollment=%s, probe=%s",
            group_name,
            f"{group_data['identity_count']:,}",
            f"{group_data['image_count']:,}",
            f"{group_data['enrollment_candidate_image_count']:,}",
            f"{group_data['default_enrollment_image_count']:,}",
            f"{group_data['probe_image_count']:,}",
        )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_yaml(args.config)

        dataset_config = config["dataset"]
        indexing_config = dataset_config["indexing"]
        split_config = dataset_config["split"]

        identity_groups = split_config["identity_groups"]
        enrollment_candidate_count = int(
            split_config["enrollment_candidate_count"]
        )

        default_enrollment_count = int(
            split_config["default_enrollment_count"]
        )
        index_path = (
            args.index
            if args.index is not None
            else Path(indexing_config["output_index"])
        ).expanduser().resolve()

        output_split_path = (
            args.output
            if args.output is not None
            else Path(split_config["output_split"])
        ).expanduser().resolve()

        output_summary_path = Path(
            split_config["output_summary"]
        ).expanduser().resolve()

        build_split(
            index_path=index_path,
            output_split_path=output_split_path,
            output_summary_path=output_summary_path,
            seed=int(split_config["seed"]),
            minimum_images_per_identity=int(
                split_config["minimum_images_per_identity"]
            ),
            background_count=int(
                identity_groups["background"]
            ),
            development_count=int(
                identity_groups["development"]
            ),
            evaluation_count=int(
                identity_groups["evaluation"]
            ),
            enrollment_candidate_count=(
                enrollment_candidate_count
            ),
            default_enrollment_count=(
                default_enrollment_count
            ),
            overwrite=args.overwrite,
        )
        return 0
    except KeyError as exc:
        LOGGER.error(
            "missing required configuration key: %s",
            exc,
        )
        return 2

    except DatasetSplitError as exc:
        LOGGER.error("%s", exc)
        return 1

    except KeyboardInterrupt:
        LOGGER.warning("split generation interrupted by user")
        return 130

    except Exception:
        LOGGER.exception(
            "unexpected split-generation error"
        )
        return 99

if __name__ == "__main__":
    sys.exit(main())