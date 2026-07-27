from __future__ import annotations

import argparse
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


def assign_enrollment_probe_roles(
    dataframe: pd.DataFrame,
    enrollment_count: int,
    seed: int,
) -> pd.DataFrame:
    """
    Assign exactly `enrollment_count` images per identity to enrollment.

    Remaining images are assigned as probes.

    The image selection is deterministic for a fixed seed.
    """

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

        if image_count <= enrollment_count:
            raise DatasetSplitError(
                f"identity {identity_id} has {image_count} images, "
                f"but enrollment_count={enrollment_count}"
            )

        identity_seed = (
            seed
            + int.from_bytes(
                str(identity_id).encode("utf-8"),
                byteorder="little",
                signed=False,
            )
        ) % (2**32)

        rng = np.random.default_rng(identity_seed)

        selected_positions = rng.choice(
            image_count,
            size=enrollment_count,
            replace=False,
        )

        identity_frame["sample_role"] = "probe"

        selected_index = identity_frame.index[
            selected_positions
        ]

        identity_frame.loc[
            selected_index,
            "sample_role",
        ] = "enrollment"

        identity_frame["enrollment_rank"] = pd.NA

        enrollment_rows = identity_frame.loc[
            selected_index
        ].sort_values(
            by="image_id",
            kind="stable",
        )

        for rank, row_index in enumerate(
            enrollment_rows.index,
            start=1,
        ):
            identity_frame.loc[
                row_index,
                "enrollment_rank",
            ] = rank

        output_frames.append(identity_frame)

    output = pd.concat(
        output_frames,
        ignore_index=True,
    )

    output["enrollment_rank"] = (
        output["enrollment_rank"]
        .astype("Int64")
    )

    return output


def validate_split(
    dataframe: pd.DataFrame,
    expected_group_counts: dict[str, int],
    enrollment_count: int,
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

    enrollment_counts = (
        dataframe[
            dataframe["sample_role"] == "enrollment"
        ]
        .groupby("identity_id")
        .size()
    )

    invalid_enrollment_counts = enrollment_counts[
        enrollment_counts != enrollment_count
    ]

    if not invalid_enrollment_counts.empty:
        raise DatasetSplitError(
            "some identities do not have the required number of "
            f"enrollment images:\n{invalid_enrollment_counts.head(20)}"
        )

    probe_counts = (
        dataframe[
            dataframe["sample_role"] == "probe"
        ]
        .groupby("identity_id")
        .size()
    )

    if (probe_counts <= 0).any():
        raise DatasetSplitError(
            "some identities have no probe images"
        )

    group_membership_counts = (
        dataframe.groupby("identity_id")["experiment_group"]
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
    enrollment_count: int,
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
            "enrollment_image_count": int(
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
        "enrollment_count_per_identity": enrollment_count,
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
    enrollment_count: int,
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

    split_dataframe = assign_enrollment_probe_roles(
        dataframe=filtered,
        enrollment_count=enrollment_count,
        seed=seed,
    )

    validate_split(
        dataframe=split_dataframe,
        expected_group_counts={
            "background": background_count,
            "development": development_count,
            "evaluation": evaluation_count,
        },
        enrollment_count=enrollment_count,
    )

    preferred_columns = [
        "identity_id",
        "experiment_group",
        "sample_role",
        "enrollment_rank",
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
            "sample_role",
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
        minimum_images_per_identity=minimum_images_per_identity,
        enrollment_count=enrollment_count,
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

    for group_name, group_data in summary[
        "experiment_groups"
    ].items():
        LOGGER.info(
            "%s: identities=%s, images=%s, enrollment=%s, probe=%s",
            group_name,
            f"{group_data['identity_count']:,}",
            f"{group_data['image_count']:,}",
            f"{group_data['enrollment_image_count']:,}",
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
            enrollment_count=int(
                split_config["enrollment_count"]
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