from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


class RepairError(RuntimeError):
    """Raised when Hamming trial metadata repair fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair truncated probe_identity_id values in "
            "Hamming trial Parquet files using probe_image_id."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def parse_identity_from_image_id(
    probe_image_id: str,
) -> str:
    value = str(probe_image_id)

    parts = value.split("__")

    if len(parts) < 3:
        raise RepairError(
            f"Unexpected probe_image_id format: {value}"
        )

    identity_id = parts[1]

    if len(identity_id) != 7:
        raise RepairError(
            f"Unexpected identity length parsed from "
            f"{value}: {identity_id}"
        )

    if not identity_id.startswith("n"):
        raise RepairError(
            f"Unexpected identity prefix parsed from "
            f"{value}: {identity_id}"
        )

    return identity_id


def main() -> int:
    args = parse_args()

    try:
        input_path = args.input.expanduser().resolve()

        if not input_path.is_file():
            raise RepairError(
                f"Input file does not exist: {input_path}"
            )

        if args.output is None:
            output_path = input_path
        else:
            output_path = (
                args.output
                .expanduser()
                .resolve()
            )

        if (
            output_path.exists()
            and output_path != input_path
            and not args.overwrite
        ):
            raise RepairError(
                f"Output already exists: {output_path}. "
                "Use --overwrite."
            )

        dataframe = pd.read_parquet(
            input_path
        )

        required_columns = {
            "trial_type",
            "claimant_identity_id",
            "probe_identity_id",
            "probe_image_id",
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise RepairError(
                f"Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        original_probe_identity = (
            dataframe["probe_identity_id"]
            .astype(str)
        )

        parsed_probe_identity = (
            dataframe["probe_image_id"]
            .astype(str)
            .map(parse_identity_from_image_id)
        )

        truncated_mask = (
            original_probe_identity.str.len()
            != 7
        )

        repair_count = int(
            truncated_mask.sum()
        )

        print(
            "Rows:",
            len(dataframe),
        )

        print(
            "Truncated probe identities:",
            repair_count,
        )

        if repair_count == 0:
            print(
                "No truncated probe identities found."
            )

        dataframe[
            "probe_identity_id_original"
        ] = original_probe_identity

        dataframe.loc[
            truncated_mask,
            "probe_identity_id",
        ] = parsed_probe_identity[
            truncated_mask
        ]

        dataframe[
            "probe_identity_id"
        ] = dataframe[
            "probe_identity_id"
        ].astype("string")

        repaired_lengths = (
            dataframe[
                "probe_identity_id"
            ]
            .astype(str)
            .str.len()
        )

        invalid_length_count = int(
            (
                repaired_lengths != 7
            ).sum()
        )

        if invalid_length_count > 0:
            raise RepairError(
                "Repair left invalid probe identity values: "
                f"{invalid_length_count}"
            )

        genuine = dataframe[
            dataframe["trial_type"]
            == "genuine"
        ]

        genuine_mismatch_count = int(
            (
                genuine[
                    "claimant_identity_id"
                ].astype(str)
                != genuine[
                    "probe_identity_id"
                ].astype(str)
            ).sum()
        )

        if genuine_mismatch_count > 0:
            raise RepairError(
                "Genuine trials contain claimant/probe "
                "identity mismatches after repair: "
                f"{genuine_mismatch_count}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        print(
            "Saved:",
            output_path,
        )

        print(
            "Genuine claimant/probe mismatches:",
            genuine_mismatch_count,
        )

        print(
            "Repair completed successfully."
        )

        return 0

    except RepairError as exc:
        print(
            f"Repair failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Repair interrupted",
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