from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import galois
import pandas as pd
import yaml


class BCHParameterDiscoveryError(RuntimeError):
    """Raised when BCH parameter discovery fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover valid binary BCH parameters for the "
            "fuzzy-extractor experiment."
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

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise BCHParameterDiscoveryError(
            f"Configuration does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise BCHParameterDiscoveryError(
            f"Failed to read configuration: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise BCHParameterDiscoveryError(
            f"Invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise BCHParameterDiscoveryError(
            "Configuration root must be a mapping"
        )

    return loaded


def construct_bch_from_t(
    *,
    codeword_length: int,
    requested_t: int,
) -> galois.BCH:
    """
    Construct a narrow-sense binary BCH code using designed
    distance d = 2t + 1.

    The resulting BCH object must be inspected because its actual
    minimum distance or correction capability may differ from the
    requested design parameter.
    """

    if requested_t <= 0:
        raise BCHParameterDiscoveryError(
            "requested_t must be positive"
        )

    designed_distance = (
        2 * requested_t + 1
    )

    try:
        code = galois.BCH(
            codeword_length,
            d=designed_distance,
            field=galois.GF(2),
            systematic=True,
        )
    except Exception as exc:
        raise BCHParameterDiscoveryError(
            f"Could not construct BCH code for "
            f"n={codeword_length}, "
            f"requested_t={requested_t}, "
            f"designed_distance={designed_distance}: {exc}"
        ) from exc

    return code


def get_integer_attribute(
    instance: Any,
    name: str,
) -> int | None:
    value = getattr(
        instance,
        name,
        None,
    )

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_parameters(
    *,
    codeword_length: int,
    candidate_t_values: list[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for requested_t in candidate_t_values:
        designed_distance = (
            2 * requested_t + 1
        )

        record: dict[str, Any] = {
            "requested_t": requested_t,
            "requested_designed_distance": (
                designed_distance
            ),
            "construction_status": "failed",
            "failure_reason": None,
        }

        try:
            code = construct_bch_from_t(
                codeword_length=codeword_length,
                requested_t=requested_t,
            )

            n = int(code.n)
            k = int(code.k)
            d = int(code.d)

            actual_t = (
                d - 1
            ) // 2

            parity_bits = n - k
            code_rate = (
                k / n
                if n > 0
                else 0.0
            )

            generator_degree = (
                get_integer_attribute(
                    code.generator_poly,
                    "degree",
                )
            )

            record.update(
                {
                    "construction_status": "success",
                    "n": n,
                    "k": k,
                    "d": d,
                    "actual_t": actual_t,
                    "parity_bits": parity_bits,
                    "code_rate": code_rate,
                    "generator_degree": (
                        generator_degree
                    ),
                    "message_bytes_floor": k // 8,
                    "message_entropy_bits_upper_bound": k,
                    "meets_requested_t": (
                        actual_t >= requested_t
                    ),
                }
            )

        except BCHParameterDiscoveryError as exc:
            record["failure_reason"] = str(exc)

        records.append(record)

    return records


def main() -> int:
    args = parse_args()

    try:
        config = load_yaml(
            args.config
        )

        fuzzy_config = config[
            "fuzzy_extractor"
        ]

        output_dir = Path(
            fuzzy_config["output_dir"]
        ).expanduser().resolve()

        codeword_length = int(
            fuzzy_config[
                "bch_codeword_length"
            ]
        )

        candidate_t_values = sorted(
            {
                int(value)
                for value in fuzzy_config[
                    "candidate_correction_capabilities"
                ]
            }
        )

        if codeword_length <= 0:
            raise BCHParameterDiscoveryError(
                "bch_codeword_length must be positive"
            )

        if not candidate_t_values:
            raise BCHParameterDiscoveryError(
                "No candidate correction capabilities configured"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_path = (
            output_dir
            / "bch_parameter_candidates.csv"
        )

        json_path = (
            output_dir
            / "bch_parameter_candidates.json"
        )

        if (
            not args.overwrite
            and (
                csv_path.exists()
                or json_path.exists()
            )
        ):
            raise BCHParameterDiscoveryError(
                "Output already exists. Use --overwrite."
            )

        records = discover_parameters(
            codeword_length=codeword_length,
            candidate_t_values=(
                candidate_t_values
            ),
        )

        dataframe = pd.DataFrame.from_records(
            records
        )

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        successful = dataframe[
            dataframe["construction_status"]
            == "success"
        ].copy()

        payload = {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "library": {
                "name": "galois",
                "version": galois.__version__,
            },
            "field": "GF(2)",
            "systematic": True,
            "codeword_length": codeword_length,
            "records": records,
        }

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

        print(
            "BCH codeword length:",
            codeword_length,
        )

        print(
            "Candidate t values:",
            candidate_t_values,
        )

        print()

        if successful.empty:
            print(
                "No valid BCH parameter set was found."
            )
        else:
            columns = [
                "requested_t",
                "n",
                "k",
                "d",
                "actual_t",
                "parity_bits",
                "code_rate",
                "message_bytes_floor",
                "meets_requested_t",
            ]

            print(
                successful[
                    columns
                ].to_string(
                    index=False,
                    formatters={
                        "code_rate": (
                            "{:.6f}".format
                        ),
                    },
                )
            )

        failed = dataframe[
            dataframe["construction_status"]
            == "failed"
        ]

        if not failed.empty:
            print()
            print("Failed constructions:")

            print(
                failed[
                    [
                        "requested_t",
                        "failure_reason",
                    ]
                ].to_string(
                    index=False
                )
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

    except BCHParameterDiscoveryError as exc:
        print(
            f"BCH parameter discovery failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "BCH parameter discovery interrupted",
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