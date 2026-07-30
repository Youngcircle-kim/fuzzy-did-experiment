from __future__ import annotations

import argparse
import sys

import numpy as np

from fuzzy_did.fuzzy_extractor import (
    BCHCodeOffsetFuzzyExtractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--t",
        type=int,
        default=7,
    )

    return parser.parse_args()


def flip_exact_bits(
    template: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    if count < 0 or count > len(template):
        raise ValueError(
            "Invalid flip count"
        )

    result = template.copy()

    rng = np.random.default_rng(seed)

    positions = rng.choice(
        len(result),
        size=count,
        replace=False,
    )

    result[positions] ^= 1

    return result


def main() -> int:
    args = parse_args()

    extractor = BCHCodeOffsetFuzzyExtractor(
        codeword_length=127,
        correction_capability=args.t,
        digest_algorithm="sha256",
    )

    parameters = extractor.parameters

    print("BCH parameters:", parameters)

    rng = np.random.default_rng(42)

    enrollment_template = rng.integers(
        0,
        2,
        size=parameters.n,
        dtype=np.uint8,
    )

    record = extractor.gen(
        enrollment_template
    )

    within_radius = flip_exact_bits(
        enrollment_template,
        parameters.t,
        seed=100,
    )

    within_result = extractor.rep(
        within_radius,
        record.helper_data,
        expected_key_digest=record.key_digest,
    )

    outside_radius = flip_exact_bits(
        enrollment_template,
        parameters.t + 1,
        seed=101,
    )

    outside_result = extractor.rep(
        outside_radius,
        record.helper_data,
        expected_key_digest=record.key_digest,
    )

    exact_result = extractor.rep(
        enrollment_template,
        record.helper_data,
        expected_key_digest=record.key_digest,
    )

    print()
    print("Exact template")
    print(
        " decode:",
        exact_result.decode_succeeded,
    )
    print(
        " key match:",
        exact_result.key_matched,
    )

    print()
    print(
        f"Exactly t={parameters.t} errors"
    )
    print(
        " decode:",
        within_result.decode_succeeded,
    )
    print(
        " key match:",
        within_result.key_matched,
    )
    print(
        " corrected errors:",
        within_result.decoder_error_count,
    )

    print()
    print(
        f"t+1={parameters.t + 1} errors"
    )
    print(
        " decode:",
        outside_result.decode_succeeded,
    )
    print(
        " key match:",
        outside_result.key_matched,
    )
    print(
        " decoder result:",
        outside_result.decoder_error_count,
    )

    if not exact_result.key_matched:
        print(
            "Exact reproduction failed",
            file=sys.stderr,
        )
        return 1

    if not within_result.key_matched:
        print(
            "Within-radius reproduction failed",
            file=sys.stderr,
        )
        return 1

    print()
    print("Smoke test passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())