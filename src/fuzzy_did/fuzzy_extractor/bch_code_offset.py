from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import galois
import numpy as np
import numpy.typing as npt


BitArray = npt.NDArray[np.uint8]


class FuzzyExtractorError(RuntimeError):
    """Raised when BCH code-offset generation or reproduction fails."""


@dataclass(frozen=True)
class BCHCodeParameters:
    n: int
    k: int
    d: int
    t: int

    @property
    def parity_bits(self) -> int:
        return self.n - self.k

    @property
    def code_rate(self) -> float:
        return self.k / self.n


@dataclass(frozen=True)
class EnrollmentRecord:
    """
    Output of Gen().

    helper_data is public.
    key_digest is retained for experimental verification only.
    message should not be stored in a deployed system.
    """

    helper_data: BitArray
    key_digest: bytes

    # Experimental ground truth only.
    message: BitArray
    codeword: BitArray


@dataclass(frozen=True)
class ReproductionResult:
    decode_succeeded: bool
    key_matched: bool
    recovered_message: BitArray | None
    recovered_key_digest: bytes | None
    decoder_error_count: int | None


def validate_binary_vector(
    values: npt.ArrayLike,
    *,
    expected_length: int,
    name: str,
) -> BitArray:
    array = np.asarray(
        values,
        dtype=np.uint8,
    )

    if array.ndim != 1:
        raise FuzzyExtractorError(
            f"{name} must be one-dimensional, got {array.shape}"
        )

    if len(array) != expected_length:
        raise FuzzyExtractorError(
            f"{name} length mismatch: "
            f"expected={expected_length}, actual={len(array)}"
        )

    if not np.isin(array, [0, 1]).all():
        raise FuzzyExtractorError(
            f"{name} contains values other than 0 and 1"
        )

    return array


def hash_message(
    message: BitArray,
    *,
    digest_algorithm: str = "sha256",
) -> bytes:
    """
    Hash a binary BCH message.

    This creates a fixed-length digest but does not increase
    the entropy beyond the BCH message length k.
    """

    message = np.asarray(
        message,
        dtype=np.uint8,
    )

    packed = np.packbits(
        message,
        bitorder="big",
    ).tobytes()

    try:
        digest = hashlib.new(
            digest_algorithm
        )
    except ValueError as exc:
        raise FuzzyExtractorError(
            f"Unsupported digest algorithm: {digest_algorithm}"
        ) from exc

    # Include the exact bit length to distinguish padded encodings.
    digest.update(
        len(message).to_bytes(
            4,
            byteorder="big",
            signed=False,
        )
    )
    digest.update(packed)

    return digest.digest()


def random_bit_message(
    length: int,
) -> BitArray:
    if length <= 0:
        raise FuzzyExtractorError(
            "Message length must be positive"
        )

    random_bytes = secrets.token_bytes(
        (length + 7) // 8
    )

    bits = np.unpackbits(
        np.frombuffer(
            random_bytes,
            dtype=np.uint8,
        ),
        bitorder="big",
    )

    return bits[:length].astype(
        np.uint8,
        copy=True,
    )


class BCHCodeOffsetFuzzyExtractor:
    """
    Code-offset fuzzy extractor based on a binary BCH code.

    Gen(w):
        m <- random {0,1}^k
        c = BCH.Encode(m)
        P = w XOR c
        R = Hash(m)

    Rep(w', P):
        c' = w' XOR P
        m' = BCH.Decode(c')
        R' = Hash(m')
    """

    def __init__(
        self,
        *,
        codeword_length: int,
        correction_capability: int,
        digest_algorithm: str = "sha256",
    ) -> None:
        if codeword_length <= 0:
            raise FuzzyExtractorError(
                "codeword_length must be positive"
            )

        if correction_capability <= 0:
            raise FuzzyExtractorError(
                "correction_capability must be positive"
            )

        designed_distance = (
            2 * correction_capability + 1
        )

        try:
            self.code = galois.BCH(
                codeword_length,
                d=designed_distance,
                field=galois.GF(2),
                systematic=True,
            )
        except Exception as exc:
            raise FuzzyExtractorError(
                "Failed to construct BCH code: "
                f"n={codeword_length}, "
                f"requested_t={correction_capability}"
            ) from exc

        actual_t = (
            int(self.code.d) - 1
        ) // 2

        if actual_t < correction_capability:
            raise FuzzyExtractorError(
                f"Constructed BCH code does not meet requested t: "
                f"requested={correction_capability}, "
                f"actual={actual_t}"
            )

        self.parameters = BCHCodeParameters(
            n=int(self.code.n),
            k=int(self.code.k),
            d=int(self.code.d),
            t=actual_t,
        )

        self.digest_algorithm = digest_algorithm

    def gen(
        self,
        enrollment_template: npt.ArrayLike,
        *,
        message: npt.ArrayLike | None = None,
    ) -> EnrollmentRecord:
        template = validate_binary_vector(
            enrollment_template,
            expected_length=self.parameters.n,
            name="enrollment_template",
        )

        if message is None:
            message_bits = random_bit_message(
                self.parameters.k
            )
        else:
            message_bits = validate_binary_vector(
                message,
                expected_length=self.parameters.k,
                name="message",
            )

        field_message = self.code.field(
            message_bits
        )

        try:
            field_codeword = self.code.encode(
                field_message
            )
        except Exception as exc:
            raise FuzzyExtractorError(
                "BCH encoding failed"
            ) from exc

        codeword = np.asarray(
            field_codeword,
            dtype=np.uint8,
        )

        if codeword.shape != (
            self.parameters.n,
        ):
            raise FuzzyExtractorError(
                f"Unexpected codeword shape: {codeword.shape}"
            )

        helper_data = np.bitwise_xor(
            template,
            codeword,
        ).astype(np.uint8)

        key_digest = hash_message(
            message_bits,
            digest_algorithm=self.digest_algorithm,
        )

        return EnrollmentRecord(
            helper_data=helper_data,
            key_digest=key_digest,
            message=message_bits.copy(),
            codeword=codeword.copy(),
        )

    def rep(
        self,
        probe_template: npt.ArrayLike,
        helper_data: npt.ArrayLike,
        *,
        expected_key_digest: bytes,
    ) -> ReproductionResult:
        probe = validate_binary_vector(
            probe_template,
            expected_length=self.parameters.n,
            name="probe_template",
        )

        helper = validate_binary_vector(
            helper_data,
            expected_length=self.parameters.n,
            name="helper_data",
        )

        noisy_codeword = np.bitwise_xor(
            probe,
            helper,
        ).astype(np.uint8)

        field_codeword = self.code.field(
            noisy_codeword
        )

        try:
            decoded_message, error_count = (
                self.code.decode(
                    field_codeword,
                    errors=True,
                )
            )
        except Exception:
            return ReproductionResult(
                decode_succeeded=False,
                key_matched=False,
                recovered_message=None,
                recovered_key_digest=None,
                decoder_error_count=None,
            )

        error_count_int = int(
            np.asarray(error_count).item()
        )

        # The galois decoder may use -1 to report decoding failure.
        if error_count_int < 0:
            return ReproductionResult(
                decode_succeeded=False,
                key_matched=False,
                recovered_message=None,
                recovered_key_digest=None,
                decoder_error_count=error_count_int,
            )

        recovered_message = np.asarray(
            decoded_message,
            dtype=np.uint8,
        )

        if recovered_message.shape != (
            self.parameters.k,
        ):
            return ReproductionResult(
                decode_succeeded=False,
                key_matched=False,
                recovered_message=None,
                recovered_key_digest=None,
                decoder_error_count=error_count_int,
            )

        recovered_digest = hash_message(
            recovered_message,
            digest_algorithm=self.digest_algorithm,
        )

        key_matched = secrets.compare_digest(
            recovered_digest,
            expected_key_digest,
        )

        return ReproductionResult(
            decode_succeeded=True,
            key_matched=key_matched,
            recovered_message=recovered_message,
            recovered_key_digest=recovered_digest,
            decoder_error_count=error_count_int,
        )