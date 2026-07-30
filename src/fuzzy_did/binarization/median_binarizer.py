from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]
BitArray = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int16]
StringArray = npt.NDArray[np.str_]


class BinarizationError(RuntimeError):
    """Raised when binary templates cannot be generated."""


@dataclass(frozen=True)
class MedianBinarizerConfig:
    """
    Configuration for background-median binarization.

    Since robust normalization subtracts the background median,
    threshold=0 corresponds to background-median thresholding
    in the original embedding space.
    """

    threshold: float = 0.0
    positive_when_greater: bool = True
    bitorder: str = "big"

    def validate(self) -> None:
        if not np.isfinite(self.threshold):
            raise BinarizationError(
                "threshold must be finite"
            )

        if self.bitorder not in {"big", "little"}:
            raise BinarizationError(
                "bitorder must be either 'big' or 'little'"
            )


@dataclass(frozen=True)
class BinaryTemplateSet:
    enrollment_count: int
    top_k: int
    threshold: float
    bitorder: str

    identity_ids: StringArray
    experiment_groups: StringArray
    selected_dimensions: IntArray

    normalized_selected_values: FloatArray
    binary_templates: BitArray
    packed_binary_templates: BitArray
    one_counts: npt.NDArray[np.int16]

    @property
    def identity_count(self) -> int:
        return int(self.binary_templates.shape[0])

    @property
    def template_length(self) -> int:
        return int(self.binary_templates.shape[1])

    @property
    def packed_length_bytes(self) -> int:
        return int(
            self.packed_binary_templates.shape[1]
        )

    @property
    def global_one_ratio(self) -> float:
        return float(
            self.binary_templates.mean()
        )


def binarize_values(
    values: FloatArray,
    *,
    config: MedianBinarizerConfig,
) -> BitArray:
    """
    Convert normalized floating-point values to binary values.

    Default policy:
        value > 0  -> 1
        value <= 0 -> 0
    """

    config.validate()

    array = np.asarray(
        values,
        dtype=np.float32,
    )

    if array.ndim not in {1, 2}:
        raise BinarizationError(
            f"values must be one- or two-dimensional, "
            f"got shape={array.shape}"
        )

    if not np.isfinite(array).all():
        raise BinarizationError(
            "values contain NaN or infinity"
        )

    if config.positive_when_greater:
        binary = array > config.threshold
    else:
        binary = array >= config.threshold

    return binary.astype(
        np.uint8,
        copy=False,
    )


def pack_binary_templates(
    binary_templates: BitArray,
    *,
    bitorder: str,
) -> BitArray:
    """
    Pack binary templates into bytes.

    Example:
        [N, 128] bits -> [N, 16] bytes
    """

    templates = np.asarray(
        binary_templates,
        dtype=np.uint8,
    )

    if templates.ndim != 2:
        raise BinarizationError(
            "binary_templates must have shape [N, L]"
        )

    if not np.isin(
        templates,
        [0, 1],
    ).all():
        raise BinarizationError(
            "binary_templates contain values other than 0 and 1"
        )

    if bitorder not in {"big", "little"}:
        raise BinarizationError(
            "bitorder must be either 'big' or 'little'"
        )

    packed = np.packbits(
        templates,
        axis=1,
        bitorder=bitorder,
    )

    return packed.astype(
        np.uint8,
        copy=False,
    )


def unpack_binary_templates(
    packed_templates: BitArray,
    *,
    template_length: int,
    bitorder: str,
) -> BitArray:
    """
    Restore packed byte templates to binary bit arrays.
    """

    packed = np.asarray(
        packed_templates,
        dtype=np.uint8,
    )

    if packed.ndim != 2:
        raise BinarizationError(
            "packed_templates must have shape [N, B]"
        )

    if template_length <= 0:
        raise BinarizationError(
            "template_length must be positive"
        )

    unpacked = np.unpackbits(
        packed,
        axis=1,
        count=template_length,
        bitorder=bitorder,
    )

    return unpacked.astype(
        np.uint8,
        copy=False,
    )


def validate_binary_templates(
    binary_templates: BitArray,
    packed_templates: BitArray,
    *,
    bitorder: str,
) -> None:
    templates = np.asarray(
        binary_templates,
        dtype=np.uint8,
    )

    packed = np.asarray(
        packed_templates,
        dtype=np.uint8,
    )

    if templates.ndim != 2:
        raise BinarizationError(
            "binary_templates must have shape [N, L]"
        )

    if packed.ndim != 2:
        raise BinarizationError(
            "packed_templates must have shape [N, B]"
        )

    if templates.shape[0] != packed.shape[0]:
        raise BinarizationError(
            "identity count differs between binary and packed templates"
        )

    if not np.isin(
        templates,
        [0, 1],
    ).all():
        raise BinarizationError(
            "binary template contains a non-binary value"
        )

    expected_byte_count = (
        templates.shape[1] + 7
    ) // 8

    if packed.shape[1] != expected_byte_count:
        raise BinarizationError(
            f"packed byte length mismatch: "
            f"expected={expected_byte_count}, "
            f"actual={packed.shape[1]}"
        )

    restored = unpack_binary_templates(
        packed,
        template_length=templates.shape[1],
        bitorder=bitorder,
    )

    if not np.array_equal(
        restored,
        templates,
    ):
        raise BinarizationError(
            "packed/unpacked template validation failed"
        )


def build_binary_template_set(
    *,
    identity_ids: StringArray,
    experiment_groups: StringArray,
    selected_dimensions: IntArray,
    normalized_selected_values: FloatArray,
    enrollment_count: int,
    top_k: int,
    config: MedianBinarizerConfig,
) -> BinaryTemplateSet:
    config.validate()

    identity_ids = np.asarray(
        identity_ids,
        dtype=np.str_,
    )

    experiment_groups = np.asarray(
        experiment_groups,
        dtype=np.str_,
    )

    selected_dimensions = np.asarray(
        selected_dimensions,
        dtype=np.int16,
    )

    normalized_selected_values = np.asarray(
        normalized_selected_values,
        dtype=np.float32,
    )

    if enrollment_count <= 0:
        raise BinarizationError(
            "enrollment_count must be positive"
        )

    if top_k <= 0:
        raise BinarizationError(
            "top_k must be positive"
        )

    if normalized_selected_values.ndim != 2:
        raise BinarizationError(
            "normalized_selected_values must have shape [N, K]"
        )

    identity_count = normalized_selected_values.shape[0]

    expected_shape = (
        identity_count,
        top_k,
    )

    if selected_dimensions.shape != expected_shape:
        raise BinarizationError(
            f"selected_dimensions must have shape "
            f"{expected_shape}, got {selected_dimensions.shape}"
        )

    if normalized_selected_values.shape != expected_shape:
        raise BinarizationError(
            f"normalized_selected_values must have shape "
            f"{expected_shape}, "
            f"got {normalized_selected_values.shape}"
        )

    if len(identity_ids) != identity_count:
        raise BinarizationError(
            "identity_ids length mismatch"
        )

    if len(experiment_groups) != identity_count:
        raise BinarizationError(
            "experiment_groups length mismatch"
        )

    if len(np.unique(identity_ids)) != identity_count:
        raise BinarizationError(
            "duplicate identity IDs detected"
        )

    if not np.isfinite(
        normalized_selected_values
    ).all():
        raise BinarizationError(
            "normalized values contain NaN or infinity"
        )

    binary_templates = binarize_values(
        normalized_selected_values,
        config=config,
    )

    packed_templates = pack_binary_templates(
        binary_templates,
        bitorder=config.bitorder,
    )

    validate_binary_templates(
        binary_templates,
        packed_templates,
        bitorder=config.bitorder,
    )

    one_counts = binary_templates.sum(
        axis=1,
        dtype=np.int64,
    ).astype(np.int16)

    return BinaryTemplateSet(
        enrollment_count=enrollment_count,
        top_k=top_k,
        threshold=config.threshold,
        bitorder=config.bitorder,
        identity_ids=identity_ids,
        experiment_groups=experiment_groups,
        selected_dimensions=selected_dimensions,
        normalized_selected_values=(
            normalized_selected_values
        ),
        binary_templates=binary_templates,
        packed_binary_templates=(
            packed_templates
        ),
        one_counts=one_counts,
    )