from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fuzzy_did.binarization import (
    MedianBinarizerConfig,
    binarize_values,
    pack_binary_templates,
    validate_binary_templates,
)
from fuzzy_did.data import IdentityEmbeddingCache
from fuzzy_did.normalization import (
    RobustScalerState,
    gather_selected_dimensions,
    transform_embeddings,
)


FloatArray = npt.NDArray[np.float32]
BitArray = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int16]
StringArray = npt.NDArray[np.str_]


class ProbeTemplateError(RuntimeError):
    """Raised when probe binary templates cannot be generated."""


@dataclass(frozen=True)
class ProbeBinaryTemplateSet:
    identity_id: str
    experiment_group: str
    enrollment_count: int

    image_ids: StringArray
    relative_paths: StringArray

    selected_dimensions: IntArray
    binary_templates: BitArray
    packed_binary_templates: BitArray
    one_counts: IntArray

    @property
    def probe_count(self) -> int:
        return int(self.binary_templates.shape[0])

    @property
    def template_length(self) -> int:
        return int(self.binary_templates.shape[1])

    @property
    def packed_length_bytes(self) -> int:
        return int(self.packed_binary_templates.shape[1])


def build_probe_mask(
    cache: IdentityEmbeddingCache,
) -> npt.NDArray[np.bool_]:
    """
    Select only true probe images.

    All enrollment candidates, including unused ranks, are excluded.
    The cache stores -1 for images that are not enrollment candidates.
    """

    ranks = np.asarray(
        cache.enrollment_candidate_ranks,
        dtype=np.int16,
    )

    roles = cache.sample_roles.astype(str)

    mask = (
        (ranks < 0)
        & (roles == "probe")
    )

    if not mask.any():
        raise ProbeTemplateError(
            f"{cache.identity_id}: no valid probe images remain"
        )

    return mask


def build_probe_binary_template_set(
    *,
    cache: IdentityEmbeddingCache,
    enrollment_count: int,
    selected_dimensions: IntArray,
    scaler_state: RobustScalerState,
    binarizer_config: MedianBinarizerConfig,
) -> ProbeBinaryTemplateSet:
    if enrollment_count <= 0:
        raise ProbeTemplateError(
            "enrollment_count must be positive"
        )

    dimensions = np.asarray(
        selected_dimensions,
        dtype=np.int16,
    )

    if dimensions.ndim != 1:
        raise ProbeTemplateError(
            "selected_dimensions must be one-dimensional"
        )

    if len(dimensions) == 0:
        raise ProbeTemplateError(
            "selected_dimensions cannot be empty"
        )

    if len(np.unique(dimensions)) != len(dimensions):
        raise ProbeTemplateError(
            f"{cache.identity_id}: selected dimensions contain duplicates"
        )

    if (
        (dimensions < 0).any()
        or (dimensions >= cache.embedding_dimension).any()
    ):
        raise ProbeTemplateError(
            f"{cache.identity_id}: selected dimension is out of range"
        )

    if scaler_state.dimension != cache.embedding_dimension:
        raise ProbeTemplateError(
            f"{cache.identity_id}: scaler dimension mismatch. "
            f"Expected={cache.embedding_dimension}, "
            f"actual={scaler_state.dimension}"
        )

    probe_mask = build_probe_mask(cache)

    probe_embeddings = cache.embeddings[
        probe_mask
    ].astype(
        np.float32,
        copy=False,
    )

    probe_image_ids = cache.image_ids[
        probe_mask
    ].astype(np.str_)

    probe_relative_paths = cache.relative_paths[
        probe_mask
    ].astype(np.str_)

    normalized_probe_embeddings = transform_embeddings(
        probe_embeddings,
        scaler_state=scaler_state,
    )

    # gather_selected_dimensions expects [N, D] values and [N, K] indices.
    repeated_dimensions = np.broadcast_to(
        dimensions[None, :],
        (
            len(probe_embeddings),
            len(dimensions),
        ),
    ).astype(
        np.int16,
        copy=True,
    )

    normalized_selected_values = gather_selected_dimensions(
        normalized_probe_embeddings,
        repeated_dimensions,
    )

    binary_templates = binarize_values(
        normalized_selected_values,
        config=binarizer_config,
    )

    packed_templates = pack_binary_templates(
        binary_templates,
        bitorder=binarizer_config.bitorder,
    )

    validate_binary_templates(
        binary_templates,
        packed_templates,
        bitorder=binarizer_config.bitorder,
    )

    one_counts = binary_templates.sum(
        axis=1,
        dtype=np.int64,
    ).astype(np.int16)

    return ProbeBinaryTemplateSet(
        identity_id=cache.identity_id,
        experiment_group=cache.experiment_group,
        enrollment_count=enrollment_count,
        image_ids=probe_image_ids,
        relative_paths=probe_relative_paths,
        selected_dimensions=dimensions,
        binary_templates=binary_templates,
        packed_binary_templates=packed_templates,
        one_counts=one_counts,
    )