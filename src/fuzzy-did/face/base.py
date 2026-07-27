from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from fuzzy_did.face.models import FaceEmbedding


class FaceFeatureExtractionError(RuntimeError):
    """Raised when a face embedding cannot be extracted."""


class FaceFeatureExtractor(ABC):
    """Interface implemented by face embedding backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the face recognition model name."""

    @abstractmethod
    def extract_one(self, image_path: str | Path) -> FaceEmbedding:
        """Extract one embedding from an image."""

    def extract_many(
        self,
        image_paths: Sequence[str | Path],
    ) -> list[FaceEmbedding]:
        """
        Extract embeddings from multiple image paths.

        The first implementation executes sequentially. Dataset-level
        batching, error recovery, and cache storage will be added later.
        """

        return [self.extract_one(path) for path in image_paths]