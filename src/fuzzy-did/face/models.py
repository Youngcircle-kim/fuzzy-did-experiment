from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class FaceRegion:
    """Detected face bounding box."""

    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None


@dataclass(frozen=True)
class FaceEmbedding:
    """Embedding and metadata extracted from one face image."""

    image_path: Path
    model_name: str
    detector_backend: str
    embedding: FloatArray
    face_region: FaceRegion | None

    @property
    def dimension(self) -> int:
        return int(self.embedding.shape[0])