from pathlib import Path
from typing import Any

import numpy as np
from deepface import DeepFace
from fuzzy_did.config.schema import FaceModelConfig
from fuzzy_did.face.base import (
    FaceFeatureExtractionError,
    FaceFeatureExtractor,
)
from fuzzy_did.face.models import FaceEmbedding, FaceRegion


class DeepFaceFeatureExtractor(FaceFeatureExtractor):
    """Face embedding extractor implemented with DeepFace."""

    def __init__(self, config: FaceModelConfig) -> None:
        self._config = config

        # Load model weights once when the extractor is constructed.
        # This prevents rebuilding the model for every image.
        try:
            self._model = DeepFace.build_model(
                task="facial_recognition",
                model_name=config.model_name,
            )
        except Exception as exc:
            raise FaceFeatureExtractionError(
                f"failed to load DeepFace model: {config.model_name}"
            ) from exc

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def extract_one(self, image_path: str | Path) -> FaceEmbedding:
        resolved_path = Path(image_path).expanduser().resolve()

        if not resolved_path.exists():
            raise FaceFeatureExtractionError(
                f"image file does not exist: {resolved_path}"
            )

        if not resolved_path.is_file():
            raise FaceFeatureExtractionError(
                f"image path is not a file: {resolved_path}"
            )

        try:
            representations = DeepFace.represent(
                img_path=str(resolved_path),
                model_name=self._config.model_name,
                detector_backend=self._config.detector_backend,
                enforce_detection=self._config.enforce_detection,
                align=self._config.align,
                normalization=self._config.normalization,
                max_faces=self._config.max_faces,
                silent=True,
            )
        except Exception as exc:
            raise FaceFeatureExtractionError(
                f"DeepFace failed to process image: {resolved_path}"
            ) from exc

        if not representations:
            raise FaceFeatureExtractionError(
                f"no face representation returned: {resolved_path}"
            )

        if len(representations) > 1:
            raise FaceFeatureExtractionError(
                "multiple faces were returned even though the experiment "
                f"expects one face: {resolved_path}"
            )

        representation = representations[0]

        embedding = self._parse_embedding(
            representation=representation,
            image_path=resolved_path,
        )

        face_region = self._parse_face_region(representation)

        return FaceEmbedding(
            image_path=resolved_path,
            model_name=self._config.model_name,
            detector_backend=self._config.detector_backend,
            embedding=embedding,
            face_region=face_region,
        )

    @staticmethod
    def _parse_embedding(
        representation: dict[str, Any],
        image_path: Path,
    ) -> np.ndarray:
        raw_embedding = representation.get("embedding")

        if raw_embedding is None:
            raise FaceFeatureExtractionError(
                f"DeepFace response has no embedding: {image_path}"
            )

        embedding = np.asarray(raw_embedding, dtype=np.float32)

        if embedding.ndim != 1:
            raise FaceFeatureExtractionError(
                f"embedding must be one-dimensional, got shape "
                f"{embedding.shape}: {image_path}"
            )

        if embedding.size == 0:
            raise FaceFeatureExtractionError(
                f"embedding is empty: {image_path}"
            )

        if not np.all(np.isfinite(embedding)):
            raise FaceFeatureExtractionError(
                f"embedding contains NaN or infinity: {image_path}"
            )

        return embedding

    @staticmethod
    def _parse_face_region(
        representation: dict[str, Any],
    ) -> FaceRegion | None:
        facial_area = representation.get("facial_area")

        if not isinstance(facial_area, dict):
            return None

        required_fields = ("x", "y", "w", "h")

        if not all(field in facial_area for field in required_fields):
            return None

        confidence = representation.get("face_confidence")

        if confidence is not None:
            confidence = float(confidence)

        return FaceRegion(
            x=int(facial_area["x"]),
            y=int(facial_area["y"]),
            width=int(facial_area["w"]),
            height=int(facial_area["h"]),
            confidence=confidence,
        )