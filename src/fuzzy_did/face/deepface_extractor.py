from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]


class FaceExtractionError(RuntimeError):
    """Raised when DeepFace cannot produce a valid embedding."""


@dataclass(frozen=True)
class FaceExtractionResult:
    """Embedding and detection metadata for one image."""

    embedding: FloatArray
    face_confidence: float | None
    facial_area: dict[str, Any] | None
    detected_face_count: int


def configure_tensorflow_gpu(
    enable_memory_growth: bool,
) -> None:
    """
    Configure TensorFlow GPU behavior before DeepFace model initialization.

    CUDA_VISIBLE_DEVICES must be set before starting the Python process.
    """

    if not enable_memory_growth:
        return

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise FaceExtractionError(
            "TensorFlow is not installed."
        ) from exc

    physical_gpus = tf.config.list_physical_devices("GPU")

    if not physical_gpus:
        raise FaceExtractionError(
            "TensorFlow cannot detect a GPU."
        )

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError as exc:
            raise FaceExtractionError(
                "TensorFlow GPU was initialized before memory-growth "
                "configuration."
            ) from exc


class DeepFaceExtractor:
    """DeepFace-based facial feature extractor."""

    def __init__(
        self,
        model_name: str,
        detector_backend: str,
        normalization: str,
        align: bool = True,
        enforce_detection: bool = True,
        max_faces: int = 1,
        expected_dimension: int = 512,
        memory_growth: bool = True,
    ) -> None:
        os.environ.setdefault(
            "TF_USE_LEGACY_KERAS",
            "1",
        )
        os.environ.setdefault(
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "true",
        )
        os.environ.setdefault(
            "TF_CPP_MIN_LOG_LEVEL",
            "1",
        )

        configure_tensorflow_gpu(
            enable_memory_growth=memory_growth,
        )

        # Import after TensorFlow GPU configuration.
        from deepface import DeepFace

        self._deepface = DeepFace

        self.model_name = model_name
        self.detector_backend = detector_backend
        self.normalization = normalization
        self.align = align
        self.enforce_detection = enforce_detection
        self.max_faces = max_faces
        self.expected_dimension = expected_dimension

        self._supported_represent_parameters = set(
            inspect.signature(
                self._deepface.represent
            ).parameters
        )

        # Load model once per process.
        self._load_model()

    def _load_model(self) -> None:
        try:
            build_signature = inspect.signature(
                self._deepface.build_model
            ).parameters

            if "task" in build_signature:
                self._deepface.build_model(
                    task="facial_recognition",
                    model_name=self.model_name,
                )
            else:
                self._deepface.build_model(
                    model_name=self.model_name,
                )

        except Exception as exc:
            raise FaceExtractionError(
                f"Failed to load DeepFace model: {self.model_name}"
            ) from exc

    def _build_represent_kwargs(
        self,
        image_path: Path,
    ) -> dict[str, Any]:
        requested_kwargs: dict[str, Any] = {
            "img_path": str(image_path),
            "model_name": self.model_name,
            "detector_backend": self.detector_backend,
            "enforce_detection": self.enforce_detection,
            "align": self.align,
            "normalization": self.normalization,
            "max_faces": self.max_faces,
            "silent": True,
        }

        return {
            key: value
            for key, value in requested_kwargs.items()
            if key in self._supported_represent_parameters
        }

    def extract(
        self,
        image_path: str | Path,
    ) -> FaceExtractionResult:
        resolved_path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        if not resolved_path.is_file():
            raise FaceExtractionError(
                f"Image file does not exist: {resolved_path}"
            )

        represent_kwargs = self._build_represent_kwargs(
            resolved_path
        )

        try:
            representations = self._deepface.represent(
                **represent_kwargs
            )
        except Exception as exc:
            raise FaceExtractionError(
                f"DeepFace extraction failed for "
                f"{resolved_path}: {exc}"
            ) from exc

        if not representations:
            raise FaceExtractionError(
                f"DeepFace returned no representation: {resolved_path}"
            )

        selected = max(
            representations,
            key=lambda item: float(
                item.get(
                    "face_confidence",
                    0.0,
                )
                or 0.0
            ),
        )

        raw_embedding = selected.get("embedding")

        if raw_embedding is None:
            raise FaceExtractionError(
                f"DeepFace result has no embedding: {resolved_path}"
            )

        embedding = np.asarray(
            raw_embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            raise FaceExtractionError(
                f"Embedding must be one-dimensional, "
                f"got {embedding.shape}: {resolved_path}"
            )

        if embedding.shape[0] != self.expected_dimension:
            raise FaceExtractionError(
                f"Unexpected embedding dimension: "
                f"expected={self.expected_dimension}, "
                f"actual={embedding.shape[0]}"
            )

        if not np.all(np.isfinite(embedding)):
            raise FaceExtractionError(
                f"Embedding contains NaN or infinity: {resolved_path}"
            )

        if float(np.linalg.norm(embedding)) == 0.0:
            raise FaceExtractionError(
                f"Embedding has zero norm: {resolved_path}"
            )

        confidence = selected.get(
            "face_confidence"
        )

        if confidence is not None:
            confidence = float(confidence)

        facial_area = selected.get(
            "facial_area"
        )

        if not isinstance(facial_area, dict):
            facial_area = None

        return FaceExtractionResult(
            embedding=embedding,
            face_confidence=confidence,
            facial_area=facial_area,
            detected_face_count=len(
                representations
            ),
        )