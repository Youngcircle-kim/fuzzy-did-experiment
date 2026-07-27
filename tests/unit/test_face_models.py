from pathlib import Path

import numpy as np
import pytest
from fuzzy_did.config.schema import FaceModelConfig
from fuzzy_did.face.models import FaceEmbedding, FaceRegion


def test_facenet512_config() -> None:
    config = FaceModelConfig(
        backend="deepface",
        model_name="Facenet512",
        detector_backend="retinaface",
        align=True,
        enforce_detection=True,
        normalization="Facenet2018",
        max_faces=1,
    )

    assert config.model_name == "Facenet512"
    assert config.detector_backend == "retinaface"
    assert config.align is True


def test_unsupported_model_name_raises_error() -> None:
    with pytest.raises(ValueError):
        FaceModelConfig(
            backend="deepface",
            model_name="UnsupportedModel",
            detector_backend="retinaface",
            align=True,
            enforce_detection=True,
            normalization="base",
            max_faces=1,
        )


def test_face_embedding_dimension() -> None:
    result = FaceEmbedding(
        image_path=Path("sample.jpg"),
        model_name="Facenet512",
        detector_backend="retinaface",
        embedding=np.zeros(512, dtype=np.float32),
        face_region=FaceRegion(
            x=10,
            y=20,
            width=100,
            height=100,
            confidence=0.99,
        ),
    )

    assert result.dimension == 512
    assert result.embedding.dtype == np.float32