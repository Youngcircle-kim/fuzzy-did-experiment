from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown configuration fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class ProjectConfig(StrictConfigModel):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    output_dir: Path


class DataConfig(StrictConfigModel):
    dataset_name: str = Field(min_length=1)
    raw_dir: Path
    metadata_dir: Path
    split_dir: Path
    embedding_cache_dir: Path


class RuntimeConfig(StrictConfigModel):
    name: str = Field(min_length=1)
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_workers: int = Field(default=0, ge=0)
    batch_size: int = Field(default=32, ge=1)


class FaceModelConfig(StrictConfigModel):
    """DeepFace feature extraction configuration."""

    backend: Literal["deepface"] = "deepface"

    model_name: Literal[
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "ArcFace",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "Dlib",
        "SFace",
        "GhostFaceNet",
        "Buffalo_L",
    ] = "Facenet512"

    detector_backend: Literal[
        "opencv",
        "retinaface",
        "mtcnn",
        "ssd",
        "dlib",
        "mediapipe",
        "yolov8",
        "yolov11n",
        "yolov11s",
        "yolov11m",
        "centerface",
        "skip",
    ] = "retinaface"

    align: bool = True
    enforce_detection: bool = True
    normalization: str = Field(default="base", min_length=1)
    max_faces: int = Field(default=1, ge=1)


class TemplateConfig(StrictConfigModel):
    length: int = Field(ge=1)
    enrollment_count: int = Field(ge=1)

    @field_validator("length")
    @classmethod
    def validate_template_length(cls, value: int) -> int:
        supported_lengths = {127, 128, 255}

        if value not in supported_lengths:
            raise ValueError(
                f"template length must be one of {sorted(supported_lengths)}, "
                f"but received {value}"
            )

        return value


class LoggingConfig(StrictConfigModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    save_file: bool = True


class ExperimentConfig(StrictConfigModel):
    project: ProjectConfig
    data: DataConfig
    experiment: RuntimeConfig
    face_model: FaceModelConfig
    template: TemplateConfig
    logging: LoggingConfig