from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# TensorFlow와 DeepFace를 import하기 전에 설정해야 한다.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


class SmokeTestError(RuntimeError):
    """Raised when the Facenet512 smoke test fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-image Facenet512 GPU smoke test "
            "using one image from an embedding shard."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/embedding_facenet512.yaml"
        ),
        help="Path to the embedding experiment YAML file.",
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index from which one test image is selected.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Optional image path override. "
            "If omitted, the first valid image in the shard is used."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/smoke_tests/facenet512_result.json"
        ),
        help="Path to the smoke-test result JSON.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise SmokeTestError(
            f"configuration file does not exist: {resolved}"
        )

    try:
        with resolved.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise SmokeTestError(
            f"failed to read configuration file: {resolved}"
        ) from exc
    except yaml.YAMLError as exc:
        raise SmokeTestError(
            f"invalid YAML syntax: {resolved}"
        ) from exc

    if not isinstance(loaded, dict):
        raise SmokeTestError(
            f"configuration root must be a mapping: {resolved}"
        )

    return loaded


def configure_gpu() -> list[tf.config.PhysicalDevice]:
    """
    Verify TensorFlow GPU visibility and enable memory growth.

    CUDA_VISIBLE_DEVICES should be set before this process starts.
    For example:
        CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_facenet512.py
    """

    physical_gpus = tf.config.list_physical_devices("GPU")

    if not physical_gpus:
        raise SmokeTestError(
            "TensorFlow cannot detect a GPU. "
            "Check CUDA_VISIBLE_DEVICES and the TensorFlow CUDA environment."
        )

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError as exc:
            raise SmokeTestError(
                "failed to configure GPU memory growth. "
                "TensorFlow may have initialized the GPU too early."
            ) from exc

    return physical_gpus


def select_image_from_shard(
    shard_dir: Path,
    shard_index: int,
) -> tuple[Path, dict[str, Any]]:
    shard_path = (
        shard_dir
        / f"shard_{shard_index:03d}.parquet"
    )

    if not shard_path.is_file():
        raise SmokeTestError(
            f"shard file does not exist: {shard_path}"
        )

    dataframe = pd.read_parquet(
        shard_path,
        columns=[
            "identity_id",
            "image_id",
            "image_path",
            "relative_path",
            "experiment_group",
            "sample_role",
        ],
    )

    if dataframe.empty:
        raise SmokeTestError(
            f"shard dataframe is empty: {shard_path}"
        )

    required_columns = {
        "identity_id",
        "image_id",
        "image_path",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise SmokeTestError(
            f"shard is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # 우선 enrollment 이미지를 선택한다.
    enrollment_rows = dataframe[
        dataframe["sample_role"] == "enrollment"
    ]

    if not enrollment_rows.empty:
        selected_row = enrollment_rows.sort_values(
            by=[
                "identity_id",
                "image_id",
            ],
            kind="stable",
        ).iloc[0]
    else:
        selected_row = dataframe.sort_values(
            by=[
                "identity_id",
                "image_id",
            ],
            kind="stable",
        ).iloc[0]

    image_path = Path(
        str(selected_row["image_path"])
    ).expanduser().resolve()

    if not image_path.is_file():
        raise SmokeTestError(
            f"selected image does not exist: {image_path}"
        )

    metadata = {
        "identity_id": str(
            selected_row["identity_id"]
        ),
        "image_id": str(
            selected_row["image_id"]
        ),
        "relative_path": str(
            selected_row.get(
                "relative_path",
                "",
            )
        ),
        "experiment_group": str(
            selected_row.get(
                "experiment_group",
                "",
            )
        ),
        "sample_role": str(
            selected_row.get(
                "sample_role",
                "",
            )
        ),
        "shard_path": str(
            shard_path.resolve()
        ),
    }

    return image_path, metadata


def validate_embedding(
    embedding: np.ndarray,
    expected_dimension: int,
) -> None:
    if embedding.ndim != 1:
        raise SmokeTestError(
            f"embedding must be one-dimensional, "
            f"got shape={embedding.shape}"
        )

    if embedding.shape[0] != expected_dimension:
        raise SmokeTestError(
            f"unexpected embedding dimension: "
            f"expected={expected_dimension}, "
            f"actual={embedding.shape[0]}"
        )

    if not np.all(np.isfinite(embedding)):
        raise SmokeTestError(
            "embedding contains NaN or infinity"
        )

    norm = float(np.linalg.norm(embedding))

    if norm == 0.0:
        raise SmokeTestError(
            "embedding L2 norm is zero"
        )


def run_deepface(
    image_path: Path,
    face_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], float]:
    # GPU memory-growth 설정 후 DeepFace를 import한다.
    from deepface import DeepFace

    model_name = str(
        face_config.get(
            "model_name",
            "Facenet512",
        )
    )

    detector_backend = str(
        face_config.get(
            "detector_backend",
            "retinaface",
        )
    )

    normalization = str(
        face_config.get(
            "normalization",
            "Facenet2018",
        )
    )

    align = bool(
        face_config.get(
            "align",
            True,
        )
    )

    enforce_detection = bool(
        face_config.get(
            "enforce_detection",
            True,
        )
    )

    max_faces = int(
        face_config.get(
            "max_faces",
            1,
        )
    )

    print(f"Loading DeepFace model: {model_name}")

    # 첫 실행에서는 weight를 다운로드할 수 있다.
    started_at = time.perf_counter()

    try:
        representations = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization,
            max_faces=max_faces,
        )
    except Exception as exc:
        raise SmokeTestError(
            f"DeepFace.represent failed for {image_path}: {exc}"
        ) from exc

    elapsed_seconds = time.perf_counter() - started_at

    if not representations:
        raise SmokeTestError(
            "DeepFace returned no representations"
        )

    # 복수 얼굴이 반환된 경우 confidence가 높은 얼굴 선택
    selected_representation = max(
        representations,
        key=lambda item: float(
            item.get(
                "face_confidence",
                0.0,
            )
            or 0.0
        ),
    )

    raw_embedding = selected_representation.get(
        "embedding"
    )

    if raw_embedding is None:
        raise SmokeTestError(
            "DeepFace result does not contain an embedding"
        )

    embedding = np.asarray(
        raw_embedding,
        dtype=np.float32,
    )

    metadata = {
        "model_name": model_name,
        "detector_backend": detector_backend,
        "normalization": normalization,
        "align": align,
        "enforce_detection": enforce_detection,
        "face_confidence": selected_representation.get(
            "face_confidence"
        ),
        "facial_area": selected_representation.get(
            "facial_area"
        ),
        "number_of_detected_faces": len(
            representations
        ),
    }

    return embedding, metadata, elapsed_seconds


def save_result(
    output_path: Path,
    result: dict[str, Any],
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> int:
    args = parse_args()

    try:
        print(
            "CUDA_VISIBLE_DEVICES:",
            os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                "<not set>",
            ),
        )

        physical_gpus = configure_gpu()

        print(
            "TensorFlow version:",
            tf.__version__,
        )
        print(
            "TensorFlow CUDA build:",
            tf.test.is_built_with_cuda(),
        )
        print(
            "Visible physical GPUs:",
            len(physical_gpus),
        )

        for index, gpu in enumerate(physical_gpus):
            details = (
                tf.config.experimental
                .get_device_details(gpu)
            )

            print(
                f"GPU {index}:",
                gpu,
                details,
            )

        config = load_yaml(args.config)

        data_config = config["data"]
        face_config = config["face_model"]

        shard_dir = Path(
            data_config["shard_dir"]
        ).expanduser().resolve()

        if args.image is not None:
            image_path = (
                args.image
                .expanduser()
                .resolve()
            )

            if not image_path.is_file():
                raise SmokeTestError(
                    f"input image does not exist: {image_path}"
                )

            image_metadata = {
                "identity_id": None,
                "image_id": None,
                "relative_path": None,
                "experiment_group": None,
                "sample_role": None,
                "shard_path": None,
            }
        else:
            image_path, image_metadata = (
                select_image_from_shard(
                    shard_dir=shard_dir,
                    shard_index=args.shard_index,
                )
            )

        print("Selected image:", image_path)
        print(
            "Selected identity:",
            image_metadata["identity_id"],
        )

        embedding, deepface_metadata, elapsed_seconds = (
            run_deepface(
                image_path=image_path,
                face_config=face_config,
            )
        )

        validate_embedding(
            embedding=embedding,
            expected_dimension=512,
        )

        logical_gpus = tf.config.list_logical_devices(
            "GPU"
        )

        result = {
            "status": "success",
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "tensorflow_version": tf.__version__,
            "tensorflow_cuda_build": bool(
                tf.test.is_built_with_cuda()
            ),
            "physical_gpu_count": len(
                physical_gpus
            ),
            "logical_gpu_count": len(
                logical_gpus
            ),
            "logical_gpus": [
                device.name
                for device in logical_gpus
            ],
            "image_path": str(image_path),
            **image_metadata,
            **deepface_metadata,
            "embedding_shape": list(
                embedding.shape
            ),
            "embedding_dtype": str(
                embedding.dtype
            ),
            "embedding_l2_norm": float(
                np.linalg.norm(embedding)
            ),
            "embedding_min": float(
                embedding.min()
            ),
            "embedding_max": float(
                embedding.max()
            ),
            "embedding_mean": float(
                embedding.mean()
            ),
            "elapsed_seconds": elapsed_seconds,
        }

        save_result(
            output_path=args.output,
            result=result,
        )

        print()
        print("Smoke test passed")
        print("Embedding shape:", embedding.shape)
        print("Embedding dtype:", embedding.dtype)
        print(
            "Embedding L2 norm:",
            f"{np.linalg.norm(embedding):.6f}",
        )
        print(
            "Face confidence:",
            deepface_metadata[
                "face_confidence"
            ],
        )
        print(
            "Elapsed seconds:",
            f"{elapsed_seconds:.3f}",
        )
        print(
            "Result JSON:",
            args.output.expanduser().resolve(),
        )

        return 0

    except KeyError as exc:
        print(
            f"Missing configuration key: {exc}",
            file=sys.stderr,
        )
        return 2

    except SmokeTestError as exc:
        print(
            f"Smoke test failed: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print(
            "Smoke test interrupted",
            file=sys.stderr,
        )
        return 130

    except Exception as exc:
        print(
            f"Unexpected smoke-test error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 99


if __name__ == "__main__":
    sys.exit(main())