import argparse
from pathlib import Path

import numpy as np
from fuzzy_did.config import load_config
from fuzzy_did.face import DeepFaceFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one face embedding with DeepFace."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to the YAML experiment configuration.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a face image.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    extractor = DeepFaceFeatureExtractor(config.face_model)
    result = extractor.extract_one(args.image)

    embedding_norm = float(np.linalg.norm(result.embedding))

    print(f"image: {result.image_path}")
    print(f"model: {result.model_name}")
    print(f"detector: {result.detector_backend}")
    print(f"embedding dimension: {result.dimension}")
    print(f"embedding dtype: {result.embedding.dtype}")
    print(f"embedding L2 norm: {embedding_norm:.6f}")

    if result.face_region is not None:
        print(
            "face region: "
            f"x={result.face_region.x}, "
            f"y={result.face_region.y}, "
            f"w={result.face_region.width}, "
            f"h={result.face_region.height}, "
            f"confidence={result.face_region.confidence}"
        )


if __name__ == "__main__":
    main()