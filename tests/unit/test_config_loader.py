from pathlib import Path

import pytest
from fuzzy_did.config.loader import ConfigLoadError, load_config


def test_load_base_config() -> None:
    config = load_config("configs/base.yaml")

    assert config.project.name == "fuzzy-did-experiments"
    assert config.project.seed == 42

    assert config.data.dataset_name == "vggface2"
    assert config.data.raw_dir == Path("data/raw/vggface2")

    assert config.experiment.device == "cuda"
    assert config.experiment.batch_size == 64

    assert config.template.length == 127
    assert config.template.enrollment_count == 5

    assert config.logging.level == "INFO"
    assert config.logging.save_file is True


def test_missing_config_file_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigLoadError, match="does not exist"):
        load_config(missing_path)


def test_invalid_extension_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="must use .yaml or .yml"):
        load_config(config_path)


def test_empty_config_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="is empty"):
        load_config(config_path)


def test_unknown_field_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown_field.yaml"
    config_path.write_text(
        """
project:
  name: fuzzy-did-experiments
  seed: 42
  output_dir: outputs/runs
  unknown_option: true

data:
  dataset_name: vggface2
  raw_dir: data/raw/vggface2
  metadata_dir: data/metadata
  split_dir: data/splits
  embedding_cache_dir: data/cache/embeddings

experiment:
  name: test
  device: cpu
  num_workers: 0
  batch_size: 8

template:
  length: 127
  enrollment_count: 5

logging:
  level: INFO
  save_file: false
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="unknown_option"):
        load_config(config_path)


def test_unsupported_template_length_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_template.yaml"
    config_path.write_text(
        """
project:
  name: fuzzy-did-experiments
  seed: 42
  output_dir: outputs/runs

data:
  dataset_name: vggface2
  raw_dir: data/raw/vggface2
  metadata_dir: data/metadata
  split_dir: data/splits
  embedding_cache_dir: data/cache/embeddings

experiment:
  name: test
  device: cpu
  num_workers: 0
  batch_size: 8

template:
  length: 100
  enrollment_count: 5

logging:
  level: INFO
  save_file: false
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="template length must be one of"):
        load_config(config_path)