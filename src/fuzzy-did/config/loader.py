from pathlib import Path
from typing import Any

import yaml
from fuzzy_did.config.schema import ExperimentConfig
from pydantic import ValidationError


class ConfigLoadError(RuntimeError):
    """Raised when an experiment configuration cannot be loaded."""


def _read_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigLoadError(f"configuration file does not exist: {config_path}")

    if not config_path.is_file():
        raise ConfigLoadError(f"configuration path is not a file: {config_path}")

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigLoadError(
            f"configuration file must use .yaml or .yml extension: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigLoadError(
            f"failed to read configuration file: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(
            f"invalid YAML syntax in configuration file: {config_path}"
        ) from exc

    if loaded is None:
        raise ConfigLoadError(f"configuration file is empty: {config_path}")

    if not isinstance(loaded, dict):
        raise ConfigLoadError(
            f"configuration root must be a mapping: {config_path}"
        )

    return loaded


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load and validate an experiment configuration.

    Args:
        config_path:
            Path to a YAML configuration file.

    Returns:
        A validated immutable ExperimentConfig instance.

    Raises:
        ConfigLoadError:
            If the file cannot be read or validation fails.
    """

    resolved_path = Path(config_path).expanduser().resolve()
    raw_config = _read_yaml(resolved_path)

    try:
        return ExperimentConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigLoadError(
            f"configuration validation failed for {resolved_path}\n{exc}"
        ) from exc