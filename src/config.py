from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping. PyYAML is deliberately imported lazily."""
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("PyYAML is required to load experiment configuration") from exc
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return data


def load_experiment_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    config = load_yaml(path)
    return _deep_merge(config, overrides or {})


@dataclass(frozen=True)
class ReproducibilityRecord:
    experiment_id: str
    seed: int
    model_name: str
    model_revision: str = "main"
    dataset_revision: str = "unknown"
    git_commit: str = "unversioned"

