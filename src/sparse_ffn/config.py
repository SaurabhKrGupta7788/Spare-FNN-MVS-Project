"""Configuration loading and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and return a dictionary."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a configured path relative to the project root when not absolute."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ((root or Path.cwd()) / path).resolve()
