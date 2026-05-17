from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must contain a YAML mapping.")
    return config

def ensure_dir(path: str | Path) -> Path:
    """
    ensure the directory is created
    """
    dir = Path(path)
    dir.mkdir(parents=True, exist_ok=True)
    return dir