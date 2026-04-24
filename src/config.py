"""Thin YAML config loader used by all entry-point scripts."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required_top = {"model", "data", "training"}
    missing = required_top - cfg.keys()
    if missing:
        raise ValueError(f"Config {path} missing required sections: {missing}")

    required_model = {"type", "num_classes"}
    missing = required_model - cfg["model"].keys()
    if missing:
        raise ValueError(f"Config {path} model section missing: {missing}")

    return cfg
