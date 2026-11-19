"""Load experiment configs: configs/base.yaml merged with an experiment file."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from adfd.paths import REPO_ROOT

BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, base: str | Path = BASE_CONFIG) -> dict:
    with open(base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg
