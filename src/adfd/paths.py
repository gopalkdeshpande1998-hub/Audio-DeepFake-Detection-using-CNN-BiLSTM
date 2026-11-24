"""Project paths: dataset roots, generated-data folders and cache locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"

DATASET_NAMES = ("asvspoof2019_la", "in_the_wild", "deep_voice")


@dataclass(frozen=True)
class Paths:
    datasets: dict[str, Path]
    manifests_dir: Path
    cache_dir: Path
    runs_dir: Path

    def dataset(self, name: str) -> Path:
        if name not in self.datasets:
            raise KeyError(f"Unknown dataset {name!r}; expected one of {sorted(self.datasets)}")
        return self.datasets[name]

    def manifest(self, name: str) -> Path:
        return self.manifests_dir / f"{name}.csv"


def _resolve(p: str | Path) -> Path:
    """Absolute paths as given; ~ and $VARS expanded (macOS / Linux homes); relative ones from the repo root."""
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_paths(path: str | Path | None = None) -> Paths:
    """Load paths from ADFD_PATHS, an explicit file, or configs/paths.yaml."""
    path = Path(path or os.environ.get("ADFD_PATHS") or DEFAULT_PATHS_FILE)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = {name: _resolve(p) for name, p in cfg["datasets"].items()}
    missing = set(DATASET_NAMES) - set(datasets)
    if missing:
        raise ValueError(f"{path} is missing dataset entries: {sorted(missing)}")
    return Paths(
        datasets=datasets,
        manifests_dir=_resolve(cfg.get("manifests_dir", "data/manifests")),
        cache_dir=_resolve(cfg.get("cache_dir", ".cache")),
        runs_dir=_resolve(cfg.get("runs_dir", "runs")),
    )


def configure_caches(paths: Paths | None = None) -> Path:
    """Send Hugging Face / torch / pip caches to the project cache folder.

    Call this before importing transformers or downloading models. Variables
    the user has already set are left alone.
    """
    cache = (paths or load_paths()).cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    # macOS (Apple silicon): run operators the MPS backend lacks on the CPU instead of failing
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return cache
