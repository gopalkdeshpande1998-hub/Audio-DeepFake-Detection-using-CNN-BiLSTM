"""Record everything needed to reproduce a run next to its results."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from adfd.paths import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def describe_device() -> dict:
    """GPU name, VRAM and CUDA versions (or CPU-only)."""
    info: dict = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            vram_gb=round(props.total_memory / 1024**3, 2),
            compute_capability=f"{props.major}.{props.minor}",
            cuda_runtime=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        )
    return info


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("torch", "torchaudio", "transformers", "numpy", "librosa", "soundfile", "scikit-learn", "pandas"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def collect_run_info(config: dict, manifests_summary: Path | None = None) -> dict:
    """Config, code version, environment and data fingerprint for one run."""
    manifests = None
    if manifests_summary and manifests_summary.exists():
        summary = json.loads(manifests_summary.read_text(encoding="utf-8"))
        manifests = {name: s.get("sha256") for name, s in summary.items()}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": sys.argv,
        "config": config,
        "device": describe_device(),
        "packages": _package_versions(),
        "manifests_sha256": manifests,
    }


def save_run_info(run_dir: Path, config: dict, manifests_summary: Path | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "run_info.json"
    out.write_text(json.dumps(collect_run_info(config, manifests_summary), indent=2), encoding="utf-8")
    return out
