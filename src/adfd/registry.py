"""Which model the app, the API and explain.py use.

Since 2026-11-23 VoiceGuard runs ONE model: WavLM-Base+ + CNN-BiLSTM-Att (E7a), run folder
runs/e7a_wavlm_crnn_s42 (best.pt + reliability.json), through adfd.single.SingleModelAnalyzer.
The VoiceGuard improved / v2 checkpoints and the merged ensemble are no longer listed, loaded or run.

ADFD_MODEL may still point at another run folder for offline experiments (explain.py, scripts);
the app and the API only offer MODEL.
"""

from __future__ import annotations

import os
from pathlib import Path

from adfd.paths import load_paths
from adfd.single import MODEL_LABEL, MODEL_RUN

MODEL = MODEL_RUN
LABELS = {MODEL: MODEL_LABEL}


def runs_dir() -> Path:
    return load_paths().runs_dir


def model_ready() -> bool:
    return (runs_dir() / MODEL / "best.pt").exists()


def available() -> dict[str, str]:
    """{run name: label}: the single production model, when its checkpoint is on this machine."""
    if not model_ready():
        return {}
    calibrated = (runs_dir() / MODEL / "reliability.json").exists()
    return {MODEL: LABELS.get(MODEL, MODEL_LABEL) + ("" if calibrated else " (uncalibrated)")}


def default_run() -> Path:
    env = os.environ.get("ADFD_MODEL")
    if env:
        p = Path(env)
        return p if p.is_absolute() or p.exists() else runs_dir() / env
    return runs_dir() / MODEL


def make_analyzer(run: str = MODEL, device: str = "cpu"):
    """The production analyzer (E7a only). Other run names load a plain Analyzer for offline experiments."""
    if run == MODEL:
        from adfd.single import SingleModelAnalyzer

        return SingleModelAnalyzer(device=device, run_dir=runs_dir() / run, label=LABELS.get(run, MODEL_LABEL))
    from adfd.inference import Analyzer

    return Analyzer(runs_dir() / run, device)
