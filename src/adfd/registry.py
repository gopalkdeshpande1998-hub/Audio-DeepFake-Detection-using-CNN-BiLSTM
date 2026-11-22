"""Which trained models the app, API and explain.py can use.

A model is a run folder in runs/ with best.pt (and ideally reliability.json from
scripts/fit_reliability.py). The default is the model chosen for the app in
experiments/tables/phase3.json (dev-data selection), else the Phase 2 selection.
Override with the ADFD_MODEL environment variable (a run name or folder).

Besides run folders there are two kinds of entries:
  vg_improved, vg_v2   the external VoiceGuard checkpoints (adfd.models.voiceguard),
                       available when the checkpoint exists and scripts/fit_reliability.py
                       has written runs/<name>/reliability.json
  ensemble             the merged detector (adfd.ensemble), available when
                       scripts/fit_ensemble.py has written runs/ensemble/ensemble.json;
                       it is the default model when present
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from adfd.paths import REPO_ROOT, load_paths

FALLBACK = "e4a_rawboost_s42"
LABELS = {
    "e1_cnn_bilstm_mfcc": "CNN-BiLSTM, MFCC (v1 paper model)",
    "e2a_lfcc": "CNN-BiLSTM, LFCC (untrimmed)",
    "e4a_rawboost": "CNN-BiLSTM, LFCC + RawBoost (Phase 2 pick)",
    "e4c_rawboost_codec": "CNN-BiLSTM, LFCC + RawBoost + codec",
    "e5a_lcnn": "LCNN, LFCC",
    "e5b_rawnet2": "RawNet2 (raw waveform)",
    "e5c_aasist": "AASIST (raw waveform)",
    "e6b_crnn_seres_asp": "SE-ResNet CNN-BiLSTM-Att, LFCC",
    "e7a_wavlm_crnn": "WavLM-Base+ + CNN-BiLSTM-Att",
}


def runs_dir() -> Path:
    return load_paths().runs_dir


ENSEMBLE = "ensemble"
ENSEMBLE_LABEL = "Merged: WavLM E7a + VoiceGuard improved + VoiceGuard v2 (recommended)"


def ensemble_ready() -> bool:
    return (runs_dir() / ENSEMBLE / "ensemble.json").exists()


def available() -> dict[str, str]:
    """{run name: label} for runs with a checkpoint, seed 42 only (one per model)."""
    from adfd.models.voiceguard import VOICEGUARD_MODELS, available_voiceguard

    out = {ENSEMBLE: ENSEMBLE_LABEL} if ensemble_ready() else {}
    for name in available_voiceguard():
        if (runs_dir() / name / "reliability.json").exists():
            out[name] = VOICEGUARD_MODELS[name]["label"]
    for prefix, label in LABELS.items():
        run = runs_dir() / f"{prefix}_s42"
        if (run / "best.pt").exists() and (run / "train_summary.json").exists():   # finished training only
            out[run.name] = label + ("" if (run / "reliability.json").exists() else " (uncalibrated)")
    return out


def default_run() -> Path:
    env = os.environ.get("ADFD_MODEL")
    if env:
        p = Path(env)
        return p if p.is_absolute() or p.exists() else runs_dir() / env
    if ensemble_ready():
        return runs_dir() / ENSEMBLE
    f = REPO_ROOT / "experiments" / "tables" / "phase3.json"
    if f.exists():
        best = json.loads(f.read_text(encoding="utf-8")).get("_app_model")
        if best and (runs_dir() / f"{best}_s42" / "reliability.json").exists():
            return runs_dir() / f"{best}_s42"
    return runs_dir() / FALLBACK


def make_analyzer(run: str, device: str = "cpu"):
    """Analyzer (single model) or EnsembleAnalyzer for a name from available()."""
    if run == ENSEMBLE:
        from adfd.ensemble import EnsembleAnalyzer

        return EnsembleAnalyzer(device=device)
    from adfd.inference import Analyzer

    return Analyzer(runs_dir() / run, device)
