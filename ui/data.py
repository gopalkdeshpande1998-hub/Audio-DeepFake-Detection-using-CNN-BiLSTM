"""Cached readers for the result files in experiments/ (no PyTorch import)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def read_json(rel: str) -> dict | None:
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def error_analysis() -> dict | None:
    return read_json("experiments/error_analysis_e7a/error_analysis.json")


# ---------------------------------------------------------------------------------------
# Results page: 100 real + 100 fake ASVspoof 2019 eval clips scored by both models
# ---------------------------------------------------------------------------------------
SAMPLE_METHODS = {"e7a_fake": "App verdict (E7a)", "t1_fake": "New model: whole clip",
                  "t1_any_fake_segment": "New model: any FAKE segment"}


def sample_test() -> list[dict] | None:
    """Per-clip results (experiments/sample_test_100/results.json): truth, speaker, system and each method's call."""
    return read_json("experiments/sample_test_100/results.json")


def confusion_metrics(truth_fake: list[bool], pred_fake: list[bool]) -> dict:
    """FAKE is the positive class. Rates per 100: real clips called FAKE, fake clips called REAL."""
    tp = sum(t and p for t, p in zip(truth_fake, pred_fake))
    fp = sum((not t) and p for t, p in zip(truth_fake, pred_fake))
    fn = sum(t and (not p) for t, p in zip(truth_fake, pred_fake))
    tn = sum((not t) and (not p) for t, p in zip(truth_fake, pred_fake))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    n_real, n_fake = fp + tn, tp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_real": n_real, "n_fake": n_fake,
            "accuracy": 100 * (tp + tn) / max(tp + fp + fn + tn, 1), "precision": 100 * prec, "recall": 100 * rec,
            "f1": 100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0),
            "real_called_fake_per100": 100 * fp / max(n_real, 1), "fake_called_real_per100": 100 * fn / max(n_fake, 1)}


def temporal_test() -> dict | None:
    return read_json("experiments/temporal_t1/test.json")
