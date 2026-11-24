"""Cached readers for the result files in experiments/ and docs/ (no PyTorch import)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"


@st.cache_data(show_spinner=False)
def read_json(rel: str) -> dict | None:
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def read_text(rel: str) -> str | None:
    try:
        return (ROOT / rel).read_text(encoding="utf-8")
    except OSError:
        return None


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def path(rel: str) -> Path:
    return ROOT / rel


def strip_title(md: str) -> str:
    """Drop the first '# ' heading so the page's own heading is the only h1."""
    return re.sub(r"\A\s*# [^\n]*\n", "", md, count=1)


def error_analysis() -> dict | None:
    return read_json("experiments/error_analysis_e7a/error_analysis.json")


def phase3() -> dict | None:
    return read_json("experiments/tables/phase3.json")


def ensemble_binary() -> dict | None:
    return read_json("experiments/ensemble_binary/ensemble.json")
