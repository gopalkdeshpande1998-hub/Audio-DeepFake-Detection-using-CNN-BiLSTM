"""Trust lookup and app smoke tests (CPU, no model weights needed for the research pages)."""

from __future__ import annotations

import json

import pytest

from adfd import trust


def test_band_edges():
    assert trust.band(0.5) == 0 and trust.band(0.55) == 0
    assert trust.band(0.65) == 1 and trust.band(0.25) == 2
    assert trust.band(0.85) == 3 and trust.band(0.02) == 4 and trust.band(1.0) == 4


def test_assess_without_table(monkeypatch, tmp_path):
    trust.load.cache_clear()
    monkeypatch.setattr(trust, "PATH", tmp_path / "missing.json")
    monkeypatch.setattr(trust, "load", lambda path=None: None)
    t = trust.assess(0.97, [0.9, 0.99, 0.8])
    assert t["level"] == "very high" and t["agree"] == 3 and t["sets"] == {}
    assert "agree" in t["advice"]


def test_assess_uses_grid_then_band(monkeypatch):
    table = {"confidence_bins": ["b0", "b1", "b2", "b3", "b4"],
             "sets": {"S": {"reliability": {"grid": {"3|4": {"n": 50, "accuracy_pct": 99.0}},
                                            "by_confidence": {"b1": {"n": 70, "accuracy_pct": 80.0}}}}}}
    monkeypatch.setattr(trust, "load", lambda path=None: table)
    assert trust.assess(0.99, [0.9, 0.9, 0.9])["sets"]["S"] == {"n": 50, "accuracy_pct": 99.0,
                                                                "basis": "same confidence and agreement"}
    t = trust.assess(0.35, [0.2, 0.7, 0.4])        # REAL, 2 of 3 agree, band 1 -> falls back to the band
    assert t["agree"] == 2 and t["sets"]["S"]["basis"] == "same confidence"
    assert "weak" in t["advice"]


def test_error_analysis_file_shape():
    path = trust.PATH
    if not path.exists():
        pytest.skip("experiments/error_analysis/error_analysis.json not built")
    d = json.loads(path.read_text(encoding="utf-8"))
    for s in d["sets"].values():
        assert {"overall", "reliability", "false_accepts", "false_rejects", "by_duration"} <= set(s)


@pytest.mark.parametrize("page", ["overview", "results", "errors", "card", "help", "history"])
def test_research_pages_render(page):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    fn = {"card": "model_card", "help": "help_page"}.get(page, page)
    script = f"""
from ui import nav, pages, theme
theme.inject()
nav.build()
pages.{fn}()
"""
    at = st_testing.AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception, at.exception
