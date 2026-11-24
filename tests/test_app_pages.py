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


@pytest.mark.parametrize("page", ["results", "errors", "history"])
def test_research_pages_render(page):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    script = f"""
from ui import nav, pages, theme
theme.inject()
nav.build()
pages.{page}()
"""
    at = st_testing.AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("p", [0.013, 0.993])
def test_single_model_result_view_renders(p):
    """The single-model verdict panel, model card, summary, feature cards and report (stub E7a scores)."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    script = f"""
import streamlit as st
from tests.test_single import single, speech
from ui import theme, detector as det
theme.inject()
res = single({p!r}).analyze(speech(8.0), explain=True)
st.markdown(det.verdict_html(res, det.model_label(res), 1.0, det.trust_of(res)), unsafe_allow_html=True)
st.markdown(det.model_card_html(res, "Loaded · test"), unsafe_allow_html=True)
st.markdown(det.summary_html(res), unsafe_allow_html=True)
for c in det.feature_cards(res):
    st.markdown(c, unsafe_allow_html=True)
st.altair_chart(det.timeline_chart(res))
html = det.analysis_report("clip.wav", res, b"")
st.session_state["report"] = html
det.remember("clip.wav", det.registry.MODEL, res, "k")
"""
    at = st_testing.AppTest.from_string(script, default_timeout=120).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    word = "SYNTHETIC" if p >= 0.5 else "REAL"
    assert word in text and "WavLM + CNN-BiLSTM-Att (E7a)" in text
    assert f"{100 * max(p, 1 - p):.1f}%" in text                                 # confidence = its own probability
    for gone in ("Majority vote", "Weighted ensemble", "Model disagreement", "models agree", "Three-model",
                 "VoiceGuard improved", "VoiceGuard v2", "Own call at 50%", "ensemble"):
        assert gone not in text, gone
    rep = at.session_state["report"]
    assert "sole detection model" in rep and "The three models" not in rep
    h = at.session_state["history"][0]
    assert h["verdict"] == word and "models_agreeing" not in h


def test_compare_shows_only_loudness_frequency_and_amplitude_features():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    script = """
import io
import numpy as np
import soundfile as sf
import streamlit as st
from ui import theme
from ui import pages
theme.inject()
def wav(f, amp):
    t = np.arange(3 * 16000) / 16000
    x = (amp * np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    b = io.BytesIO(); sf.write(b, x, 16000, format="WAV"); return b.getvalue()
st.session_state.setdefault("cmp_A", {"name": "a.wav", "data": wav(220, 0.3)})
st.session_state.setdefault("cmp_B", {"name": "b.wav", "data": wav(440, 0.1)})
pages.compare()
"""
    at = st_testing.AppTest.from_string(script, default_timeout=180).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    for kept in ("RMS energy", "Spectral centroid", "Spectral roll-off", "Crest factor"):
        assert kept in text, kept
    for gone in ("Zero-crossing rate", "Spectral flatness", "MFCC"):
        assert gone not in text, gone


def test_navigation_has_no_overview_or_about():
    from ui import nav

    assert "About" not in nav.SECTIONS
    assert not {"overview", "card", "help"} & {k for keys in nav.SECTIONS.values() for k in keys}
    assert nav.SECTIONS["Use"][0] == "detect"
