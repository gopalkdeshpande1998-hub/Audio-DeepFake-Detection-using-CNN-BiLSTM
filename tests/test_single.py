"""Single-model detector (adfd.single): WavLM + CNN-BiLSTM-Att (E7a) alone decides REAL / SYNTHETIC.

Stub analyzers stand in for the checkpoint (test-only scores); no audio model weights are needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

SR = 16000


class StubE7a:
    """Mimics adfd.inference.Analyzer: one calibrated P(fake), 4-s windows, identity calibration."""

    name = "e7a_wavlm_crnn_s42"

    def __init__(self, p_fake: float, calibrated: bool = True):
        self.z = math.log(p_fake / (1 - p_fake))
        self.calibrated = calibrated

    def windows(self, x):
        return [(0, len(x))]

    def prob(self, z):
        return 1 / (1 + np.exp(-np.asarray(z, float)))

    def analyze(self, x, on_batch=None):
        p = float(self.prob(self.z))
        return {"verdict": "Fake" if p >= 0.5 else "Real", "p_fake": p, "logit": self.z,
                "out_of_distribution": False, "ood_reasons": [], "signal": {}, "embedding_distance": None,
                "analysed_region_s": [0.0, len(x) / SR], "calibrated": self.calibrated,
                "timeline": [{"start_s": 0.0, "end_s": len(x) / SR, "logit": self.z, "p_fake": p}]}

    def score_segments(self, x, spans, on_batch=None):
        return np.full(len(spans), self.z)


def speech(sec=6.0):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def single(p_fake, **kw):
    from adfd.single import SingleModelAnalyzer

    return SingleModelAnalyzer(analyzer=StubE7a(p_fake, **kw))


@pytest.mark.parametrize("p,verdict", [(0.013, "Real"), (0.993, "Fake"), (0.5, "Fake"), (0.4999999, "Real")])
def test_the_one_model_decides_at_its_threshold(p, verdict):
    res = single(p).analyze(speech(), explain=False)
    assert res["verdict"] == verdict
    assert res["p_fake"] == pytest.approx(p) and res["p_real"] == pytest.approx(1 - p)
    assert res["p_fake"] + res["p_real"] == pytest.approx(1.0)
    assert "members" not in res and "decision" not in res
    assert res["model"] == "e7a_wavlm_crnn_s42" and res["model_label"] == "WavLM + CNN-BiLSTM-Att (E7a)"


def test_timeline_segments_and_explanation():
    res = single(0.9).analyze(speech(10.0), explain=True)
    assert res["time_resolved"] and res["local_window_s"] == 4.0
    assert all(w["p_fake"] is None or 0 <= w["p_fake"] <= 1 for w in res["per_second"])
    assert res["segments"] and res["segments"][0]["severity"]
    text = " ".join(res["explanation"]["lines"])
    assert "SYNTHETIC" in text and "WavLM + CNN-BiLSTM-Att (E7a) alone" in text
    assert "models" not in res["segments"][0]["evidence"]["model"][-1]      # no "x of 3 models" line


def test_registry_offers_and_loads_only_e7a(monkeypatch, tmp_path):
    """make_analyzer builds exactly one Analyzer, for E7a, and never touches the VoiceGuard loader."""
    from adfd import inference, registry, single as single_mod

    run = tmp_path / "e7a_wavlm_crnn_s42"
    run.mkdir()
    (run / "best.pt").write_bytes(b"")
    (run / "reliability.json").write_text("{}")
    monkeypatch.setattr(registry, "runs_dir", lambda: tmp_path)
    built = []
    monkeypatch.setattr(single_mod, "Analyzer", lambda d, dev: built.append(d) or StubE7a(0.2))
    monkeypatch.setattr(inference, "load_voiceguard", lambda *a, **k: pytest.fail("VoiceGuard must not load"))
    assert list(registry.available()) == ["e7a_wavlm_crnn_s42"]
    an = registry.make_analyzer(registry.MODEL)
    assert built == [run] and isinstance(an, single_mod.SingleModelAnalyzer)
    assert list(an.analyzers) == ["e7a_wavlm_crnn_s42"]


def test_trust_lookup_is_single_model():
    from adfd import trust

    t = trust.assess(0.97, [0.97])
    assert t["agree"] == 1 and t["n_members"] == 1
    assert "agree" not in t["advice"].lower() and "disagree" not in t["advice"].lower()
