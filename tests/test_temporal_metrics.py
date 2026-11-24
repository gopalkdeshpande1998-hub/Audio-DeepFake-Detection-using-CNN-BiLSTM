"""Segment-level metrics and temporal inference helpers."""

from __future__ import annotations

import numpy as np
import pytest

from adfd.temporal import centre_weights, frame_labels, regions, window_starts
from adfd.temporal_metrics import frame_metrics, runs, segment_counts, segment_summary, short_recall


def test_frame_metrics_confusion():
    m = frame_metrics(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1]))
    assert m["confusion"] == {"tp_fake": 2, "fp_real_called_fake": 1, "fn_fake_called_real": 1, "tn_real": 1}
    assert m["precision"] == pytest.approx(2 / 3) and m["fpr"] == pytest.approx(0.5)


def test_segment_matching_and_boundary_error():
    y = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 0])
    p = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1])
    c = segment_counts(y, p)
    assert runs(y) == [(1, 5), (7, 8)] and c["tp"] == 1 and c["n_pred"] == 2
    s = segment_summary([c])
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["boundary_error_s"]["start_mean"] == pytest.approx(0.16)


def test_short_recall_bins():
    y = np.array([0, 1, 0, 0] + [1] * 20)
    p = np.array([0, 0, 0, 0] + [1] * 20)
    r = short_recall([(y, p)])
    assert r["< 0.5 s"]["recall"] == 0.0 and r[">= 2 s"]["recall"] == 1.0


def test_windows_cover_every_frame_and_end_aligned():
    for n in (1, 25, 26, 37, 100):
        s = window_starts(n)
        cov = np.zeros(max(n, 25))
        for a in s:
            cov[a:a + 25] += 1
        assert cov[:n].min() >= 1 and s[-1] == max(n - 25, 0)
    assert centre_weights()[12] == 1.0 and centre_weights()[0] == pytest.approx(0.25)


def test_regions_and_silence_never_fake():
    p = np.array([0.1, 0.1, 0.9, 0.95, 0.5, 0.9])
    speech = np.array([True, True, True, True, True, False])
    labs = frame_labels(p, speech, 0.35, 0.65)
    assert labs == ["REAL", "REAL", "FAKE", "FAKE", "UNCERTAIN", "NO SPEECH"]
    r = regions(p, labs, 0.96)
    assert [(x["label"], x["start_s"], x["end_s"]) for x in r] == [
        ("REAL", 0.0, 0.32), ("FAKE", 0.32, 0.64), ("UNCERTAIN", 0.64, 0.8), ("NO SPEECH", 0.8, 0.96)]
    assert r[1]["confidence"] == pytest.approx(0.925) and r[3]["confidence"] is None


class StubTemporal:
    """Stands in for adfd.temporal.TemporalAnalyzer with fixed (test-only) frame probabilities."""

    def __init__(self, p):
        self.p = np.asarray(p, float)

    def analyze(self, x):
        from adfd.temporal import LABELS

        dur = len(x) / 16000
        labs = frame_labels(self.p, np.ones(len(self.p), bool), 0.35, 0.65)
        regs = regions(self.p, labs, dur)
        secs = {k: sum(r["end_s"] - r["start_s"] for r in regs if r["label"] == k) for k in LABELS}
        sp = secs["REAL"] + secs["FAKE"] + secs["UNCERTAIN"]
        return {"model": "stub", "frame_s": 0.16, "duration_s": dur, "thresholds": {"t_lo": 0.35, "t_hi": 0.65},
                "regions": regs, "seconds": secs, "analysed_s": sp,
                "pct_of_speech": {k: 100 * secs[k] / sp for k in ("REAL", "FAKE", "UNCERTAIN")},
                "counts": {k: sum(r["label"] == k for r in regs) for k in LABELS}}


def test_segment_model_output_drives_lab_and_card():
    from tests.test_single import StubE7a, speech
    from adfd.single import SingleModelAnalyzer
    from ui import detector as det, lab

    p = [0.05] * 19 + [0.97] * 19 + [0.04] * 12                      # 8 s: real, fake, real
    an = SingleModelAnalyzer(analyzer=StubE7a(0.2), temporal=StubTemporal(p))
    res = an.analyze(speech(8.0), explain=False)
    assert res["verdict"] == "Real" and res["temporal"]["counts"]["FAKE"] == 1   # verdict untouched
    d = lab.lane(speech(8.0), 16000, "clip.wav", "main", res)
    assert d["seg"]["source"] == "temporal"
    assert [(b[0], b[1], b[2]) for b in d["blocks"]] == [(0.0, 3.04, "REAL"), (3.04, 6.08, "FAKE"), (6.08, 8.0, "REAL")]
    html = det.segment_card(res, {}, {}, 0)
    assert "regions from the segment model" in html and "Mixed real + fake" in html
