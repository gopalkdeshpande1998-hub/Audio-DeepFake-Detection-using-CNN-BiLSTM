"""Segment-level authenticity blocks (adfd.timeline.segment_blocks): built only from model timeline steps."""

from __future__ import annotations

import pytest

from adfd.timeline import block_label, segment_blocks


def steps(ps, step=0.5):
    return [{"start_s": i * step, "end_s": (i + 1) * step, "p_fake": p, "speech": p is not None}
            for i, p in enumerate(ps)]


def test_mixed_recording_gives_binary_blocks_even_near_50_percent():
    ps = [0.02, 0.03, 0.05, 0.04, 0.97, 0.95, 0.5, 0.52, 0.1, 0.08]       # 5 s at 0.5-s steps
    sb = segment_blocks(steps(ps), 5.0, 0.5, 4.0, True)
    assert sb["available"] and sb["block_s"] == 1.0
    assert [b["label"] for b in sb["blocks"]] == ["REAL", "REAL", "FAKE", "FAKE", "REAL"]      # 51% -> FAKE
    assert [(b["start_s"], b["end_s"]) for b in sb["blocks"]][2] == (2.0, 3.0)
    fake = sb["blocks"][2]
    assert fake["confidence"] == pytest.approx(fake["p_fake"]) and 0.95 < fake["p_fake"] < 0.97
    assert sb["counts"] == {"REAL": 3, "FAKE": 2, "NO SPEECH": 0}
    assert sb["pct_of_speech"]["FAKE"] == pytest.approx(40.0)
    low = sb["blocks"][3]                                          # 51% fake: low confidence, still FAKE
    assert low["confidence"] == pytest.approx(low["p_fake"]) and 0.5 < low["p_fake"] < 0.52


def test_unscored_steps_are_no_speech_and_excluded_from_shares():
    sb = segment_blocks(steps([0.1, 0.1, None, None, 0.9, 0.9]), 3.0, 0.5, 4.0, True)
    assert [b["label"] for b in sb["blocks"]] == ["REAL", "NO SPEECH", "FAKE"]
    assert sb["blocks"][1]["confidence"] is None
    assert sb["analysed_s"] == pytest.approx(2.0) and sb["pct_of_speech"]["REAL"] == pytest.approx(50.0)


def test_single_window_clip_is_unavailable_not_invented():
    sb = segment_blocks(steps([0.8]), 3.2, 3.2, 3.2, False)
    assert not sb["available"] and sb["blocks"] == [] and "one" in sb["reason"]


def test_long_recording_keeps_block_count_bounded_and_short_tail_folded():
    n = 600 * 2
    sb = segment_blocks(steps([0.2] * n), 600.2, 0.5, 4.0, True)
    assert len(sb["blocks"]) <= 60 and sb["block_s"] == 15.0          # 600.2 s / 10 s would be 61 blocks
    assert sb["blocks"][-1]["end_s"] == pytest.approx(600.2)


@pytest.mark.parametrize("p,lab", [(None, "NO SPEECH"), (0.49, "REAL"), (0.5, "REAL"), (0.51, "FAKE"), (0.01, "REAL"),
                                   (0.99, "FAKE")])
def test_block_label_is_argmax_and_ties_are_real(p, lab):
    assert block_label(p) == lab


def _res(ps, step=0.5, resolved=True):
    dur = len(ps) * step
    return {"per_second": steps(ps, step), "duration_s": dur, "step_s": step, "local_window_s": 4.0,
            "time_resolved": resolved, "verdict": "Fake", "p_fake": 0.7, "segments": []}


def test_lab_lane_and_temporal_card_use_the_same_blocks():
    np = pytest.importorskip("numpy")
    from ui import detector as det, lab

    res = _res([0.02, 0.03, 0.05, 0.04, 0.97, 0.95, 0.9, 0.92, 0.1, 0.08])
    d = lab.lane(np.zeros(16000 * 5, np.float32), 16000, "clip.wav", "main", res)
    assert [b[2] for b in d["blocks"]] == ["REAL", "REAL", "FAKE", "FAKE", "REAL"] and d["seg"]["available"]
    html = det.segment_card(res, {}, {}, 0)
    assert "Mixed real + fake" in html and html.count("<i style='left:") == 5


def test_unavailable_state_shows_no_blocks():
    np = pytest.importorskip("numpy")
    from ui import detector as det, lab

    res = _res([0.8], step=3.0, resolved=False)
    d = lab.lane(np.zeros(16000 * 3, np.float32), 16000, "clip.wav", "main", res)
    assert "blocks" not in d and not d["seg"]["available"]
    assert "Segment-level analysis unavailable" in det.segment_card(res, {}, {}, 0)
