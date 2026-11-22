"""VoiceGuard port, merged detector decision rule, and the explanation helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adfd import forensics
from adfd.ensemble import EnsembleAnalyzer, local_grid, step_logodds
from adfd.models.voiceguard import CLIP, LogMelZ, VoiceGuardDetector, fit_clip

SR = 16000


def _speechlike(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 15)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    return (0.3 * x / np.abs(x).max() + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def test_logmel_matches_librosa():
    librosa = pytest.importorskip("librosa")
    x = fit_clip(_speechlike(4.2))
    mel = librosa.feature.melspectrogram(y=x, sr=SR, n_mels=128, n_fft=1024, hop_length=256)
    db = librosa.power_to_db(mel, ref=np.max)
    ref = (db - db.mean()) / (db.std() + 1e-8)
    got = LogMelZ()(torch.from_numpy(x)[None])[0].numpy()
    assert got.shape == (128, 313)
    assert np.abs(got - ref).max() < 1e-3


def test_fit_clip_zero_pads_and_cuts():
    assert len(fit_clip(np.ones(100))) == CLIP and fit_clip(np.ones(100))[100:].sum() == 0
    assert len(fit_clip(np.ones(CLIP + 5))) == CLIP


def test_detector_forward_and_saliency_shapes():
    m = VoiceGuardDetector().eval()
    wav = torch.from_numpy(fit_clip(_speechlike(2.0)))[None]
    assert m(wav).shape == (1,)
    feat = m.frontend(wav).detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        m.backbone(feat).sum().backward()
    assert feat.grad.shape == (1, 1, 128, 313)


CFG = {"fusion": "clipped_weighted_logodds", "decision": "binary", "clip": 4.0, "A": 1.0, "B": 0.0,
       "members": [{"run": r, "label": r, "weight": w, "a": 1.0, "b": 0.0} for r, w in zip("abc", (0.45, 0.35, 0.2))],
       "local": {"window_s": 2.0, "A": 1.0, "B": 0.0}}


def ens(analyzers=None):
    return EnsembleAnalyzer(CFG, analyzers=analyzers or {r: None for r in "abc"})


@pytest.mark.parametrize("p, ood", [(0.0, False), (0.2, False), (0.4999, False), (0.5, False), (0.5001, True),
                                    (0.9, True), (1.0, False)])
def test_decision_is_always_real_or_fake(p, ood):
    v, gate = ens().decide(p, [0.5, 0.5, 0.5], ood)
    assert v == ("Fake" if p >= 0.5 else "Real") and gate is None      # never "Uncertain", OOD only warns


def _single_vs_pair(e: EnsembleAnalyzer, fuse) -> None:
    """Any one member fully confident one way, the other two fully confident the other way:
    the pair must win, for the file verdict and the local timeline alike."""
    for i in range(3):
        for sign in (+1, -1):
            z = np.full(3, -sign * 20.0)
            z[i] = sign * 20.0
            assert (fuse(z)[0] >= 0.5) == (sign < 0), (i, sign)


def test_no_single_model_decides():
    e = ens()
    _single_vs_pair(e, e.fuse)
    _single_vs_pair(e, e.fuse_local)


def test_shipped_config_obeys_no_single_model_rule():
    from adfd.ensemble import ensemble_dir, load_config

    f = ensemble_dir() / "ensemble.json"
    if not f.exists():
        pytest.skip("no fitted ensemble on this machine")
    e = EnsembleAnalyzer(load_config(f), analyzers={m["run"]: None for m in load_config(f)["members"]})
    assert abs(sum(m["weight"] for m in e.members) - 1) < 1e-9
    assert max(m["weight"] for m in e.members) < 0.5
    _single_vs_pair(e, e.fuse)
    _single_vs_pair(e, e.fuse_local)


def test_local_grid_and_steps():
    spans, step, win = local_grid(10 * SR)
    assert win == 2.0 and step == 0.5 and spans[0] == (0, 2 * SR) and spans[-1][1] == 10 * SR
    z = np.array([5.0 if a < 4 * SR else -5.0 for a, _ in spans])          # windows starting before 4 s: fake
    t0, zs = step_logodds(spans, z, 10 * SR, step)
    assert len(t0) == 20 and np.isfinite(zs).all()
    assert zs[0] > 0 > zs[-1]
    # a window marked invalid (no speech) leaves its steps unscored instead of inventing a value
    t0, zs = step_logodds([(0, 2 * SR), (2 * SR, 4 * SR)], np.array([1.0, 2.0]), 4 * SR, 0.5,
                          valid=np.array([True, False]))
    assert np.isfinite(zs[:4]).all() and np.isnan(zs[4:]).all()
    # long files: at most MAX_LOCAL windows, and the windows still tile the whole file
    spans, step, win = local_grid(600 * SR)
    assert len(spans) <= 121 and win >= step and spans[-1][1] == 600 * SR


class _Stub:
    """Member stub: file logit fixed; local logit +4 where the signal is loud (the "fake" part), -4 elsewhere."""

    def __init__(self, file_logit):
        self.file_logit = file_logit

    def analyze(self, x):
        return {"logit": self.file_logit, "timeline": [{"start_s": 0.0, "end_s": len(x) / SR, "logit": self.file_logit,
                                                        "p_fake": 0.5}], "ood_reasons": [], "signal": {}}

    def score_segments(self, x, spans):
        return np.array([4.0 if np.abs(x[a:b]).mean() > 0.1 else -4.0 for a, b in spans])


def _mixed_clip():
    """10 s: quiet speech-like 0-4 s, loud 4-7 s (the stub's "fake" part), quiet 7-10 s."""
    x = _speechlike(10.0) * 0.4
    x[4 * SR:7 * SR] = _speechlike(3.0, seed=1) * 3.0
    return np.clip(x, -1, 1).astype(np.float32)


def test_analyze_timeline_is_local_not_the_file_score():
    e = ens({r: _Stub(-3.0) for r in "abc"})            # every member: file looks REAL
    res = e.analyze(_mixed_clip(), explain=True)
    assert res["verdict"] == "Real" and abs(res["p_fake"] + res["p_real"] - 1) < 1e-12
    p = np.array([w["p_fake"] for w in res["per_second"]], float)
    assert np.nanstd(p) > 0.2                           # not a flat line of the file score
    assert p[0] < 0.5 and p[11] >= 0.5 and p[-1] < 0.5  # 0.0-0.5 s real, 5.5-6.0 s fake, end real
    assert {r["label"] for r in res["regions"]} <= {"Real", "Fake", "No speech"}
    (seg,) = res["segments"]                             # one flagged segment around 4-7 s (2 s windows blur it)
    assert 2.5 <= seg["start_s"] <= 4.5 and 6.5 <= seg["end_s"] <= 8.5
    assert seg["severity"] in {"Low", "Medium", "High"}
    ev = seg["evidence"]
    assert ev["model"] and any("3 of 3 models" in t for t in ev["model"])
    assert any(a["cue"] == "level_db" for a in ev["acoustic"])       # the loud part is measured as louder
    assert all("Possible reading" in a["statement"] or a["kind"] != "contrast" for a in ev["acoustic"])
    assert "Uncertain" not in " ".join(res["explanation"]["lines"])


def test_short_clip_is_marked_not_time_resolved():
    e = ens({r: _Stub(3.0) for r in "abc"})
    res = e.analyze(_speechlike(1.5), explain=False)
    assert res["verdict"] == "Fake" and not res["time_resolved"] and len(res["per_second"]) == 1


def test_silence_is_not_scored():
    x = np.concatenate([np.zeros(3 * SR, np.float32), _speechlike(4.0)])
    res = ens({r: _Stub(0.0) for r in "abc"}).analyze(x, explain=False)
    assert res["per_second"][0]["p_fake"] is None and res["regions"][0]["label"] == "No speech"
    assert res["per_second"][-1]["p_fake"] is not None


def test_regions_merge_consecutive_seconds():
    r = forensics.regions(np.arange(6.0), np.array([.9, .9, .5, .1, .1, .9]), 0.3, 0.7, 1.0)
    assert [x["label"] for x in r] == ["Fake", "Uncertain", "Real", "Fake"]
    assert r[0]["end_s"] == 2.0


def test_measure_and_observations():
    meas = forensics.measure(_speechlike(3.0))
    assert set(forensics.CUES) <= set(meas)
    ref = {"grid": [0, 5, 50, 95, 100], "percentiles": {k: [0, 1, 2, 3, 4] for k in forensics.CUES}}
    meas = {k: 10.0 for k in forensics.CUES}           # far above every reference value
    obs = forensics.observations(meas, ref)
    assert obs and all(o["side"] == "high" for o in obs)


def test_band_shares_sum_to_100():
    shares = forensics.band_shares(np.random.default_rng(0).random((128, 50)))
    assert abs(sum(b["share_pct"] for b in shares) - 100) < 1e-6
