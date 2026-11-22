"""VoiceGuard port, merged detector decision rule, and the explanation helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adfd import forensics
from adfd.ensemble import EnsembleAnalyzer, _per_second
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


CFG = {"t_real": 0.3, "t_fake": 0.7, "vote_fake": 0.5, "vote_real": 0.5, "min_agree": 2, "veto_p": 0.95,
       "bias": 0.0, "members": [{"run": r, "label": r, "weight": 1 / 3, "a": 1.0, "b": 0.0} for r in "abc"]}


def ens():
    return EnsembleAnalyzer(CFG, analyzers={r: None for r in "abc"})


@pytest.mark.parametrize("p, members, ood, expect", [
    (0.9, [0.9, 0.8, 0.2], False, "Fake"),        # two agree
    (0.9, [0.99, 0.4, 0.3], False, "Uncertain"),  # one confident model is not enough for Fake
    (0.1, [0.1, 0.2, 0.3], False, "Real"),
    (0.1, [0.01, 0.02, 0.97], False, "Uncertain"),  # veto: one model is sure it is fake
    (0.5, [0.5, 0.5, 0.5], False, "Uncertain"),
    (0.95, [0.9, 0.9, 0.9], True, "Uncertain"),   # not speech -> no verdict
])
def test_decision_rule(p, members, ood, expect):
    assert ens().decide(p, members, ood)[0] == expect


def test_per_second_grid_covers_every_second():
    tl = [{"start_s": 0.0, "end_s": 4.0, "p_fake": 0.9}, {"start_s": 1.0, "end_s": 5.0, "p_fake": 0.1}]
    z = _per_second(tl, 7)
    assert z.shape == (7,) and np.isfinite(z).all()
    assert z[0] > 0 > z[4]           # second 0 only in window 1 (fake), second 4 only in window 2 (real)


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
