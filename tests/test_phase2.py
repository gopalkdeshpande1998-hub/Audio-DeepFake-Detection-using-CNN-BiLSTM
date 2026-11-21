"""Phase 2: front ends, silence trimming, augmentation, bootstrap CIs, robustness subset."""

import random

import numpy as np
import pandas as pd
import pytest
import torch

from adfd.config import load_config
from adfd.data.augment import Augmenter, add_noise, mp3, opus, telephone
from adfd.data.degrade import select_subset
from adfd.data.silence import speech_bounds
from adfd.metrics import bootstrap_eer, eer_from_labels
from adfd.models import build_model

SR = 16000


def tone(seconds=2.0, f=440.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.mark.parametrize("frontend, feat_dim", [("lfcc", 60), ("logmel", 80), ("cqcc", 60), ("sinc", 70)])
def test_frontends_shape_and_model_runs(frontend, feat_dim):
    cfg = load_config()
    cfg["model"] = {"name": "cnn_bilstm", "frontend": {"name": frontend}}
    model = build_model(cfg).eval()
    wav = torch.randn(2, 64000) * 0.1
    feats = model.frontend(wav)
    assert feats.shape[:3] == (2, 1, feat_dim)
    assert torch.isfinite(feats).all()
    assert model(wav).shape == (2,)


def test_speech_bounds_finds_padded_silence():
    x = np.concatenate([np.zeros(8000), tone(1.0), np.zeros(16000)]).astype(np.float32)
    start, end = speech_bounds((x * 32767).astype(np.int16))
    assert abs(start - 8000) <= 400          # within one frame
    assert abs(end - 24000) <= 800
    # An all-silent clip is left (almost) untrimmed: frames cover up to the last full frame.
    start, end = speech_bounds(np.zeros(16000, dtype=np.int16))
    assert start == 0 and end >= 16000 - 400


def test_augmenter_keeps_length_and_range():
    random.seed(0)
    np.random.seed(0)
    aug = Augmenter({"rawboost": {"p": 1.0, "algos": ["lnl", "isd", "ssi"]},
                     "codec": {"p": 1.0}, "gain": {"p": 1.0}})
    x = tone(4.0)
    for _ in range(5):
        y = aug(x)
        assert y.shape == x.shape and y.dtype == np.float32
        assert np.isfinite(y).all() and np.abs(y).max() <= 1.0 + 1e-6


def test_codecs_roundtrip_length_close():
    x = tone(2.0)
    for fn in (mp3, opus, telephone):
        y = fn(x)
        assert abs(len(y) - len(x)) < 4000  # codecs may add a short delay/padding


def test_add_noise_hits_target_snr():
    x = tone(2.0)
    for snr in (5, 20):
        y = add_noise(x, snr, rng=np.random.default_rng(0))
        measured = 10 * np.log10(np.sum(x ** 2) / np.sum((y - x) ** 2))
        assert measured == pytest.approx(snr, abs=0.05)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(1, 1, 2000), rng.normal(-1, 1, 2000)]
    labels = np.r_[np.zeros(2000), np.ones(2000)]
    eer = 100 * eer_from_labels(scores, labels)[0]
    ci = bootstrap_eer(scores, labels, n_boot=200)
    assert ci["lo_pct"] < eer < ci["hi_pct"]
    groups = np.repeat(np.arange(400), 10)
    ci_g = bootstrap_eer(scores, labels, groups, n_boot=200)
    assert ci_g["resampled"] == "groups"


def test_robustness_subset_is_stratified_and_fixed():
    rows = [{"utt_id": f"b{i}", "label": 0, "attack": "-"} for i in range(3000)]
    for a in ("A07", "A08"):
        rows += [{"utt_id": f"{a}_{i}", "label": 1, "attack": a} for i in range(1000)]
    df = pd.DataFrame(rows)
    s1, s2 = select_subset(df, 2000, 770), select_subset(df, 2000, 770)
    assert (s1["label"] == 0).sum() == 2000
    assert s1[s1["label"] == 1]["attack"].value_counts().to_dict() == {"A07": 770, "A08": 770}
    assert s1["utt_id"].tolist() == s2["utt_id"].tolist()


def test_augmenter_pickles_for_dataloader_workers():
    import pickle
    aug = Augmenter({"rawboost": {"p": 1.0}, "codec": {"p": 1.0}, "gain": {"p": 1.0}})
    y = pickle.loads(pickle.dumps(aug))(tone(1.0))
    assert y.shape == (SR,) and y.dtype == np.float32


def test_speed_and_resample_keep_length_without_silence():
    random.seed(1)
    aug = Augmenter({"resample": {"p": 1.0}, "speed": {"p": 1.0, "min": 1.1, "max": 1.1}})
    x = tone(4.0)
    y = aug(x)
    assert y.shape == x.shape and y.dtype == np.float32
    # A sped-up crop is refilled by repeating the clip, never by appending zeros.
    assert np.abs(y[-1600:]).mean() > 0.05


def test_normalize_level_rms_and_peak():
    from adfd.data.datasets import TARGET_PEAK, TARGET_RMS_DBFS, normalize_level
    x = tone(1.0, amp=0.05)
    y = normalize_level(x, "rms")
    assert 20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2))) == pytest.approx(TARGET_RMS_DBFS, abs=0.1)
    assert np.abs(normalize_level(tone(1.0, amp=0.2), "peak")).max() == pytest.approx(TARGET_PEAK, abs=1e-3)
    quiet = tone(1.0, amp=1e-4)  # gain is limited to +20 dB
    assert np.abs(normalize_level(quiet, "rms")).max() <= 1e-3 + 1e-6
    assert normalize_level(x, None) is x


def test_level_db_matches_int16_and_float():
    from adfd.data.loudness import level_db
    x = tone(1.0, amp=0.5)
    rms_f, peak_f = level_db(x)
    rms_i, peak_i = level_db((x * 32767).astype(np.int16))
    assert rms_f == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.05)
    assert peak_f == pytest.approx(20 * np.log10(0.5), abs=0.05)
    assert rms_i == pytest.approx(rms_f, abs=0.05) and peak_i == pytest.approx(peak_f, abs=0.05)
