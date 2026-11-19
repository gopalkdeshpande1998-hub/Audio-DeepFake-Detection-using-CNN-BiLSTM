"""Audio cache, crop dataset, MFCC front end and CNN-BiLSTM tests (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from adfd.config import load_config
from adfd.data.audio_cache import AudioCache, build_cache
from adfd.data.datasets import CropDataset, fit_length
from adfd.features.mfcc import MFCCFrontend
from adfd.models import build_model


def test_fit_length_modes():
    x = np.array([1, 2, 3], dtype=np.int16)
    assert fit_length(x, 7, "repeat").tolist() == [1, 2, 3, 1, 2, 3, 1]
    assert fit_length(x, 5, "zero").tolist() == [1, 2, 3, 0, 0]
    assert fit_length(np.arange(10), 4).tolist() == [0, 1, 2, 3]
    assert fit_length(np.array([], dtype=np.int16), 3).tolist() == [0, 0, 0]


@pytest.fixture
def tiny_cache(tmp_path):
    """Two recordings: 2 s mono at 16 kHz, and 3 s stereo at 48 kHz (needs resampling)."""
    root = tmp_path / "data"
    root.mkdir()
    t16 = np.arange(32000) / 16000
    sf.write(root / "a.wav", 0.5 * np.sin(2 * np.pi * 440 * t16), 16000)
    t48 = np.arange(144000) / 48000
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 300 * t48)] * 2, axis=1)
    sf.write(root / "b.wav", stereo, 48000)
    manifest = pd.DataFrame({
        "utt_id": ["a", "b_0", "b_1"],
        "recording_id": ["a", "b", "b"],
        "path": ["a.wav", "b.wav", "b.wav"],
        "label": [0, 1, 1],
        "start_s": [0.0, 0.0, 1.0],
        "end_s": [2.0, 2.0, 3.0],
    })
    build_cache(manifest, root, tmp_path / "cache", "tiny", workers=1)
    return manifest, AudioCache(tmp_path / "cache", "tiny")


def test_cache_resamples_and_downmixes(tiny_cache):
    _, cache = tiny_cache
    assert cache.length("a") == 32000
    assert abs(cache.length("b") - 48000) <= 1  # 3 s at 16 kHz
    b = cache.read("b").astype(np.float32) / 32768
    assert 0.25 < np.abs(b).max() < 0.35        # mono mix of two equal channels keeps the level


def test_crop_dataset_lengths_and_determinism(tiny_cache):
    manifest, cache = tiny_cache
    crop = 16000
    eval_ds = CropDataset(manifest, cache, crop, train=False)
    w1, label, idx = eval_ds[0]
    w2, _, _ = eval_ds[0]
    assert w1.shape == (crop,) and torch.equal(w1, w2) and label.item() == 0 and idx == 0
    train_ds = CropDataset(manifest, cache, crop, train=True)
    assert all(train_ds[i][0].shape == (crop,) for i in range(3))
    # Short items are repeated, not zero-padded.
    long_ds = CropDataset(manifest, cache, 40000, train=False)
    w = long_ds[0][0].numpy()
    assert np.abs(w[-8000:]).max() > 0.1


def test_segment_window_reads_inside_its_segment(tiny_cache):
    manifest, cache = tiny_cache
    ds = CropDataset(manifest, cache, 16000, train=False)
    seg = ds[2][0].numpy()
    full = cache.read("b").astype(np.float32) / 32768
    # Centre crop of the segment [1 s, 3 s) is samples [24000, 40000).
    np.testing.assert_allclose(seg, full[24000:40000], atol=1e-6)


def test_mfcc_shape_and_per_item_clamp():
    fe = MFCCFrontend()
    loud = torch.randn(1, 64000)
    quiet = 1e-3 * torch.randn(1, 64000)
    batch = fe(torch.cat([loud, quiet]))
    single = fe(quiet)
    assert batch.shape == (2, 1, 40, 126)
    # The quiet clip's features must not depend on the loud clip in the batch.
    torch.testing.assert_close(batch[1:], single, atol=1e-3, rtol=1e-4)


def test_cnn_bilstm_forward_backward():
    cfg = load_config()
    cfg["model"] = {"name": "cnn_bilstm"}
    model = build_model(cfg)
    wav = torch.randn(4, 64000)
    logit = model(wav)
    assert logit.shape == (4,)
    logit.sum().backward()
    assert all(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)
