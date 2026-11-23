"""Grad-CAM, attention and layer-weight explanations on a tiny CRNN (CPU, no checkpoints)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from adfd import interpret  # noqa: E402
from adfd.models.crnn import CRNN  # noqa: E402


class _Front(nn.Module):
    """Waveform (B, N) -> a (B, 1, 16, T) 'spectrogram' (framed energies), plus SSL-like layer weights."""

    def __init__(self):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.tensor([0.0, 1.0, 2.0]))

    def forward(self, wav):
        fr = wav.unfold(1, 160, 160)[:, :64]                       # (B, 64 frames, 160)
        spec = fr.reshape(fr.shape[0], 64, 16, 10).abs().mean(-1)   # (B, 64, 16)
        return spec.transpose(1, 2).unsqueeze(1)                   # (B, 1, 16, 64)


class _Model(nn.Module):
    def __init__(self, pooling="asp"):
        super().__init__()
        self.frontend = _Front()
        self.backbone = CRNN(n_feat=16, cnn="light", pooling=pooling, rnn_input=32, hidden=16)


class _An:
    def __init__(self, pooling="asp"):
        torch.manual_seed(0)
        self.model = _Model(pooling).eval()
        self.device = torch.device("cpu")
        self.crop = 64 * 160

    def _prep(self, seg):
        seg = np.asarray(seg, dtype=np.float32)
        return np.pad(seg, (0, max(0, self.crop - len(seg))))[: self.crop]


def test_grad_cam_shape_and_range():
    an = _An()
    gc = interpret.grad_cam(an, np.random.default_rng(0).standard_normal(64 * 160).astype(np.float32))
    assert gc["cam"].shape == gc["feature"].shape == (16, 64)
    assert gc["cam"].min() >= 0 and gc["cam"].max() <= 1 + 1e-6
    assert np.isfinite(gc["logit"])


def test_attention_sums_to_one():
    att = interpret.attention(_An(), np.random.default_rng(1).standard_normal(64 * 160).astype(np.float32))
    assert att is not None and abs(att["weights"].sum() - 1) < 1e-5
    assert len(att["times_s"]) == len(att["weights"])


def test_no_attention_without_asp_pooling():
    assert interpret.attention(_An("last"), np.zeros(64 * 160, np.float32)) is None


def test_layer_weights_softmax():
    w = interpret.layer_weights(_An())
    assert w.shape == (3,) and abs(w.sum() - 1) < 1e-6 and w[2] > w[1] > w[0]
