"""LFCC-LCNN, the ASVspoof 2021 spectral baseline (Wang & Yamagishi, Interspeech 2021).

Follows the baseline in github.com/asvspoof-challenge/2021 (LA/Baseline-LFCC-LCNN,
project/baseline_LA/model.py): a light CNN with max-feature-map (MFM) activations,
two BLSTM layers with a residual connection, average pooling over time and a linear
output. Input is the LFCC front end (60 x frames, i.e. 20 LFCC + deltas, 20 ms / 10 ms),
given as (B, 1, F, T). Changes: one spoof logit; the baseline's P2SActivation /
one-class variants are not used.
"""

from __future__ import annotations

import torch
from torch import nn


class MFM2d(nn.Module):
    """Max-feature-map: split channels in two halves and keep the element-wise max."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


def _conv(cin, cout, k):
    return [nn.Conv2d(cin, cout, k, padding=k // 2), MFM2d()]


class LCNN(nn.Module):
    def __init__(self, n_feat: int = 60, dropout: float = 0.7):
        super().__init__()
        bn = lambda c: nn.BatchNorm2d(c, affine=False)  # noqa: E731
        self.cnn = nn.Sequential(
            *_conv(1, 64, 5), nn.MaxPool2d(2),
            *_conv(32, 64, 1), bn(32), *_conv(32, 96, 3), nn.MaxPool2d(2), bn(48),
            *_conv(48, 96, 1), bn(48), *_conv(48, 128, 3), nn.MaxPool2d(2),
            *_conv(64, 128, 1), bn(64), *_conv(64, 64, 3), bn(32),
            *_conv(32, 64, 1), bn(32), *_conv(32, 64, 3), nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        dim = 32 * (n_feat // 16)
        self.blstm1 = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(dim, 1)

    def forward(self, x):                                   # (B, 1, F, T)
        x = self.cnn(x.transpose(2, 3))                     # baseline layout (B, 1, T, F)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)      # (B, T', C*F')
        h, _ = self.blstm1(x)
        h, _ = self.blstm2(h)
        return self.out((h + x).mean(dim=1)).squeeze(1)
