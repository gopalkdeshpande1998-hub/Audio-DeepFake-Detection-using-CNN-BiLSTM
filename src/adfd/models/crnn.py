"""P3.2 upgraded CNN-RNN: the v1 CNN-BiLSTM with each Phase 3 change switchable.

  cnn      "v1"      3 x [conv3x3 -> ReLU -> BN -> maxpool], 32/64/128 channels (the paper's CNN)
           "se_res"  3 residual blocks (two conv3x3 + BN, squeeze-and-excitation, 1x1 skip), 32/64/128
           "light"   the se_res design with 16/32/64 channels
  sequence always time-major: frequency is folded into the features and projected to
           `rnn_input` (the v1 model read a frequency-major grid, see cnn_bilstm.py)
  rnn      "lstm" | "gru", 2 bidirectional layers
  pooling  "last"   forward-last + backward-last state (as v1)
           "asp"    attentive statistics pooling: attention-weighted mean and std over time
Input (B, 1, F, T) from any front end; output one spoof logit.

  frame_head  (temporal detection) an extra head on the BiLSTM output, before any pooling: one spoof logit per
           backbone frame (the CNN halves time three times, so a 50 frame/s WavLM input gives one frame every
           0.16 s). forward() still returns the clip logit only; forward_frames() returns (clip, frames).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True), nn.Linear(max(c // r, 4), c))

    def forward(self, x):
        w = torch.sigmoid(self.fc(x.mean(dim=(2, 3))))
        return x * w[:, :, None, None]


class SEResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SE(cout)
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return self.pool(F.relu(out + self.skip(x), inplace=True))


def v1_cnn():
    layers, cin = [], 1
    for c in (32, 64, 128):
        layers += [nn.Conv2d(cin, c, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(c), nn.MaxPool2d(2)]
        cin = c
    return nn.Sequential(*layers), 128


def se_res_cnn(channels=(32, 64, 128)):
    layers, cin = [], 1
    for c in channels:
        layers.append(SEResBlock(cin, c))
        cin = c
    return nn.Sequential(*layers), channels[-1]


class AttentiveStatsPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h):                                   # (B, T, D) -> (B, 2D)
        w = torch.softmax(self.att(h).float(), dim=1)
        h = h.float()
        mean = (w * h).sum(dim=1)
        var = (w * h * h).sum(dim=1) - mean * mean
        return torch.cat([mean, torch.sqrt(var.clamp(min=1e-6))], dim=1)


class CRNN(nn.Module):
    def __init__(self, n_feat: int = 60, cnn: str = "se_res", rnn: str = "lstm", pooling: str = "asp",
                 rnn_input: int = 256, hidden: int = 128, dropout: float = 0.3, head_dropout: float = 0.5,
                 frame_head: bool = False):
        super().__init__()
        if cnn == "v1":
            self.cnn, c = v1_cnn()
        elif cnn == "se_res":
            self.cnn, c = se_res_cnn((32, 64, 128))
        elif cnn == "light":
            self.cnn, c = se_res_cnn((16, 32, 64))
        else:
            raise ValueError(f"unknown cnn {cnn!r}")
        self.proj = nn.Sequential(nn.Linear(c * (n_feat // 8), rnn_input), nn.ReLU(inplace=True), nn.Dropout(dropout))
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn]
        self.rnn = rnn_cls(rnn_input, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pooling = pooling
        if pooling == "asp":
            self.pool = AttentiveStatsPool(2 * hidden)
            out_dim = 4 * hidden
        elif pooling == "last":
            out_dim = 2 * hidden
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        self.head = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(inplace=True), nn.Dropout(head_dropout),
                                  nn.Linear(128, 1))
        self.frame_head = (nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                         nn.Linear(64, 1)) if frame_head else None)

    def forward(self, x):                                   # (B, 1, F, T)
        return self._run(x)[0]

    def forward_frames(self, x):
        """(clip logit (B,), frame logits (B, T')) - needs frame_head=True."""
        if self.frame_head is None:
            raise RuntimeError("this model has no frame head")
        return self._run(x, frames=True)

    def _run(self, x, frames: bool = False):
        x = self.cnn(x)                                     # (B, C, F', T')
        b, c, f, t = x.shape
        x = self.proj(x.permute(0, 3, 1, 2).reshape(b, t, c * f))   # time-major (B, T', rnn_input)
        h, last = self.rnn(x)
        if self.pooling == "asp":
            z = self.pool(h)
        else:
            hn = last[0] if isinstance(last, tuple) else last         # (layers*2, B, H)
            z = torch.cat([hn[-2], hn[-1]], dim=1)
        clip = self.head(z).squeeze(1)
        return clip, (self.frame_head(h).squeeze(2) if frames else None)
