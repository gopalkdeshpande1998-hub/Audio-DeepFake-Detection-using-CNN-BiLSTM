"""The v1 paper's CNN-BiLSTM, ported layer for layer from Keras to PyTorch.

Keras original (legacy/Model_Training.ipynb):
    Reshape(40, 500, 1)
    3 x [Conv2D(32/64/128, 3x3, same, relu) -> BatchNorm -> MaxPool(2x2)]
    Reshape(-1, 128)
    Bidirectional(LSTM(128, return_sequences=True, dropout=.3, recurrent_dropout=.2)) -> BatchNorm
    Bidirectional(LSTM(128, dropout=.3, recurrent_dropout=.2)) -> BatchNorm
    Dense(128, relu, L2 1e-3) -> Dropout(.5) -> Dense(1, sigmoid)

Faithful details kept on purpose:
  * Keras' Reshape(-1, 128) flattens the (freq, time) grid frequency-major into
    one sequence, so the "sequence" the BiLSTM reads is not purely temporal.
    This is reproduced exactly (a time-only sequence is a Phase 3 upgrade).
  * BatchNorm uses Keras defaults (momentum 0.99 -> PyTorch 0.01, eps 1e-3).
Unavoidable differences:
  * PyTorch's cuDNN LSTM has no recurrent dropout; input dropout (0.3) is kept.
  * The Dense layer's L2 penalty is replaced by AdamW weight decay.
  * The output is a logit (sigmoid is folded into the loss).
"""

from __future__ import annotations

import torch
from torch import nn

_BN = {"momentum": 0.01, "eps": 1e-3}


class CNNBiLSTM(nn.Module):
    def __init__(self, n_mfcc: int = 40, lstm_hidden: int = 128, lstm_dropout: float = 0.3, head_dropout: float = 0.5):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in (32, 64, 128):
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch, **_BN),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self.in_drop1 = nn.Dropout(lstm_dropout)
        self.lstm1 = nn.LSTM(128, lstm_hidden, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(2 * lstm_hidden, **_BN)
        self.in_drop2 = nn.Dropout(lstm_dropout)
        self.lstm2 = nn.LSTM(2 * lstm_hidden, lstm_hidden, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(2 * lstm_hidden, **_BN)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mfcc, frames) -> spoof logit (B,)."""
        x = self.cnn(x)                                   # (B, 128, F', T')
        b, c, _, _ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c)       # Keras Reshape(-1, 128): freq-major sequence
        x, _ = self.lstm1(self.in_drop1(x))               # (B, L, 256)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm2(self.in_drop2(x))          # h: (2, B, 128)
        x = torch.cat([h[0], h[1]], dim=1)                # forward last + backward last
        x = self.bn2(x)
        return self.head(x).squeeze(1)
