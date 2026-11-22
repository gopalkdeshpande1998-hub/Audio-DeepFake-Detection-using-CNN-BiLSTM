"""RawNet2 for anti-spoofing (Tak et al., ICASSP 2021), the ASVspoof 2021 raw-waveform baseline.

Ported from the ASVspoof 2021 baseline code (github.com/asvspoof-challenge/2021,
LA/Baseline-RawNet2/model.py) with the LA settings: 20 fixed mel-spaced sinc filters
of length 1024, residual blocks [20,20]x2 and [20,128],[128,128]x3 with filter-wise
feature-map scaling (FMS), a 3-layer GRU(1024) and FC(1024). Changes: one spoof logit
instead of two outputs. Kept on purpose: conv1 reads the block input `x`, as in the
original code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from adfd.models.aasist import FixedSincConv


class ResBlock1d(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.lrelu = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], kernel_size=3, padding=1)
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=1)
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(self.lrelu(self.bn2(out)))
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


class RawNet2(nn.Module):
    def __init__(self, sinc_filters: int = 20, first_conv: int = 1024, filts=((20, 20), (20, 128), (128, 128)),
                 gru_node: int = 1024, nb_gru_layer: int = 3, nb_fc_node: int = 1024, sample_rate: int = 16000):
        super().__init__()
        self.sinc = FixedSincConv(sinc_filters, first_conv, sample_rate)
        self.first_bn = nn.BatchNorm1d(sinc_filters)
        self.selu = nn.SELU(inplace=True)
        specs = [(filts[0], True), (filts[0], False), (filts[1], False), (filts[2], False), (filts[2], False),
                 (filts[2], False)]
        self.blocks = nn.ModuleList(ResBlock1d(f, first) for f, first in specs)
        self.fms = nn.ModuleList(nn.Linear(f[1], f[1]) for f, _ in specs)
        c = filts[-1][-1]
        self.bn_before_gru = nn.BatchNorm1d(c)
        self.gru = nn.GRU(c, gru_node, num_layers=nb_gru_layer, batch_first=True)
        self.fc1_gru = nn.Linear(gru_node, nb_fc_node)
        self.fc2_gru = nn.Linear(nb_fc_node, 1)

    def forward(self, x):                                   # (B, N)
        x = self.sinc(x.unsqueeze(1))
        x = self.selu(self.first_bn(F.max_pool1d(torch.abs(x), 3)))
        for block, fms in zip(self.blocks, self.fms):
            x = block(x)
            y = torch.sigmoid(fms(x.mean(dim=2))).unsqueeze(-1)   # filter-wise feature map scaling
            x = x * y + y
        x = self.selu(self.bn_before_gru(x)).transpose(1, 2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        return self.fc2_gru(self.fc1_gru(x[:, -1])).squeeze(1)
