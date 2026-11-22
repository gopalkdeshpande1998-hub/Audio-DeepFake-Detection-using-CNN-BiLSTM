"""AASIST: spectro-temporal graph attention (Jung et al., ICASSP 2022).

Ported from the authors' MIT-licensed code (github.com/clovaai/aasist, models/AASIST.py),
with the same layer sizes (about 297k parameters). Changes, all small:
  * One spoof logit instead of two softmax outputs (the project trains with BCE).
  * The frequency-mask augmentation inside the sinc layer is left out (we use RawBoost).
  * `input="features"` accepts a (B, 1, F, T) feature map instead of a waveform, as in
    the SSL-AASIST recipe (Tak et al., Odyssey 2022): the sinc layer and abs() are skipped
    and the number of spectral graph nodes follows F. An SSL model gives only 50 frames
    per second, so by default only the first encoder block pools time (`time_pools`);
    4 s then gives 22 temporal nodes (the waveform model has 29).
Kept on purpose: the residual block applies conv1 to its input `x`, not to the
batch-normalised activation (as in the official code); the reported 0.83% EER
comes from that version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _param(*size):
    p = nn.Parameter(torch.empty(*size))
    nn.init.xavier_normal_(p)
    return p


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = _param(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x):
        x = self.input_drop(x)
        n = x.size(1)
        pair = x.unsqueeze(2).expand(-1, -1, n, -1)
        att = torch.tanh(self.att_proj(pair * pair.transpose(1, 2)))
        att = F.softmax(torch.matmul(att, self.att_weight) / self.temp, dim=-2)
        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        x = self.bn(x.reshape(-1, x.size(-1))).view(x.shape)
        return self.act(x)


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous graph attention over temporal + spectral nodes, with a master node."""

    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = _param(out_dim, 1)
        self.att_weight22 = _param(out_dim, 1)
        self.att_weight12 = _param(out_dim, 1)
        self.att_weightM = _param(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x1, x2, master=None):
        n1, n2 = x1.size(1), x2.size(1)
        x = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        if master is None:
            master = x.mean(dim=1, keepdim=True)
        x = self.input_drop(x)
        # attention map between all nodes, with a separate weight per node-type pair
        pair = x.unsqueeze(2).expand(-1, -1, x.size(1), -1)
        a = torch.tanh(self.att_proj(pair * pair.transpose(1, 2)))
        board = torch.zeros_like(a[..., :1])
        board[:, :n1, :n1] = torch.matmul(a[:, :n1, :n1], self.att_weight11)
        board[:, n1:, n1:] = torch.matmul(a[:, n1:, n1:], self.att_weight22)
        board[:, :n1, n1:] = torch.matmul(a[:, :n1, n1:], self.att_weight12)
        board[:, n1:, :n1] = torch.matmul(a[:, n1:, :n1], self.att_weight12)
        att = F.softmax(board / self.temp, dim=-2)
        # master node update
        am = torch.tanh(self.att_projM(x * master))
        am = F.softmax(torch.matmul(am, self.att_weightM) / self.temp, dim=-2)
        master = self.proj_with_attM(torch.matmul(am.squeeze(-1).unsqueeze(1), x)) + self.proj_without_attM(master)
        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        x = self.act(self.bn(x.reshape(-1, x.size(-1))).view(x.shape))
        return x.narrow(1, 0, n1), x.narrow(1, n1, n2), master


class GraphPool(nn.Module):
    """Keep the top-k fraction of nodes, weighted by a learned sigmoid score."""

    def __init__(self, k, in_dim, p):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, h):
        scores = torch.sigmoid(self.proj(self.drop(h)))
        n = max(int(h.size(1) * self.k), 1)
        _, idx = torch.topk(scores, n, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, h.size(2)))


def mel_sinc_filters(out_channels: int, kernel_size: int, sample_rate: int) -> torch.Tensor:
    """Fixed band-pass filters with mel-spaced edges (AASIST / RawNet2 `CONV`)."""
    to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)  # noqa: E731
    to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)  # noqa: E731
    f = (sample_rate // 2) * np.linspace(0, 1, 257)
    fmel = to_mel(f)
    edges = to_hz(np.linspace(fmel.min(), fmel.max(), out_channels + 1))
    hsupp = np.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
    bank = np.zeros((out_channels, kernel_size), dtype=np.float32)
    for i in range(out_channels):
        lo, hi = edges[i], edges[i + 1]
        ideal = (2 * hi / sample_rate) * np.sinc(2 * hi * hsupp / sample_rate) - \
                (2 * lo / sample_rate) * np.sinc(2 * lo * hsupp / sample_rate)
        bank[i] = np.hamming(kernel_size) * ideal
    return torch.from_numpy(bank).unsqueeze(1)


class FixedSincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.register_buffer("filters", mel_sinc_filters(out_channels, kernel_size, sample_rate))

    def forward(self, x):  # (B, 1, N) -> (B, C, N - K + 1)
        return F.conv1d(x, self.filters.to(x.dtype))


class ResBlock2d(nn.Module):
    def __init__(self, nb_filts, first=False, pool=True):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1))
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1))
        self.mp = nn.MaxPool2d((1, 3)) if pool else nn.Identity()

    def forward(self, x):
        identity = x
        # The official code computes bn1/selu here but feeds x to conv1; kept for fidelity.
        out = self.conv1(x)
        out = self.conv2(self.selu(self.bn2(out)))
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


class AASIST(nn.Module):
    def __init__(self, input: str = "wave", in_freq: int = 70, first_conv: int = 128, sample_rate: int = 16000,
                 filts=(70, (1, 32), (32, 32), (32, 64), (64, 64)), gat_dims=(64, 32),
                 pool_ratios=(0.5, 0.7, 0.5, 0.5), temperatures=(2.0, 2.0, 100.0, 100.0), time_pools=None):
        super().__init__()
        self.input = input
        if input == "wave":
            in_freq = filts[0]
            self.conv_time = FixedSincConv(filts[0], first_conv, sample_rate)
        self.first_bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        self.selu = nn.SELU(inplace=True)
        if time_pools is None:
            time_pools = 6 if input == "wave" else 1
        specs = [filts[1], filts[2], filts[3], filts[4], filts[4], filts[4]]
        self.encoder = nn.Sequential(*(ResBlock2d(f, first=i == 0, pool=i < time_pools) for i, f in enumerate(specs)))
        c = filts[-1][-1]
        self.pos_S = nn.Parameter(torch.randn(1, in_freq // 3, c))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(c, gat_dims[0], temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(c, gat_dims[0], temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 1)

    def forward(self, x):
        if self.input == "wave":                       # (B, N)
            x = self.conv_time(x.unsqueeze(1)).unsqueeze(1)
            x = F.max_pool2d(torch.abs(x), (3, 3))
        else:                                          # (B, 1, F, T)
            x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))
        e = self.encoder(x)                            # (B, C, F', T')

        e_S = torch.max(torch.abs(e), dim=3)[0].transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T = torch.max(torch.abs(e), dim=2)[0].transpose(1, 2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1, out_T1 = self.pool_hS1(out_S1), self.pool_hT1(out_T1)
        t, s, m = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1, out_S1, master1 = out_T1 + t, out_S1 + s, master1 + m

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2, out_T2 = self.pool_hS2(out_S2), self.pool_hT2(out_T2)
        t, s, m = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2, out_S2, master2 = out_T2 + t, out_S2 + s, master2 + m

        d = self.drop_way
        out_T = torch.max(d(out_T1), d(out_T2))
        out_S = torch.max(d(out_S1), d(out_S2))
        master = torch.max(d(master1), d(master2))
        hidden = torch.cat([torch.abs(out_T).max(dim=1)[0], out_T.mean(dim=1),
                            torch.abs(out_S).max(dim=1)[0], out_S.mean(dim=1), master.squeeze(1)], dim=1)
        return self.out_layer(self.drop(hidden)).squeeze(1)
