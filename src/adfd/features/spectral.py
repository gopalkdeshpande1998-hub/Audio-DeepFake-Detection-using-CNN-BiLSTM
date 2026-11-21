"""Front ends for the Phase 2 feature ablation. All map (B, N) waveforms to (B, 1, F, T).

  LFCCFrontend   20 linear-frequency cepstra + delta + delta-delta (60 x frames), 20 ms / 10 ms
  LogMelFrontend 80-band log-mel spectrogram, 25 ms / 10 ms
  CQCCFrontend   constant-Q cepstra: log-power CQT (15 Hz-7.5 kHz, 24 bins/octave) resampled to a
                 linear frequency scale, DCT, 20 coefficients + deltas (60 x frames), hop 16 ms
  SincFrontend   learnable band-pass filterbank on the raw waveform (SincNet/RawNet2-style),
                 rectified, pooled to 25 ms / 10 ms frames and log-compressed

Fixed (non-learned) features are computed in float32 outside autocast.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


def _with_deltas(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) -> (B, 3C, T): static, delta, delta-delta."""
    d1 = torchaudio.functional.compute_deltas(x)
    d2 = torchaudio.functional.compute_deltas(d1)
    return torch.cat([x, d1, d2], dim=1)


class LFCCFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_lfcc=20, n_filter=70, n_fft=512, win_length=320, hop_length=160,
                 deltas=True):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate, n_filter=n_filter, n_lfcc=n_lfcc, log_lf=True,
            speckwargs={"n_fft": n_fft, "win_length": win_length, "hop_length": hop_length, "center": True},
        )
        self.deltas = deltas

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = self.lfcc(wav.float())                         # (B, n_lfcc, T)
            if self.deltas:
                x = _with_deltas(x)
            return x.unsqueeze(1)


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=512, win_length=400, hop_length=160):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=n_mels, power=2.0, center=True,
        )

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return torch.log(self.mel(wav.float()) + 1e-6).unsqueeze(1)


class CQCCFrontend(nn.Module):
    def __init__(self, sample_rate=16000, fmin=15.0, n_octaves=9, bins_per_octave=24, hop_length=256,
                 n_uniform=256, n_ceps=20, deltas=True):
        super().__init__()
        from nnAudio.features import CQT2010v2

        n_bins = n_octaves * bins_per_octave
        self.cqt = CQT2010v2(sr=sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins,
                             bins_per_octave=bins_per_octave, output_format="Magnitude",
                             pad_mode="constant", verbose=False)
        # Uniform resampling: linear interpolation from the geometric CQT bins to a linear grid.
        f_cqt = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
        f_lin = np.linspace(f_cqt[0], f_cqt[-1], n_uniform)
        interp = np.zeros((n_uniform, n_bins), dtype=np.float32)
        hi = np.clip(np.searchsorted(f_cqt, f_lin), 1, n_bins - 1)
        lo = hi - 1
        w = (f_lin - f_cqt[lo]) / (f_cqt[hi] - f_cqt[lo])
        interp[np.arange(n_uniform), lo] = 1 - w
        interp[np.arange(n_uniform), hi] = w
        self.register_buffer("interp", torch.from_numpy(interp))
        self.register_buffer("dct", torchaudio.functional.create_dct(n_ceps, n_uniform, norm="ortho"))
        self.deltas = deltas

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            power = self.cqt(wav.float()) ** 2                      # (B, n_bins, T)
            logp = torch.log(power + 1e-10)
            uniform = torch.einsum("ub,bt->ut", self.interp, logp) if logp.dim() == 2 else \
                torch.einsum("ub,nbt->nut", self.interp, logp)      # (B, n_uniform, T)
            ceps = torch.einsum("nut,uc->nct", uniform, self.dct)   # (B, n_ceps, T)
            if self.deltas:
                ceps = _with_deltas(ceps)
            return ceps.unsqueeze(1)


class SincConv(nn.Module):
    """Band-pass filters parametrised by learnable low and band cut-offs (SincNet)."""

    def __init__(self, out_channels=70, kernel_size=513, sample_rate=16000, min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size, self.sample_rate = kernel_size, sample_rate
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
        # Initialise on the mel scale between 30 Hz and Nyquist.
        def to_mel(hz): return 2595 * np.log10(1 + hz / 700)
        def to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)
        hz = to_hz(np.linspace(to_mel(30), to_mel(sample_rate / 2 - (min_low_hz + min_band_hz)), out_channels + 1))
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))
        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=kernel_size // 2)
        self.register_buffer("window", (0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    def filters(self) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        f_low, f_high = low @ self.n_, high @ self.n_
        left = ((torch.sin(f_high) - torch.sin(f_low)) / (self.n_ / 2)) * self.window
        centre = 2 * band.view(-1, 1)
        bp = torch.cat([left, centre, torch.flip(left, dims=[1])], dim=1) / (2 * band[:, None])
        return bp.view(-1, 1, self.kernel_size)

    def forward(self, x):  # (B, 1, N) -> (B, C, N)
        return F.conv1d(x, self.filters().to(x.dtype), padding=self.kernel_size // 2)


class SincFrontend(nn.Module):
    def __init__(self, sample_rate=16000, out_channels=70, kernel_size=513, win_length=400, hop_length=160):
        super().__init__()
        self.sinc = SincConv(out_channels, kernel_size, sample_rate)
        self.win, self.hop = win_length, hop_length

    def forward(self, wav):
        x = self.sinc(wav.unsqueeze(1))                          # (B, C, N), learnable
        x = F.avg_pool1d(x.abs(), kernel_size=self.win, stride=self.hop)
        return torch.log(x.float() + 1e-6).unsqueeze(1)          # (B, 1, C, T)
