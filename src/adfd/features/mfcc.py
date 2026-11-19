"""MFCC front end on the GPU, matching the v1 (librosa) feature settings."""

from __future__ import annotations

import torch
import torchaudio
from torch import nn


class MFCCFrontend(nn.Module):
    """Waveform (B, N) -> MFCC (B, 1, n_mfcc, frames).

    Defaults follow librosa.feature.mfcc as used in the v1 notebooks:
    n_fft 2048, hop 512, 128 Slaney mel bands, power-to-dB with top_db 80,
    orthonormal DCT-II. A 4 s clip at 16 kHz gives 126 frames.
    """

    def __init__(self, sample_rate=16000, n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128, top_db=80.0):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            dct_type=2,
            norm="ortho",
            log_mels=False,  # power -> dB, like librosa
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "mel_scale": "slaney",
                "norm": "slaney",
                "center": True,
                "pad_mode": "constant",
            },
        )
        self.mfcc.amplitude_to_DB.top_db = top_db

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Features in float32 even under autocast, for numerical stability.
        # Shape (B, 1, N): torchaudio applies the top_db clamp per item only
        # when there is a channel dimension; with (B, N) it clamps across the batch.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return self.mfcc(wav.float().unsqueeze(1))  # (B, 1, n_mfcc, frames)
