"""The two external VoiceGuard checkpoints (CNN + BiLSTM on a 128-band log-mel input).

Both checkpoints share one architecture and one preprocessing recipe, copied from
their training code (G:\\VoiceGuard_GPU\\train_voiceguard_v2.py and the VoiceGuard
backend's audio_processor.py):

    16 kHz mono -> first 5.0 s, zero-padded -> mel (n_fft 1024, hop 256, 128 bands,
    Slaney scale, power 2) -> power_to_db(ref=max, top_db=80) -> per-clip z-score
    -> spectral branch input (1, 128, 313) and temporal branch input (313, 128).

`LogMelZ` re-implements that recipe in torch (checked against librosa in
tests/test_voiceguard.py), so a batch is scored on the GPU and gradients can be
traced back onto the mel spectrogram for explanations.

    VOICEGUARD_MODELS   name -> checkpoint path, label and what it was trained on
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

SR = 16000
CLIP_S = 5.0
CLIP = int(SR * CLIP_S)
N_MELS, N_FFT, HOP = 128, 1024, 256

VOICEGUARD_MODELS = {
    "vg_improved": {
        "path": os.environ.get("ADFD_VG_IMPROVED", r"G:\audio db\improved model\synthetic_speech_detector_improved.pth"),
        "label": "VoiceGuard improved (CNN + BiLSTM, log-mel)",
        "trained_on": "unrecorded dataset (not In-the-Wild: its In-the-Wild evaluation was external)",
        "contaminated_sets": [],
    },
    "vg_v2": {
        "path": os.environ.get("ADFD_VG_V2", r"G:\VoiceGuard_GPU\outputs\synthetic_speech_detector_v2.pth"),
        "label": "VoiceGuard v2 (CNN + BiLSTM, log-mel, In-the-Wild trained)",
        "trained_on": "In-the-Wild, speaker-independent split (20,460 train / 4,645 val / 6,674 test clips)",
        # About 79% of In-the-Wild was its training or validation data, and the split
        # files are not available, so no In-the-Wild number of this model is a test result.
        "contaminated_sets": ["in_the_wild"],
    },
}


class SyntheticSpeechDetector(nn.Module):
    """Architecture of both checkpoints (layer names must match their state_dicts)."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))

    def forward(self, spectral: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        cnn = self.cnn(spectral).flatten(1)
        lstm_out, _ = self.lstm(temporal)
        return self.fc(torch.cat([cnn, lstm_out[:, -1, :]], dim=1))


class LogMelZ(nn.Module):
    """waveform (B, 80000) -> per-clip z-scored log-mel (B, 128, 313), as librosa computes it."""

    def __init__(self):
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=0.0, f_max=SR / 2,
            power=2.0, center=True, pad_mode="constant", norm="slaney", mel_scale="slaney")

    def db(self, wav: torch.Tensor) -> torch.Tensor:
        """power_to_db(mel, ref=np.max, top_db=80), per clip."""
        s = self.mel(wav.float())
        log = 10.0 * torch.log10(torch.clamp(s, min=1e-10))
        ref = 10.0 * torch.log10(torch.clamp(s.amax(dim=(-2, -1), keepdim=True), min=1e-10))
        log = log - ref
        return torch.maximum(log, log.amax(dim=(-2, -1), keepdim=True) - 80.0)

    @staticmethod
    def zscore(db: torch.Tensor) -> torch.Tensor:
        m = db.mean(dim=(-2, -1), keepdim=True)
        s = db.std(dim=(-2, -1), keepdim=True, unbiased=False)   # numpy std
        return (db - m) / (s + 1e-8)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.zscore(self.db(wav))


class VoiceGuardDetector(nn.Module):
    """Waveform in, spoof logit out, like every other model in adfd.

    `frontend` gives the z-scored log-mel; `backbone` takes it, so the generic
    gradient x input saliency in adfd.inference works on the mel spectrogram itself.
    """

    def __init__(self):
        super().__init__()
        self.frontend = _MelAsImage()
        self.backbone = _VGBackbone()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.frontend(wav))


class _MelAsImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.logmel = LogMelZ()

    def forward(self, wav):
        return self.logmel(wav).unsqueeze(1)   # (B, 1, 128, 313)


class _VGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SyntheticSpeechDetector()

    def forward(self, feat):                    # feat: (B, 1, 128, 313)
        return self.net(feat, feat[:, 0].transpose(1, 2)).squeeze(-1)


def fit_clip(x) -> "torch.Tensor":
    """First 5 s, zero-padded at the end: exactly the training input."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if len(x) >= CLIP:
        return x[:CLIP]
    return np.pad(x, (0, CLIP - len(x)))


def load_voiceguard(name: str, device="cpu") -> VoiceGuardDetector:
    path = Path(VOICEGUARD_MODELS[name]["path"])
    if not path.exists():
        raise FileNotFoundError(f"{name}: checkpoint not found at {path} (set ADFD_VG_IMPROVED / ADFD_VG_V2)")
    model = VoiceGuardDetector()
    model.backbone.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def available_voiceguard() -> list[str]:
    return [n for n, m in VOICEGUARD_MODELS.items() if Path(m["path"]).exists()]
