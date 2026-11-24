"""PartialSpoof v1.2 (partially spoofed English speech) and frame-labelled training crops.

Labels: PartialSpoof's segment labels at 0.16 s ('0' = spoof, '1' = bona fide) are inverted here to the
project convention (1 = fake, 0 = real). One label frame covers samples [k * HOP, (k + 1) * HOP) at 16 kHz.

Model frames: the temporal CRNN halves WavLM's 50 frames/s three times, so one output frame per 0.16 s.
WavLM turns N samples into floor((N - 400) / 320) + 1 frames, so a crop of K * HOP + 400 samples gives exactly
K backbone frames, frame k covering [k * HOP, (k + 1) * HOP). Crops always start on the 0.16-s grid so the
labels stay exact.

Fully real / fully fake clips (ASVspoof 2019 LA train, from the audio cache) get constant frame labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
FRAME_S = 0.16
HOP = int(FRAME_S * SR)            # 2560 samples per label frame
EXTRA = 400                        # WavLM receptive field beyond the last hop
IGNORE = -1                        # frame label ignored by the loss


def samples_for(frames: int) -> int:
    return frames * HOP + EXTRA


def frames_for(n_samples: int) -> int:
    """Backbone frames the temporal model produces for n_samples of audio."""
    if n_samples < EXTRA + 320:
        return 0
    return ((n_samples - EXTRA) // 320 + 1) // 8


def load_seglab(path) -> dict[str, np.ndarray]:
    """utt_id -> int8 frame labels, 1 = fake, 0 = real (PartialSpoof stores '0' = spoof)."""
    raw = np.load(path, allow_pickle=True).item()
    return {k: (np.asarray(v).astype(np.int8) == 0).astype(np.int8) for k, v in raw.items()}


def read_vad(path) -> list[tuple[float, float, int]]:
    """PartialSpoof .vad: (start_s, end_s, label) with 0 = spoof, 1 = bona fide, 2 = non-speech."""
    out = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) == 3:
            out.append((float(p[0]), float(p[1]), int(p[2])))
    return out


def vad_to_regions(vad: list[tuple[float, float, int]]) -> list[tuple[float, float]]:
    """Merged fake (spoof) regions in seconds from vad entries; non-speech inside a fake run stays in it."""
    regions, cur = [], None
    for a, b, lab in vad:
        if lab == 0:
            cur = [cur[0], b] if cur is not None else [a, b]
        elif lab == 1 and cur is not None:
            regions.append(tuple(cur))
            cur = None
        elif lab == 2 and cur is not None:
            cur[1] = cur[1]                          # silence: keep the run open, do not extend it
    if cur is not None:
        regions.append(tuple(cur))
    return regions


class TemporalCropDataset(Dataset):
    """Fixed crops of `frames` label frames with frame labels.

    rows: DataFrame with columns source ('ps' | 'asv'), utt_id, path (ps), recording_id (asv), label (clip,
    1 = fake). `seglab` maps PartialSpoof utt_id -> frame labels. ASVspoof rows read from `cache`.
    train=True: random grid-aligned crop; False: the first crop (deterministic). Clips shorter than the crop
    are repeated together with their labels (as the whole-clip model's repeat padding).
    """

    def __init__(self, rows: pd.DataFrame, seglab: dict[str, np.ndarray], frames: int, train: bool,
                 cache=None, transform=None):
        self.rows = rows.reset_index(drop=True)
        self.seglab = seglab
        self.frames = int(frames)
        self.train = train
        self.cache = cache
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def _audio_labels(self, r) -> tuple[np.ndarray, np.ndarray]:
        if r.source == "ps":
            x, sr = sf.read(r.path, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                raise ValueError(f"{r.utt_id}: expected 16 kHz, got {sr}")
            lab = self.seglab[r.utt_id]
        else:
            x = self.cache.read(r.recording_id).astype(np.float32) / 32768.0
            lab = np.full(int(np.ceil(len(x) / HOP)), int(r.label), np.int8)
        n = len(lab)
        x = np.pad(x, (0, max(0, samples_for(n) - len(x))))[: samples_for(n)]
        return x, lab

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        x, lab = self._audio_labels(r)
        n, k = len(lab), self.frames
        if n < k:                                   # repeat audio and labels together, on the grid
            reps = int(np.ceil(k / n)) + 1
            core = x[: n * HOP]
            x = np.concatenate([np.tile(core, reps), np.zeros(EXTRA, np.float32)])
            lab = np.tile(lab, reps)
            n = len(lab)
        start = np.random.randint(0, n - k + 1) if self.train else 0
        seg = x[start * HOP: start * HOP + samples_for(k)]
        seg = np.pad(seg, (0, samples_for(k) - len(seg))).astype(np.float32)
        if self.transform is not None and self.train:
            seg = np.ascontiguousarray(self.transform(seg), dtype=np.float32)[: samples_for(k)]
            seg = np.pad(seg, (0, samples_for(k) - len(seg)))
        y = lab[start: start + k].astype(np.float32)
        clip = float(y.max() > 0)
        return torch.from_numpy(seg), torch.from_numpy(y), torch.tensor(clip), i


def full_utterance(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: expected 16 kHz, got {sr}")
    return x
