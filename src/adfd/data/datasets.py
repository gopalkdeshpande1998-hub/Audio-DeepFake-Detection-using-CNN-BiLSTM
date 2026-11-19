"""PyTorch datasets that serve fixed-length crops from the audio cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from adfd.data.audio_cache import AudioCache
from adfd.seed import worker_init_fn


def fit_length(x: np.ndarray, n: int, mode: str = "repeat") -> np.ndarray:
    """Make x exactly n samples: repeat (tile) or zero-pad short clips, cut long ones."""
    if len(x) >= n:
        return x[:n]
    if len(x) == 0:
        return np.zeros(n, dtype=x.dtype)
    if mode == "repeat":
        return np.tile(x, int(np.ceil(n / len(x))))[:n]
    if mode == "zero":
        return np.pad(x, (0, n - len(x)))
    raise ValueError(f"unknown pad mode {mode!r}")


class CropDataset(Dataset):
    """One crop of `crop_samples` per manifest row.

    train=True: random crop position (a new one every epoch).
    train=False: centre crop, so scores are deterministic.
    Segments (start_s/end_s, used for DEEP-VOICE windows) are respected.
    """

    def __init__(self, manifest: pd.DataFrame, cache: AudioCache, crop_samples: int,
                 train: bool, pad_mode: str = "repeat", sample_rate: int = 16000):
        self.cache = cache
        self.crop = int(crop_samples)
        self.train = train
        self.pad_mode = pad_mode
        self.rec_ids = manifest["recording_id"].astype(str).to_numpy()
        self.starts = np.round(manifest["start_s"].to_numpy(float) * sample_rate).astype(np.int64)
        self.ends = np.round(manifest["end_s"].to_numpy(float) * sample_rate).astype(np.int64)
        self.labels = manifest["label"].to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        rec = self.rec_ids[i]
        start = self.starts[i]
        # Clamp to the cached length (resampling can shift the last sample by one).
        seg_len = max(0, min(self.ends[i], self.cache.length(rec)) - start)
        if seg_len > self.crop:
            if self.train:
                offset = np.random.randint(0, seg_len - self.crop + 1)
            else:
                offset = (seg_len - self.crop) // 2
            x = self.cache.read(rec, start + offset, start + offset + self.crop)
        else:
            x = self.cache.read(rec, start, start + seg_len)
        x = fit_length(x, self.crop, self.pad_mode)
        wav = torch.from_numpy(x.astype(np.float32) / 32768.0)
        return wav, torch.tensor(self.labels[i]), i


def make_loader(dataset: Dataset, batch_size: int, train: bool, num_workers: int, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
