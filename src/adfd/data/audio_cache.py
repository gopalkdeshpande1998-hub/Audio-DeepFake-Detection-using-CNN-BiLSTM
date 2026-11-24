"""Decode every recording once into a 16 kHz mono int16 cache.

Opening and decoding many small audio files each epoch is slow on Windows
(per-file overhead), and the DEEP-VOICE recordings also need resampling and
downmixing. The cache stores every unique recording of a dataset back to back
in one raw int16 file, plus an index of (recording_id, offset, length).

    <cache_dir>/audio/<dataset>.int16       raw samples, little-endian int16
    <cache_dir>/audio/<dataset>.index.npz   recording_ids, offsets, lengths
    <cache_dir>/audio/<dataset>.json        sample rate, totals, manifest hash
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr

SAMPLE_RATE = 16000


def load_mono_16k(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a file, downmix to mono, resample to target_sr, return int16."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr, quality="HQ")
    return (np.clip(audio, -1.0, 32767 / 32768) * 32768).astype(np.int16)


def _decode(path: str) -> np.ndarray:
    return load_mono_16k(path)


def cache_files(cache_dir: Path, dataset: str) -> tuple[Path, Path, Path]:
    base = cache_dir / "audio" / dataset
    return base.with_suffix(".int16"), base.with_suffix(".index.npz"), base.with_suffix(".json")


def build_cache(
    manifest: pd.DataFrame,
    dataset_root: Path,
    cache_dir: Path,
    dataset: str,
    manifest_sha256: str | None = None,
    workers: int = 8,
    progress=None,
) -> dict:
    """Decode each unique recording in the manifest into the cache."""
    recs = manifest.drop_duplicates("recording_id")[["recording_id", "path"]]
    paths = [str(dataset_root / p) for p in recs["path"]]
    data_file, index_file, meta_file = cache_files(cache_dir, dataset)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_file.with_suffix(".int16.partial")

    offsets = np.zeros(len(paths), dtype=np.int64)
    lengths = np.zeros(len(paths), dtype=np.int64)
    pos = 0
    with open(tmp, "wb") as out, ProcessPoolExecutor(max_workers=workers) as pool:
        for i, audio in enumerate(pool.map(_decode, paths, chunksize=32)):
            offsets[i], lengths[i] = pos, len(audio)
            out.write(audio.tobytes())
            pos += len(audio)
            if progress:
                progress(i + 1, len(paths))
    tmp.replace(data_file)
    np.savez(index_file, recording_ids=recs["recording_id"].to_numpy(dtype=str),
             offsets=offsets, lengths=lengths)
    meta = {
        "dataset": dataset,
        "sample_rate": SAMPLE_RATE,
        "dtype": "int16",
        "recordings": len(paths),
        "total_samples": int(pos),
        "hours": round(pos / SAMPLE_RATE / 3600, 2),
        "size_gb": round(pos * 2 / 1024**3, 2),
        "manifest_sha256": manifest_sha256,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class AudioCache:
    """Read-only access to a dataset cache. Safe to pickle into DataLoader workers."""

    def __init__(self, cache_dir: Path, dataset: str):
        self.data_file, index_file, meta_file = cache_files(cache_dir, dataset)
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"No audio cache for {dataset!r} at {self.data_file}. Run scripts/data/cache_audio.py first."
            )
        idx = np.load(index_file)
        self.index = {rid: i for i, rid in enumerate(idx["recording_ids"])}
        self.offsets = idx["offsets"]
        self.lengths = idx["lengths"]
        self.meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self._mm = None  # opened lazily in each process

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.data_file, dtype=np.int16, mode="r")
        return self._mm

    def length(self, recording_id: str) -> int:
        return int(self.lengths[self.index[recording_id]])

    def read(self, recording_id: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Samples [start, stop) of one recording as int16."""
        i = self.index[recording_id]
        n = int(self.lengths[i])
        stop = n if stop is None else min(stop, n)
        start = max(0, min(start, stop))
        off = int(self.offsets[i])
        return np.asarray(self._memmap()[off + start: off + stop])
