"""Per-recording loudness statistics (P2.2 loudness shortcut check).

For every cached recording: RMS level of the speech region (silence bounds from
adfd.data.silence, so long silences do not lower the level), peak level of the
whole file, and the crest factor (peak minus RMS, in dB). Saved next to the
audio cache as <dataset>.loudness.npz.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from adfd.data.audio_cache import AudioCache
from adfd.data.silence import load_silence

FLOOR_DB = -100.0


def loudness_file(cache_dir: Path, dataset: str) -> Path:
    return Path(cache_dir) / "audio" / f"{dataset}.loudness.npz"


def level_db(x: np.ndarray) -> tuple[float, float]:
    """(RMS dBFS, peak dBFS) of an int16 or float array."""
    scale = 32768.0 if np.issubdtype(x.dtype, np.integer) else 1.0
    x = x.astype(np.float64) / scale
    if len(x) == 0:
        return FLOOR_DB, FLOOR_DB
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.abs(x).max()
    to_db = lambda v: float(max(FLOOR_DB, 20 * np.log10(v))) if v > 0 else FLOOR_DB  # noqa: E731
    return to_db(rms), to_db(peak)


def _worker(args):
    cache_dir, dataset, rec_ids, bounds = args
    cache = AudioCache(Path(cache_dir), dataset)
    out = []
    for rec in rec_ids:
        x = cache.read(rec)
        s, e = bounds.get(rec, (0, len(x))) if bounds else (0, len(x))
        speech = x[s:e] if e > s else x
        rms_db, _ = level_db(speech)
        _, peak_db = level_db(x)
        out.append((rms_db, peak_db))
    return out


def compute_loudness(cache_dir: Path, dataset: str, workers: int = 3, chunk: int = 2000) -> Path:
    cache = AudioCache(cache_dir, dataset)
    rec_ids = list(cache.index)
    bounds = load_silence(cache_dir, dataset)
    jobs = []
    for i in range(0, len(rec_ids), chunk):
        part = rec_ids[i:i + chunk]
        jobs.append((str(cache_dir), dataset, part, {r: bounds[r] for r in part if r in bounds} if bounds else None))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for part in pool.map(_worker, jobs) for r in part]
    rms = np.array([r[0] for r in rows])
    peak = np.array([r[1] for r in rows])
    out = loudness_file(cache_dir, dataset)
    np.savez(out, recording_ids=np.array(rec_ids), rms_db=rms, peak_db=peak, crest_db=peak - rms)
    return out
