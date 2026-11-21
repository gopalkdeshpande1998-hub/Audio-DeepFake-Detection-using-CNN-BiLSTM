"""Leading/trailing silence detection, for the silence-shortcut checks (P2.2).

ASVspoof 2019 LA is known to leak the label through the amount of leading and
trailing silence (Mueller et al., 2021). Speech bounds are found with an energy
threshold relative to the loudest frame (the same idea as librosa.effects.trim)
and stored per recording next to the audio cache:

    <cache_dir>/audio/<dataset>.silence.npz
        recording_ids, speech_start, speech_end (samples), lead_s, trail_s, total_s
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from adfd.data.audio_cache import AudioCache


def speech_bounds(x: np.ndarray, top_db: float = 40.0, frame: int = 400, hop: int = 160) -> tuple[int, int]:
    """(start, end) sample indices of the non-silent region; (0, len) if all silent."""
    n = len(x)
    if n < frame:
        return 0, n
    xf = x.astype(np.float32)
    n_frames = 1 + (n - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(xf[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    voiced = np.flatnonzero(db > -top_db)
    if voiced.size == 0:
        return 0, n
    start = int(voiced[0] * hop)
    end = int(min(n, voiced[-1] * hop + frame))
    return start, end


def _worker(args):
    cache_dir, dataset, rec_ids, top_db = args
    cache = AudioCache(Path(cache_dir), dataset)
    return [speech_bounds(cache.read(r), top_db) + (cache.length(r),) for r in rec_ids]


def silence_file(cache_dir: Path, dataset: str) -> Path:
    return cache_dir / "audio" / f"{dataset}.silence.npz"


def compute_silence(cache_dir: Path, dataset: str, sample_rate: int = 16000, top_db: float = 40.0,
                    workers: int = 8, chunk: int = 2000) -> Path:
    cache = AudioCache(cache_dir, dataset)
    rec_ids = list(cache.index)
    jobs = [(str(cache_dir), dataset, rec_ids[i:i + chunk], top_db) for i in range(0, len(rec_ids), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for part in pool.map(_worker, jobs) for r in part]
    start = np.array([r[0] for r in rows], dtype=np.int64)
    end = np.array([r[1] for r in rows], dtype=np.int64)
    total = np.array([r[2] for r in rows], dtype=np.int64)
    out = silence_file(cache_dir, dataset)
    np.savez(out, recording_ids=np.array(rec_ids), speech_start=start, speech_end=end,
             lead_s=start / sample_rate, trail_s=(total - end) / sample_rate, total_s=total / sample_rate,
             top_db=top_db)
    return out


def load_silence(cache_dir: Path, dataset: str) -> dict[str, tuple[int, int]] | None:
    """recording_id -> (speech_start, speech_end), or None if not computed."""
    f = silence_file(cache_dir, dataset)
    if not f.exists():
        return None
    d = np.load(f)
    return {r: (int(s), int(e)) for r, s, e in zip(d["recording_ids"], d["speech_start"], d["speech_end"])}
