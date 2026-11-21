"""Degraded copies of an ASVspoof eval subset for the robustness test (P2.4).

Each condition becomes its own small "dataset" with a manifest and an audio cache:
    data/manifests/asvspoof2019_la_robust_<condition>.csv
    <cache_dir>/audio/asvspoof2019_la_robust_<condition>.int16 (+ .index.npz, .json)
so adfd.evaluate can score it like any other test set:
    --sets asvspoof2019_la_robust_mp3_19k:test ...
Degradations are deterministic (seeded per condition and clip).
"""

from __future__ import annotations

import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from adfd.data.audio_cache import AudioCache, cache_files
from adfd.data.augment import add_noise, mp3, opus, telephone

SR = 16000
PREFIX = "asvspoof2019_la_robust_"         # degraded eval subset (reported only)
DEV_PREFIX = "asvspoof2019_la_robustdev_"  # degraded dev subset (used for model selection)

# name: (description, function(x, rng) -> y)
CONDITIONS = {
    "clean": ("no degradation (reference)", lambda x, rng: x),
    "mp3_51k": ("MP3, about 51 kbps", lambda x, rng: mp3(x, SR, 0.0)),
    "mp3_19k": ("MP3, about 19 kbps", lambda x, rng: mp3(x, SR, 0.99)),
    "opus_84k": ("Opus, about 84 kbps", lambda x, rng: opus(x, SR, 0.7)),
    "opus_10k": ("Opus, about 10 kbps", lambda x, rng: opus(x, SR, 0.99)),
    "telephone": ("300-3400 Hz, 8 kHz mu-law", lambda x, rng: telephone(x, SR)),
    "white_20db": ("white noise, SNR 20 dB", lambda x, rng: add_noise(x, 20, "white", rng)),
    "white_10db": ("white noise, SNR 10 dB", lambda x, rng: add_noise(x, 10, "white", rng)),
    "white_5db": ("white noise, SNR 5 dB", lambda x, rng: add_noise(x, 5, "white", rng)),
    "pink_10db": ("pink noise, SNR 10 dB", lambda x, rng: add_noise(x, 10, "pink", rng)),
}


def select_subset(eval_df: pd.DataFrame, n_bonafide=2000, n_per_attack=770, seed=0) -> pd.DataFrame:
    parts = [eval_df[eval_df["label"] == 0].sample(n=n_bonafide, random_state=seed)]
    for _, g in eval_df[eval_df["label"] == 1].groupby("attack"):
        parts.append(g.sample(n=min(n_per_attack, len(g)), random_state=seed))
    return pd.concat(parts).sort_values("utt_id").reset_index(drop=True)


def _job(args):
    cache_dir, src_name, cond, rec_ids = args
    src = AudioCache(Path(cache_dir), src_name)
    fn = CONDITIONS[cond][1]
    out = []
    for r in rec_ids:
        x = src.read(r).astype(np.float32) / 32768.0
        rng = np.random.default_rng(zlib.crc32(f"{cond}/{r}".encode()))
        y = np.asarray(fn(x, rng), dtype=np.float32)
        y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))  # keep the original length
        out.append((np.clip(y, -1.0, 32767 / 32768) * 32768).astype(np.int16))
    return out


def build_condition(subset: pd.DataFrame, cache_dir: Path, manifests_dir: Path, cond: str,
                    src_name: str = "asvspoof2019_la", workers: int = 6, chunk: int = 250,
                    prefix: str = PREFIX) -> dict:
    name = prefix + cond
    manifest = subset.assign(dataset=name, split="test")
    manifest.to_csv(manifests_dir / f"{name}.csv", index=False)
    rec_ids = list(manifest["recording_id"])
    data_file, index_file, meta_file = cache_files(cache_dir, name)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(rec_ids), dtype=np.int64)
    lengths = np.zeros(len(rec_ids), dtype=np.int64)
    jobs = [(str(cache_dir), src_name, cond, rec_ids[i:i + chunk]) for i in range(0, len(rec_ids), chunk)]
    pos, k = 0, 0
    with open(data_file, "wb") as f, ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_job, jobs):
            for y in part:
                offsets[k], lengths[k] = pos, len(y)
                f.write(y.tobytes())
                pos += len(y)
                k += 1
    np.savez(index_file, recording_ids=np.array(rec_ids), offsets=offsets, lengths=lengths)
    meta = {"dataset": name, "condition": cond, "description": CONDITIONS[cond][0], "sample_rate": SR,
            "recordings": len(rec_ids), "total_samples": int(pos), "source": src_name}
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
