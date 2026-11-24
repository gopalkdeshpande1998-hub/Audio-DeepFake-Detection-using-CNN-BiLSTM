"""Data for the E7b retrain: an In-the-Wild speaker split and the extra Speech_Dataset test set.

1. In-the-Wild speaker split (user decision, 2026-11-22): the 54 speakers are split,
   speaker-disjoint, into train (~45% of clips), dev (~10%, model selection) and test
   (~45%, never used for training or tuning). Speakers are assigned largest first to the
   role furthest below its target share, with a fixed shuffle for ties, so the split is
   deterministic. Writes data/manifests/in_the_wild_xd.csv (dataset column stays
   "in_the_wild", so the existing audio cache is reused) and data/splits/itw_speaker_split.json.
2. Speech_Dataset (G:/audio db/Speech_Dataset, REAL/ and FAKE/ folders, 933 + 933 clips):
   an extra test set only. Writes data/manifests/speech_dataset.csv, then its audio cache
   and silence bounds.

    python scripts/data/build_extra_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from adfd.data.audio_cache import build_cache
from adfd.data.silence import compute_silence
from adfd.paths import REPO_ROOT, load_paths

TARGETS = {"train": 0.45, "dev": 0.10, "test": 0.45}


def itw_split(paths) -> dict:
    m = pd.read_csv(paths.manifest("in_the_wild"), keep_default_na=False)
    counts = m.groupby("speaker").size()
    rng = np.random.default_rng(0)
    order = sorted(counts.index, key=lambda s: (-counts[s], rng.random()))
    got = {r: 0 for r in TARGETS}
    role = {}
    total = len(m)
    for s in order:
        r = min(TARGETS, key=lambda r: (got[r] / total) - TARGETS[r])
        role[s] = r
        got[r] += counts[s]
    out = m.copy()
    out["split"] = out["speaker"].map(role)
    out.to_csv(paths.manifest("in_the_wild_xd"), index=False)
    summary = {}
    for r in TARGETS:
        part = out[out["split"] == r]
        summary[r] = {"speakers": sorted(part["speaker"].unique()), "clips": len(part),
                      "real": int((part["label"] == 0).sum()), "fake": int((part["label"] == 1).sum())}
    d = REPO_ROOT / "data" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    (d / "itw_speaker_split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r, s in summary.items():
        print(f"In-the-Wild {r:5s}: {len(s['speakers']):2d} speakers, {s['clips']:6,} clips "
              f"({s['real']:,} real / {s['fake']:,} fake)")
    return summary


def speech_dataset(paths) -> None:
    root = paths.dataset("speech_dataset")
    rows = []
    for folder, label in (("REAL", 0), ("FAKE", 1)):
        for f in sorted((root / folder).iterdir()):
            if f.suffix.lower() not in (".flac", ".wav", ".mp3", ".ogg"):
                continue
            info = sf.info(str(f))
            rid = f"sd_{folder.lower()}_{f.stem}"
            rows.append({"utt_id": rid, "path": f"{folder}/{f.name}", "label": label, "dataset": "speech_dataset",
                         "split": "test", "speaker": "unknown", "attack": "-" if label == 0 else "unknown",
                         "start_s": 0.0, "end_s": info.duration, "duration_s": info.duration,
                         "sample_rate": info.samplerate, "channels": info.channels, "recording_id": rid})
    m = pd.DataFrame(rows)
    m.to_csv(paths.manifest("speech_dataset"), index=False)
    print(f"Speech_Dataset: {len(m):,} clips ({(m.label == 0).sum()} real / {(m.label == 1).sum()} fake), "
          f"median {m.duration_s.median():.1f} s, sample rates {sorted(m.sample_rate.unique())}")
    build_cache(m, root, paths.cache_dir, "speech_dataset", workers=3)
    compute_silence(paths.cache_dir, "speech_dataset", workers=3)
    print("Speech_Dataset cache and silence bounds built")


def main() -> None:
    paths = load_paths()
    itw_split(paths)
    speech_dataset(paths)


if __name__ == "__main__":
    main()
