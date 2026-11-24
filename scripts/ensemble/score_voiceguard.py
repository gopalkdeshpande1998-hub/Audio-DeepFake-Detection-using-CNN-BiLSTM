"""Score the two external VoiceGuard checkpoints on the project's manifests.

Writes runs/<name>/scores_<dataset>_<split>.csv with the same columns as
adfd.evaluate (utt_id, recording_id, speaker, attack, start_s, end_s, label, score),
so fusion and calibration read them like any other run. Each item is scored the way
the checkpoints were trained: the first 5 s of the item (DEEP-VOICE: 5 s from the
window start), zero-padded, no silence trimming.

    python scripts/ensemble/score_voiceguard.py                         # both models, default sets
    python scripts/ensemble/score_voiceguard.py --models vg_v2 --sets deep_voice:test
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from adfd.data.audio_cache import AudioCache
from adfd.metrics import auc, eer_from_labels
from adfd.models.voiceguard import CLIP, SR, VOICEGUARD_MODELS, fit_clip, load_voiceguard
from adfd.paths import load_paths

DEFAULT_SETS = ["asvspoof2019_la:dev", "asvspoof2019_la:eval", "in_the_wild:test", "deep_voice:test"] + \
    [f"asvspoof2019_la_robustdev_{c}:test" for c in
     ("clean", "mp3_51k", "mp3_19k", "opus_84k", "opus_10k", "telephone", "white_20db", "white_10db",
      "white_5db", "pink_10db")]


class ClipSet(Dataset):
    def __init__(self, df: pd.DataFrame, cache: AudioCache):
        self.rec = df["recording_id"].to_numpy(str)
        self.start = (df["start_s"].to_numpy(float) * SR).astype(np.int64) if "start_s" in df else np.zeros(len(df), int)
        self.windowed = df["dataset"].iloc[0] == "deep_voice"
        self.cache = cache

    def __len__(self):
        return len(self.rec)

    def __getitem__(self, i):
        a = int(self.start[i]) if self.windowed else 0
        x = self.cache.read(self.rec[i], a, a + CLIP).astype(np.float32) / 32768.0
        return torch.from_numpy(fit_clip(x))


@torch.no_grad()
def score(model, df, cache, device, batch=64, workers=3) -> np.ndarray:
    dl = DataLoader(ClipSet(df, cache), batch_size=batch, num_workers=workers, persistent_workers=False)
    out = []
    for wav in dl:
        out.append(model(wav.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(VOICEGUARD_MODELS))
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    paths = load_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {n: load_voiceguard(n, device) for n in args.models}
    summary = {n: {} for n in models}
    manifests = {}
    for entry in args.sets:
        name, split = entry.split(":")
        if name not in manifests:
            manifests[name] = pd.read_csv(paths.manifest(name), keep_default_na=False)
        df = manifests[name][manifests[name]["split"] == split].reset_index(drop=True)
        cache = AudioCache(paths.cache_dir, name)
        keep = [c for c in ("utt_id", "recording_id", "speaker", "attack", "start_s", "end_s") if c in df]
        for n, model in models.items():
            t0 = time.time()
            s = -score(model, df, cache, device, workers=args.workers)   # project convention: higher = bona fide
            run = paths.runs_dir / n
            run.mkdir(parents=True, exist_ok=True)
            df[keep].assign(label=df["label"].to_numpy(int), score=s).to_csv(
                run / f"scores_{name}_{split}.csv", index=False)
            eer = eer_from_labels(s, df["label"].to_numpy())[0]
            summary[n][entry] = {"n": len(df), "eer_pct": round(100 * eer, 2), "auc": round(auc(s, df["label"]), 4)}
            print(f"{n:12s} {entry:40s} n={len(df):6d} EER {100 * eer:6.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    for n in models:
        f = paths.runs_dir / n / "item_metrics.json"
        old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        old.update(summary[n])
        f.write_text(json.dumps(old, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
