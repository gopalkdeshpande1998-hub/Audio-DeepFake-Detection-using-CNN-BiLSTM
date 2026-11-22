"""Train/dev/test splits for DEEP-VOICE and Speech_Dataset, so E8 can train on every dataset.

Until E7b these two were test-only. E8 (user decision, 2026-11-22: "make use of all data
sets less than 10 GB") trains on part of each, so each gets a held-out test part that is
never trained on or tuned on. The existing audio caches are reused (the `dataset` column is
unchanged); only new manifests are written:

1. DEEP-VOICE -> data/manifests/deep_voice_xd.csv (4-s windows, as in deep_voice.csv).
   Split by the person whose voice a recording carries: the real recording of a person
   and every fake converted INTO that person's voice (target speaker) go to that person's role.
       train: biden, linus, musk, obama   dev: margot, ryan   test: taylor, trump
   Caveat: a test fake's *source* speech can come from a train person (e.g. biden-to-trump),
   so the content is not disjoint, only the voice identity. Reported in the results.
2. Speech_Dataset -> data/manifests/speech_dataset_xd.csv. No speaker labels exist, so
   the split is by clip index (audio_i and audio_{i+933} kept together, in case REAL i and
   FAKE i+933 are a pair): 50% train, 10% dev, 40% test, fixed seed. Speakers may repeat
   across parts (unknown), so its test numbers are an upper bound.
Writes data/splits/all_data_split.json with the counts.

    python scripts/build_all_data_splits.py
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from adfd.paths import REPO_ROOT, load_paths

DV_ROLES = {"biden": "train", "linus": "train", "musk": "train", "obama": "train",
            "margot": "dev", "ryan": "dev", "taylor": "test", "trump": "test"}
SD_SHARES = {"train": 0.5, "dev": 0.1, "test": 0.4}


def deep_voice(paths) -> pd.DataFrame:
    m = pd.read_csv(paths.manifest("deep_voice"), keep_default_na=False)
    person = m["target_speaker"].str.lower()
    unknown = sorted(set(person) - set(DV_ROLES))
    if unknown:
        raise SystemExit(f"DEEP-VOICE: no role for {unknown}")
    m["split"] = person.map(DV_ROLES)
    m.to_csv(paths.manifest("deep_voice_xd"), index=False)
    return m


def speech_dataset(paths) -> pd.DataFrame:
    m = pd.read_csv(paths.manifest("speech_dataset"), keep_default_na=False)
    idx = m["path"].map(lambda p: int(re.search(r"(\d+)", p.rsplit("/", 1)[-1]).group(1)))
    n_real = int((m["label"] == 0).sum())
    pair = idx % n_real                         # audio_i and audio_{i+n_real} share a pair id
    order = np.random.default_rng(0).permutation(n_real)
    cut1, cut2 = int(SD_SHARES["train"] * n_real), int((SD_SHARES["train"] + SD_SHARES["dev"]) * n_real)
    role = np.empty(n_real, dtype=object)
    role[order[:cut1]], role[order[cut1:cut2]], role[order[cut2:]] = "train", "dev", "test"
    m["split"] = role[pair.to_numpy()]
    m.to_csv(paths.manifest("speech_dataset_xd"), index=False)
    return m


def counts(m: pd.DataFrame, by_rec: bool) -> dict:
    out = {}
    for r in ("train", "dev", "test"):
        p = m[m["split"] == r]
        d = {"rows": len(p), "real": int((p["label"] == 0).sum()), "fake": int((p["label"] == 1).sum())}
        if by_rec:
            d["recordings"] = int(p["recording_id"].nunique())
        out[r] = d
    return out


def main() -> None:
    paths = load_paths()
    summary = {"deep_voice_xd": {"roles": DV_ROLES, **counts(deep_voice(paths), True)},
               "speech_dataset_xd": {"shares": SD_SHARES, **counts(speech_dataset(paths), False)}}
    d = REPO_ROOT / "data" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    (d / "all_data_split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, s in summary.items():
        for r in ("train", "dev", "test"):
            c = s[r]
            rec = f", {c['recordings']} recordings" if "recordings" in c else ""
            print(f"{name:18s} {r:5s}: {c['rows']:6,} rows ({c['real']:,} real / {c['fake']:,} fake{rec})")


if __name__ == "__main__":
    main()
