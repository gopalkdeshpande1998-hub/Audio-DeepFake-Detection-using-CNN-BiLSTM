"""Reference ranges of real speech for the acoustic observations in adfd.forensics.

Measures bona fide recordings only, from non-test data: ASVspoof 2019 LA train and
dev, plus the bona fide clips of the 10 degraded dev conditions (codecs, noise,
telephone band), so ordinary channel effects count as normal rather than suspicious.
Test sets (ASVspoof eval, In-the-Wild, DEEP-VOICE) are not used.

    python scripts/evaluation/fit_forensics.py [--n 400]
Writes experiments/forensics/reference.json (percentiles 0..100 of each measurement).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from adfd.data.audio_cache import AudioCache
from adfd.forensics import REFERENCE, measure
from adfd.paths import load_paths

CONDS = ["mp3_51k", "mp3_19k", "opus_84k", "opus_10k", "telephone", "white_20db", "white_10db", "white_5db",
         "pink_10db"]
GRID = [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]


def _job(args):
    cache_dir, ds, rec = args
    x = AudioCache(cache_dir, ds).read(rec) / 32768.0
    return measure(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="recordings per source")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    paths = load_paths()
    jobs, sources = [], {}
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    for split in ("train", "dev"):
        d = asv[(asv["split"] == split) & (asv["label"] == 0)].sample(args.n, random_state=0)
        jobs += [(paths.cache_dir, "asvspoof2019_la", r) for r in d["recording_id"]]
        sources[f"ASVspoof {split}, bona fide"] = len(d)
    for c in CONDS:
        name = f"asvspoof2019_la_robustdev_{c}"
        m = pd.read_csv(paths.manifest(name), keep_default_na=False)
        d = m[m["label"] == 0]
        d = d.sample(min(len(d), args.n // 4), random_state=0)
        jobs += [(paths.cache_dir, name, r) for r in d["recording_id"]]
        sources[f"degraded dev ({c}), bona fide"] = len(d)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(_job, jobs, chunksize=8))
    df = pd.DataFrame(rows)
    pct = {k: [float(v) for v in np.nanpercentile(df[k].to_numpy(float), GRID)] for k in df.columns}
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps({"fitted_on": sources, "n": len(df), "grid": GRID, "percentiles": pct},
                                    indent=2), encoding="utf-8")
    print(df.describe().T.to_string())
    print("saved", REFERENCE)


if __name__ == "__main__":
    main()
