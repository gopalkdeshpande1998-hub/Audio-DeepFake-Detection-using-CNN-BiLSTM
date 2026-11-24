"""Compute leading/trailing silence for every recording (needed for trimming and shortcut checks).

Usage:
    python scripts/data/compute_silence.py                 # ASVspoof 2019 LA and In-the-Wild
    python scripts/data/compute_silence.py --top-db 40

DEEP-VOICE is skipped: its 4 s windows are cut from continuous recordings, so
leading/trailing silence of a window carries no information about the label.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from adfd.data.silence import compute_silence
from adfd.paths import load_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["asvspoof2019_la", "in_the_wild"])
    ap.add_argument("--top-db", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = load_paths()
    for name in args.datasets:
        t0 = time.time()
        out = compute_silence(paths.cache_dir, name, top_db=args.top_db, workers=args.workers)
        d = np.load(out)
        print(f"[{name}] {len(d['recording_ids']):,} recordings in {time.time() - t0:.0f}s -> {out.name}  "
              f"median lead {np.median(d['lead_s']):.2f}s, trail {np.median(d['trail_s']):.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
