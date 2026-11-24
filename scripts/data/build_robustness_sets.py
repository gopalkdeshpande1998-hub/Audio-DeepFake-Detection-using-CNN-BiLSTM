"""Build the degraded ASVspoof eval subsets for the robustness test (P2.4).

Usage:
    python scripts/data/build_robustness_sets.py              # all conditions
    python scripts/data/build_robustness_sets.py --only mp3_19k telephone
    python scripts/data/build_robustness_sets.py --split dev  # degraded DEV subset, for model selection

Subset: 2,000 bona fide + 770 clips per attack (A07-A19) from ASVspoof 2019 LA eval,
fixed seed, identical across conditions. With --split dev the same is done for the dev
split (attacks A01-A06) under the name asvspoof2019_la_robustdev_<cond>; that set is
used to choose between augmentations, so the eval subset stays untouched by selection. Score with:
    python -m adfd.evaluate --run runs/<run> --tag robust --sets asvspoof2019_la_robust_clean:test ...
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from adfd.data.degrade import CONDITIONS, DEV_PREFIX, PREFIX, build_condition, select_subset
from adfd.paths import load_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(CONDITIONS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    args = ap.parse_args()
    paths = load_paths()
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    subset = select_subset(asv[asv["split"] == args.split])
    print(f"Subset: {len(subset):,} clips ({int((subset['label'] == 0).sum()):,} real, "
          f"{int((subset['label'] == 1).sum()):,} fake)")
    for cond in args.only or CONDITIONS:
        t0 = time.time()
        meta = build_condition(subset, paths.cache_dir, paths.manifests_dir, cond, workers=args.workers,
                               prefix=DEV_PREFIX if args.split == "dev" else PREFIX)
        print(f"  {cond:11s} {meta['description']:30s} {time.time() - t0:5.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
