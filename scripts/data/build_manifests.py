"""Build and validate the dataset manifests.

Usage:
    python scripts/data/build_manifests.py                 # all three datasets
    python scripts/data/build_manifests.py --only deep_voice
    python scripts/data/build_manifests.py --check          # validate existing manifests only

Writes data/manifests/<dataset>.csv plus summary.json (counts, hours and a
SHA-256 per manifest, which training runs record for reproducibility).
Exits non-zero if any manifest fails validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from adfd.data.manifests import BUILDERS, file_sha256, summarise, validate_manifest
from adfd.paths import DATASET_NAMES, load_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=DATASET_NAMES, action="append", help="build only these datasets")
    ap.add_argument("--check", action="store_true", help="validate existing manifests without rebuilding")
    ap.add_argument("--workers", type=int, default=16, help="threads for reading audio headers")
    args = ap.parse_args()

    paths = load_paths()
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or list(DATASET_NAMES)
    summary_file = paths.manifests_dir / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}

    failed = False
    for name in names:
        out = paths.manifest(name)
        if args.check:
            if not out.exists():
                print(f"[{name}] no manifest at {out}")
                failed = True
                continue
            df = pd.read_csv(out, keep_default_na=False)
        else:
            root = paths.dataset(name)
            if not root.exists():
                print(f"[{name}] dataset folder not found: {root}")
                failed = True
                continue
            t0 = time.time()
            print(f"[{name}] reading {root} ...", flush=True)
            df = BUILDERS[name](root, workers=args.workers)
            df.to_csv(out, index=False)
            print(f"[{name}] {len(df):,} rows written to {out} in {time.time() - t0:.0f}s")

        problems = validate_manifest(df, name)
        if problems:
            failed = True
            print(f"[{name}] VALIDATION FAILED:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"[{name}] validation passed")

        summary[name] = {
            "dataset_root": str(paths.dataset(name)),
            "manifest": out.name,
            "sha256": file_sha256(out),
            "valid": not problems,
            "splits": summarise(df),
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        for split, s in summary[name]["splits"].items():
            print(f"    {split:5s}: {s['rows']:>7,} rows  real {s['bonafide']:>6,}  fake {s['spoof']:>6,}"
                  f"  {s['hours']:>6.2f} h  sample rates {s['sample_rates']}")

    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
