"""Decode each dataset once into a 16 kHz mono int16 cache (see adfd.data.audio_cache).

Usage:
    python scripts/cache_audio.py                    # all datasets
    python scripts/cache_audio.py --only deep_voice

Needs the manifests (scripts/build_manifests.py). Takes about 20 GB on the
cache drive in total; skips a dataset whose cache matches its manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from adfd.data.audio_cache import build_cache, cache_files
from adfd.data.manifests import file_sha256
from adfd.paths import DATASET_NAMES, load_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=DATASET_NAMES, action="append")
    ap.add_argument("--workers", type=int, default=8, help="decoding processes")
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache is up to date")
    args = ap.parse_args()

    paths = load_paths()
    for name in args.only or DATASET_NAMES:
        manifest_path = paths.manifest(name)
        sha = file_sha256(manifest_path)
        _, _, meta_file = cache_files(paths.cache_dir, name)
        if meta_file.exists() and not args.force:
            if json.loads(meta_file.read_text(encoding="utf-8")).get("manifest_sha256") == sha:
                print(f"[{name}] cache is up to date, skipping")
                continue

        df = pd.read_csv(manifest_path, keep_default_na=False)
        t0 = time.time()
        last = [0.0]

        def progress(done, total):
            if time.time() - last[0] > 30 or done == total:
                last[0] = time.time()
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[{name}] {done:,}/{total:,} recordings ({rate:.0f}/s)", flush=True)

        print(f"[{name}] caching {df['recording_id'].nunique():,} recordings ...", flush=True)
        meta = build_cache(df, paths.dataset(name), paths.cache_dir, name, sha, args.workers, progress)
        print(f"[{name}] done in {time.time() - t0:.0f}s: {meta['hours']} h, {meta['size_gb']} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
