"""Manifest for PartialSpoof v1.2 (temporal training and evaluation).

    python scripts/data/build_partialspoof.py

Writes data/manifests/partialspoof.csv: one row per utterance with
  utt_id, path, set (dev | eval), split (train | val | test), speaker, kind (real | partial | fake),
  duration_s, n_frames (0.16-s label frames), fake_frac, label (1 = any fake frame).
Split roles: PartialSpoof dev -> train / val by SPEAKER (VAL_SPEAKERS speakers held out, chosen with a fixed
seed), PartialSpoof eval -> test only. Eval rows are written only when its audio is present.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from adfd.data.partialspoof import load_seglab
from adfd.paths import REPO_ROOT, load_paths

_P = load_paths()
ROOT = _P.dataset("partialspoof")                          # configs/paths.yaml (or ADFD_PATHS)
AUDIO = {"dev": _P.dataset("partialspoof_dev_audio"), "eval": ROOT / "eval" / "con_wav"}
VAL_SPEAKERS = 4
SEED = 2026


def rows_for(name: str) -> list[dict]:
    proto = ROOT / "protocols" / "PartialSpoof_LA_cm_protocols" / f"PartialSpoof.LA.cm.{name}.trl.txt"
    lab = load_seglab(ROOT / "segment_labels" / f"{name}_seglab_0.16.npy")
    out = []
    for line in open(proto, encoding="utf-8"):
        spk, utt, _, system, key = line.split()
        path = AUDIO[name] / f"{utt}.wav"
        y = lab[utt]
        ff = float(y.mean())
        out.append({"utt_id": utt, "path": str(path), "set": name, "speaker": spk, "system": system,
                    "kind": "real" if ff == 0 else "fake" if ff == 1 else "partial",
                    "duration_s": sf.info(str(path)).frames / 16000, "n_frames": len(y),
                    "fake_frac": round(ff, 4), "label": int(ff > 0)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "manifests" / "partialspoof.csv"))
    args = ap.parse_args()
    dev = pd.DataFrame(rows_for("dev"))
    spk = sorted(dev["speaker"].unique())
    val = sorted(random.Random(SEED).sample(spk, VAL_SPEAKERS))
    dev["split"] = np.where(dev["speaker"].isin(val), "val", "train")
    parts = [dev]
    if (AUDIO["eval"] / "CON_E_0000000.wav").exists() or any(AUDIO["eval"].glob("*.wav")):
        ev = pd.DataFrame(rows_for("eval"))
        ev["split"] = "test"
        parts.append(ev)
    else:
        print("eval audio not found yet: manifest has dev only")
    df = pd.concat(parts, ignore_index=True)
    assert not set(df.loc[df.split == "train", "speaker"]) & set(df.loc[df.split != "train", "speaker"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = {s: {"utts": int(len(g)), "speakers": int(g.speaker.nunique()), "hours": round(g.duration_s.sum() / 3600, 2),
                   "kinds": g.kind.value_counts().to_dict(), "fake_frame_share": round(float(
                       (g.fake_frac * g.n_frames).sum() / g.n_frames.sum()), 3)}
               for s, g in df.groupby("split")}
    summary["val_speakers"] = val
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
