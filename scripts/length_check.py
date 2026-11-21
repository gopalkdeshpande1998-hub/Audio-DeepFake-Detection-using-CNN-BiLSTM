"""P2.2 length shortcut check: does the score depend on clip duration?

Every model sees a fixed 4 s crop; shorter clips are repeated to fill it. If a
model still reacts to duration (for example to the repetition seam), its error
rate or score will change with the clip length. For each headline model (all
seeds) and test set this reports:
  - EER per duration bin (duration of the audio the model saw: whole file, or
    the speech region for silence-trimmed models), and the class mix per bin;
  - Spearman correlation between score and duration within real and within fake
    clips (0 = score independent of length).
The duration-only control classifier is in experiments/shortcuts/silence_control.md.

Usage:
    python scripts/length_check.py
Writes experiments/shortcuts/length_check.json and .md
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adfd.data.silence import silence_file
from adfd.metrics import eer_from_labels
from adfd.paths import REPO_ROOT, load_paths

EXP = REPO_ROOT / "experiments"
BINS = [0, 2, 4, 6, np.inf]
BIN_NAMES = ["< 2 s", "2–4 s", "4–6 s", "≥ 6 s"]
SETS = [("asvspoof2019_la", "eval"), ("in_the_wild", "test")]
MODELS = [("E1 MFCC, untrimmed", "e1_cnn_bilstm_mfcc"), ("E2a LFCC, untrimmed", "e2a_lfcc"),
          ("E3 LFCC, trimmed", "e3_trimmed"), ("E4c LFCC, trimmed, RawBoost + codec", "e4c_rawboost_codec"),
          ("E4d LFCC, trimmed, full augmentation", "e4d_full_aug")]


def seen_duration(paths, name: str, trimmed: bool) -> pd.DataFrame:
    df = pd.read_csv(paths.manifest(name), keep_default_na=False)[["utt_id", "recording_id", "duration_s"]]
    if not trimmed:
        return df.assign(seen_s=df["duration_s"])
    s = np.load(silence_file(paths.cache_dir, name))
    sp = pd.DataFrame({"recording_id": s["recording_ids"],
                       "speech_s": s["total_s"] - s["lead_s"] - s["trail_s"]})
    df = df.merge(sp, on="recording_id", how="left")
    return df.assign(seen_s=df["speech_s"].where(df["speech_s"] > 0, df["duration_s"]))


def main() -> int:
    paths = load_paths()
    durations = {}
    res = {}
    lines = ["# Length shortcut check", "",
             "EER (%) by duration of the audio the model saw (mean over seeds; n = clips real / fake). "
             "ρ = Spearman correlation of score with duration inside each class (|ρ| < 0.1 = negligible).", ""]
    for label, prefix in MODELS:
        runs = sorted(p for p in (paths.runs_dir).glob(f"{prefix}_s*") if (p / "best.pt").exists())
        if not runs:
            continue
        info = json.loads((runs[0] / "run_info.json").read_text(encoding="utf-8"))
        trimmed = bool(info["config"]["audio"].get("trim_silence"))
        lines += [f"## {label} ({len(runs)} seed{'s' if len(runs) > 1 else ''}, {'speech-only' if trimmed else 'whole-file'} duration)", "",
                  "| Set | " + " | ".join(BIN_NAMES) + " | all | ρ real | ρ fake |", "|---|" + "---|" * (len(BIN_NAMES) + 3)]
        res[prefix] = {"trimmed": trimmed, "seeds": [r.name for r in runs], "sets": {}}
        for name, split in SETS:
            key = (name, trimmed)
            if key not in durations:
                durations[key] = seen_duration(paths, name, trimmed)
            per_bin, overall, rho_r, rho_f, counts = [[] for _ in BIN_NAMES], [], [], [], None
            for run in runs:
                f = run / f"scores_{name}_{split}.csv"
                if not f.exists():
                    continue
                sc = pd.read_csv(f, keep_default_na=False).merge(durations[key][["utt_id", "seen_s"]], on="utt_id")
                sc["bin"] = pd.cut(sc["seen_s"], BINS, right=False, labels=BIN_NAMES)
                overall.append(100 * eer_from_labels(sc["score"], sc["label"])[0])
                counts = []
                for i, b in enumerate(BIN_NAMES):
                    g = sc[sc["bin"] == b]
                    nr, nf = int((g["label"] == 0).sum()), int((g["label"] == 1).sum())
                    counts.append((nr, nf))
                    if nr >= 20 and nf >= 20:
                        per_bin[i].append(100 * eer_from_labels(g["score"], g["label"])[0])
                for lab, acc in ((0, rho_r), (1, rho_f)):
                    g = sc[sc["label"] == lab]
                    acc.append(spearmanr(g["score"], g["seen_s"]).statistic)
            if not overall:
                continue
            cells = [f"{np.mean(v):.2f} (n={c[0]:,}/{c[1]:,})" if v else f"– (n={c[0]:,}/{c[1]:,})"
                     for v, c in zip(per_bin, counts)]
            lines.append(f"| {name}:{split} | " + " | ".join(cells) +
                         f" | {np.mean(overall):.2f} | {np.mean(rho_r):+.2f} | {np.mean(rho_f):+.2f} |")
            res[prefix]["sets"][f"{name}:{split}"] = {
                "bins": BIN_NAMES, "eer_pct_per_bin": [float(np.mean(v)) if v else None for v in per_bin],
                "counts_real_fake": counts, "eer_pct": float(np.mean(overall)),
                "spearman_real": float(np.mean(rho_r)), "spearman_fake": float(np.mean(rho_f))}
        lines.append("")
    lines += ["Bins with fewer than 20 clips of either class have no EER (–).", ""]
    out = EXP / "shortcuts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "length_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = "\n".join(lines)
    (out / "length_check.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
