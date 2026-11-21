"""P2.2 loudness shortcut check.

1. Control: how well do loudness statistics alone (speech RMS level, peak level,
   crest factor) predict the label? Same protocol as shortcut_control.py:
   trained on ASVspoof train only, scored on dev, eval and In-the-Wild.
2. Model sensitivity: compares each headline model's EER on normal test audio
   with its EER when every test crop is loudness-normalised (evaluate.py
   --normalize rms, tag "rmsnorm"). A large change means the model uses level.

Usage:
    python scripts/loudness_check.py --compute      # loudness stats (CPU, once)
    python scripts/loudness_check.py                # tables
Writes experiments/shortcuts/loudness_check.json and .md
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adfd.data.loudness import compute_loudness, loudness_file
from adfd.metrics import eer_from_labels
from adfd.paths import REPO_ROOT, load_paths

FEATURES = ["rms_db", "peak_db", "crest_db"]
EXP = REPO_ROOT / "experiments"
SETS = ["asvspoof2019_la:dev", "asvspoof2019_la:eval", "in_the_wild:test"]
MODELS = [("E1 MFCC, untrimmed", "e1_cnn_bilstm_mfcc"), ("E2a LFCC, untrimmed", "e2a_lfcc"),
          ("E4c LFCC, trimmed, RawBoost + codec", "e4c_rawboost_codec"),
          ("E4d LFCC, trimmed, full augmentation", "e4d_full_aug")]


def load(paths, name: str) -> pd.DataFrame:
    df = pd.read_csv(paths.manifest(name), keep_default_na=False)
    d = np.load(loudness_file(paths.cache_dir, name))
    lo = pd.DataFrame({"recording_id": d["recording_ids"], **{f: d[f] for f in FEATURES}})
    return df.merge(lo, on="recording_id", how="left")


def pct(x):
    return round(100 * float(x), 2)


def control(paths) -> dict:
    asv = load(paths, "asvspoof2019_la")
    itw = load(paths, "in_the_wild")
    sets = {"asvspoof2019_la:dev": asv[asv["split"] == "dev"], "asvspoof2019_la:eval": asv[asv["split"] == "eval"],
            "in_the_wild:test": itw}
    train = asv[asv["split"] == "train"]
    y = (train["label"] == 0).astype(int)
    res = {"features": FEATURES, "single_feature_eer_pct": {}, "models": {}, "medians_db": {}}
    for f in FEATURES:
        sign = 1.0 if eer_from_labels(train[f], train["label"])[0] <= eer_from_labels(-train[f], train["label"])[0] else -1.0
        res["single_feature_eer_pct"][f] = {
            "direction": "higher = more likely real" if sign > 0 else "higher = more likely fake",
            **{k: pct(eer_from_labels(sign * d[f], d["label"])[0]) for k, d in sets.items()}}
    for k, d in {"asvspoof2019_la:train": train, **sets}.items():
        res["medians_db"][k] = {f"{f}_{lab}": round(float(d[d["label"] == v][f].median()), 1)
                                for f in FEATURES for lab, v in (("real", 0), ("fake", 1))}
    models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
              "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=0)}
    for name, m in models.items():
        m.fit(train[FEATURES], y)
        res["models"][name] = {k: pct(eer_from_labels(m.predict_proba(d[FEATURES])[:, 1], d["label"])[0])
                               for k, d in sets.items()}
    return res


def seeds_metric(prefix: str, tag: str, s: str) -> list[float]:
    vals = []
    for seed in (42, 43, 44):
        f = EXP / (f"{prefix}_s{seed}__{tag}" if tag else f"{prefix}_s{seed}") / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))["sets"]
            if s in m:
                vals.append(m[s]["eer_pct"])
    return vals


def fmt(v: list[float]) -> str:
    if not v:
        return "–"
    return f"{v[0]:.2f}" if len(v) == 1 else f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f} (n={len(v)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute", action="store_true", help="compute loudness statistics first")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    paths = load_paths()
    if args.compute:
        for name in ("asvspoof2019_la", "in_the_wild"):
            print(f"[{name}] -> {compute_loudness(paths.cache_dir, name, workers=args.workers).name}", flush=True)

    res = control(paths)
    lines = ["# Loudness shortcut check", "",
             "## Control: classifiers that see only loudness", "",
             "EER (%). Speech RMS level (silence excluded), peak level, crest factor (peak − RMS). "
             "Trained on ASVspoof 2019 LA train. 50% = no information.", "",
             "| Input | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for f, r in res["single_feature_eer_pct"].items():
        lines.append(f"| {f} only ({r['direction']}) | " + " | ".join(f"{r[c]:.2f}" for c in SETS) + " |")
    for m, r in res["models"].items():
        lines.append(f"| all 3, {m.replace('_', ' ')} | " + " | ".join(f"{r[c]:.2f}" for c in SETS) + " |")
    lines += ["", "Median levels (dBFS; real / fake):", "",
              "| Set | speech RMS | peak | crest |", "|---|---|---|---|"]
    for k, md in res["medians_db"].items():
        lines.append(f"| {k} | {md['rms_db_real']} / {md['rms_db_fake']} | {md['peak_db_real']} / {md['peak_db_fake']} | "
                     f"{md['crest_db_real']} / {md['crest_db_fake']} |")

    lines += ["", "## Model sensitivity: EER (%) on normal vs RMS-normalised test audio", "",
              f"Every test crop scaled to the same RMS level (−26 dBFS, gain limited to ±20 dB). "
              "Mean ± std over seeds.", "",
              "| Model | set | normal | RMS-normalised | change |", "|---|---|---|---|---|"]
    sens = {}
    for label, prefix in MODELS:
        for s in SETS:
            a, b = seeds_metric(prefix, "", s), seeds_metric(prefix, "rmsnorm", s)
            sens[f"{prefix}|{s}"] = {"normal": a, "rmsnorm": b}
            if b:
                d = f"{np.mean(b) - np.mean(a):+.2f}" if a else "–"
                lines.append(f"| {label} | {s} | {fmt(a)} | {fmt(b)} | {d} |")
    res["model_sensitivity"] = sens

    out = EXP / "shortcuts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "loudness_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = "\n".join(lines) + "\n"
    (out / "loudness_check.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
