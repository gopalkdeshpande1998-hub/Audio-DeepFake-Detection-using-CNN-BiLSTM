"""P2.2 shortcut control: how well do silence and duration alone predict the label?

The classifier never sees the audio content, only four numbers per clip:
leading silence, trailing silence, total duration and speech duration.
It is trained on ASVspoof train (logistic regression and gradient boosting),
its sign is fixed on train, and it is scored on dev, eval and In-the-Wild.
A low EER means the dataset leaks the label through silence or length, so a
detector can score well without learning anything about synthetic speech.

Usage:
    python scripts/shortcut_control.py
Writes experiments/shortcuts/silence_control.json and .md
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adfd.data.silence import silence_file
from adfd.metrics import eer_from_labels
from adfd.paths import REPO_ROOT, load_paths

FEATURES = ["lead_s", "trail_s", "total_s", "speech_s"]


def load(paths, name: str) -> pd.DataFrame:
    df = pd.read_csv(paths.manifest(name), keep_default_na=False)
    s = np.load(silence_file(paths.cache_dir, name))
    sil = pd.DataFrame({"recording_id": s["recording_ids"], "lead_s": s["lead_s"],
                        "trail_s": s["trail_s"], "total_s": s["total_s"]})
    sil["speech_s"] = sil["total_s"] - sil["lead_s"] - sil["trail_s"]
    return df.merge(sil, on="recording_id", how="left")


def pct(x):
    return round(100 * x, 2)


def main() -> int:
    paths = load_paths()
    asv = load(paths, "asvspoof2019_la")
    itw = load(paths, "in_the_wild")
    sets = {
        "asvspoof2019_la:dev": asv[asv["split"] == "dev"],
        "asvspoof2019_la:eval": asv[asv["split"] == "eval"],
        "in_the_wild:test": itw,
    }
    train = asv[asv["split"] == "train"]
    y = (train["label"] == 0).astype(int)  # predict bona fide, so higher score = bona fide

    results = {"features": FEATURES, "single_feature_eer_pct": {}, "models": {}}
    # Single features, with the direction fixed on train.
    for f in FEATURES:
        sign = 1.0 if eer_from_labels(train[f], train["label"])[0] <= eer_from_labels(-train[f], train["label"])[0] else -1.0
        results["single_feature_eer_pct"][f] = {
            "direction": "longer = more likely real" if sign > 0 else "longer = more likely fake",
            **{k: pct(eer_from_labels(sign * d[f], d["label"])[0]) for k, d in sets.items()},
        }
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=0),
    }
    for name, model in models.items():
        model.fit(train[FEATURES], y)
        results["models"][name] = {k: pct(eer_from_labels(model.predict_proba(d[FEATURES])[:, 1], d["label"])[0])
                                   for k, d in sets.items()}

    out = REPO_ROOT / "experiments" / "shortcuts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "silence_control.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    cols = list(sets)
    lines = ["# Silence / duration shortcut control", "",
             "EER (%) of classifiers that see only silence and duration, never the audio content.",
             "Trained on ASVspoof 2019 LA train. 50% = no information.", "",
             "| Input | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for f, r in results["single_feature_eer_pct"].items():
        lines.append(f"| {f} only ({r['direction']}) | " + " | ".join(f"{r[c]:.2f}" for c in cols) + " |")
    for m, r in results["models"].items():
        lines.append(f"| all 4, {m.replace('_', ' ')} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + " |")
    md = "\n".join(lines) + "\n"
    (out / "silence_control.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
