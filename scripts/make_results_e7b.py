"""Results of the E7b retrain: every model on every test set, and old vs new merged detector.

Reads per-clip scores from runs/ (all models scored on exactly the same clips) and the
ensemble reports written by scripts/fit_ensemble.py:
    experiments/ensemble_e7a_xd/   old ensemble (E7a + VoiceGuard), new test sets
    experiments/ensemble_e7b_xd/   new ensemble (E7b + VoiceGuard)
Writes experiments/e7b/results.json and results.md.

    python scripts/make_results_e7b.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from adfd.metrics import auc, bootstrap_eer, eer_from_labels
from adfd.paths import REPO_ROOT, load_paths

MODELS = [("e7a_wavlm_crnn_s42", "E7a: WavLM + CNN-BiLSTM-Att (before)"),
          ("e7b_wavlm_crnn_itw_s42", "E7b: retrained, ASVspoof + ITW train speakers (after)"),
          ("vg_improved", "VoiceGuard improved"),
          ("vg_v2", "VoiceGuard v2")]
SETS = [("ASVspoof 2019 LA eval", "asvspoof2019_la_eval", "clean for all"),
        ("In-the-Wild, held-out test speakers", "in_the_wild_xd_test",
         "VoiceGuard v2 trained on most of In-the-Wild: its number here is not a test result"),
        ("DEEP-VOICE (per window)", "deep_voice_test", "clean for all"),
        ("DEEP-VOICE (per recording)", "deep_voice_test", "clean for all; 8 real / 56 fake recordings"),
        ("Speech_Dataset", "speech_dataset_test", "source not recorded; VoiceGuard improved may have seen it")]
OUT = REPO_ROOT / "experiments" / "e7b"


def scores(run, stem):
    d = load_paths().runs_dir / run
    for f in [d / f"scores_{stem}.csv", *sorted(d.glob(f"scores_{stem}__*.csv"))]:
        if f.exists():
            return pd.read_csv(f, keep_default_na=False)
    return None


def main() -> int:
    res = {"models": dict(MODELS), "sets": [s[0] for s in SETS], "notes": {s[0]: s[2] for s in SETS}, "eer": {},
           "auc": {}, "ci": {}, "n": {}}
    for run, _ in MODELS:
        res["eer"][run], res["auc"][run], res["ci"][run] = {}, {}, {}
        for name, stem, _ in SETS:
            df = scores(run, stem)
            if df is None:
                continue
            s, y = df["score"].to_numpy(float), df["label"].to_numpy(int)
            groups = None
            if "recording" in name:
                g = df.groupby("recording_id").agg(score=("score", "mean"), label=("label", "first"))
                s, y = g["score"].to_numpy(), g["label"].to_numpy()
            elif stem == "deep_voice_test":
                groups = df["recording_id"].to_numpy()
            res["eer"][run][name] = 100 * eer_from_labels(s, y)[0]
            res["auc"][run][name] = float(auc(s, y))
            if len(y) > 20:
                ci = bootstrap_eer(s, y, groups, n_boot=200)
                res["ci"][run][name] = [ci["lo_pct"], ci["hi_pct"]]
            res["n"][name] = {"n": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum())}
    hist = load_paths().runs_dir / "e7b_wavlm_crnn_itw_s42" / "history.json"
    if hist.exists():
        res["training"] = [{"epoch": h["epoch"], "asvspoof_dev_eer": 100 * h["dev_eer"],
                            "itw_dev_eer": 100 * next(iter(h.get("extra_dev_eer", {}).values()), float("nan")),
                            "loss": h["train_loss"], "minutes": (h["train_s"] + h["eval_s"]) / 60}
                           for h in json.loads(hist.read_text(encoding="utf-8"))]
    res["ensembles"] = {}
    for tag in ("e7a_xd", "e7b_xd"):
        f = REPO_ROOT / "experiments" / f"ensemble_{tag}" / "ensemble.json"
        if f.exists():
            e = json.loads(f.read_text(encoding="utf-8"))
            res["ensembles"][tag] = {"members": e["members"], "config": e["decision"]["config"],
                                     "sets": e["decision"]["sets"]["+ agreement gate and veto (used)"],
                                     "fusion_eer": e["fusions"]["lr"]["eer_pct"], "weights": e["fusions"]["lr"]["weights"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    L = ["# E7b retrain: results", "", "EER in % (lower is better), AUC in brackets. All models scored on the same clips.", "",
         "| Model | " + " | ".join(res["sets"]) + " |", "|---" * (len(res["sets"]) + 1) + "|"]
    for run, label in MODELS:
        cells = [f"{res['eer'][run][s]:.2f} ({res['auc'][run][s]:.3f})" if s in res["eer"][run] else "–"
                 for s in res["sets"]]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "Notes: " + " ".join(f"**{k}**: {v}." for k, v in res["notes"].items())]
    if res["ensembles"]:
        L += ["", "## Merged detector, old (E7a) vs new (E7b)", "",
              "| Ensemble | Set | decided % | real → Fake % | fakes caught % | fake → Real % |", "|---|---|---|---|---|---|"]
        for tag, e in res["ensembles"].items():
            for s, t in e["sets"].items():
                L.append(f"| {tag} | {s} | {t['decided_pct']:.1f} | {t['real_called_fake_pct']:.2f} | "
                         f"{t['fake_detected_pct']:.1f} | {t['fake_called_real_pct']:.2f} |")
    if res.get("training"):
        L += ["", "## Training curve", "", "| epoch | ASVspoof dev EER % | ITW dev EER % | loss | minutes |",
              "|---|---|---|---|---|"]
        L += [f"| {h['epoch']} | {h['asvspoof_dev_eer']:.2f} | {h['itw_dev_eer']:.2f} | {h['loss']:.4f} | "
              f"{h['minutes']:.0f} |" for h in res["training"]]
    md = "\n".join(L) + "\n"
    (OUT / "results.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
