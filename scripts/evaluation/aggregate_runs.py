"""Combine several runs (e.g. seeds) into mean +/- std tables.

Usage:
    python scripts/evaluation/aggregate_runs.py --name e1_cnn_bilstm_mfcc --runs e1_cnn_bilstm_mfcc_s42 e1_cnn_bilstm_mfcc_s43 ...
    python scripts/evaluation/aggregate_runs.py --name e1_cnn_bilstm_mfcc --glob "e1_cnn_bilstm_mfcc_s*"

Reads experiments/<run>/metrics.json and writes experiments/<name>/aggregate.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from adfd.paths import REPO_ROOT

EXP = REPO_ROOT / "experiments"


def ms(values) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    return {"mean": round(float(v.mean()), 3), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 3),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3), "n": int(len(v))}


def fmt(s: dict, digits=2) -> str:
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--glob")
    args = ap.parse_args()

    runs = args.runs or sorted(p.name for p in EXP.glob(args.glob or "") if (p / "metrics.json").exists())
    runs = [r for r in runs if r != args.name]
    if not runs:
        print("no runs found")
        return 1
    metrics = [json.loads((EXP / r / "metrics.json").read_text(encoding="utf-8")) for r in runs]

    sets = list(metrics[0]["sets"])
    agg = {"runs": runs, "sets": {}}
    for s in sets:
        per = [m["sets"][s] for m in metrics]
        entry = {"eer_pct": ms(p["eer_pct"] for p in per), "auc": ms(p["auc"] for p in per)}
        if "min_tdcf" in per[0]:
            entry["min_tdcf"] = ms(p["min_tdcf"] for p in per)
        if "per_attack_eer_pct" in per[0]:
            entry["per_attack_eer_pct"] = {a: ms(p["per_attack_eer_pct"][a] for p in per)
                                           for a in per[0]["per_attack_eer_pct"]}
        if "per_recording" in per[0]:
            entry["per_recording_eer_pct"] = ms(p["per_recording"]["eer_pct"] for p in per)
        agg["sets"][s] = entry
    agg["best_epoch"] = [m["best_epoch"] for m in metrics]

    out = EXP / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    lines = [f"# {args.name}: mean ± std over {len(runs)} runs", "",
             "Runs: " + ", ".join(f"`{r}`" for r in runs), "",
             "| Test set | EER % | min t-DCF | AUC |", "|---|---|---|---|"]
    for s, e in agg["sets"].items():
        tdcf = fmt(e["min_tdcf"], 4) if "min_tdcf" in e else ""
        lines.append(f"| {s} | {fmt(e['eer_pct'])} | {tdcf} | {fmt(e['auc'], 4)} |")
        if "per_recording_eer_pct" in e:
            lines.append(f"| {s} (per recording) | {fmt(e['per_recording_eer_pct'])} |  |  |")
    for s, e in agg["sets"].items():
        if "per_attack_eer_pct" in e and s.endswith(":eval"):
            att = e["per_attack_eer_pct"]
            lines += ["", f"## {s}: EER per attack (%, mean ± std)", "",
                      "| " + " | ".join(att) + " |", "|" + "---|" * len(att),
                      "| " + " | ".join(f"{v['mean']:.1f} ± {v['std']:.1f}" for v in att.values()) + " |"]
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
