"""P3.4 score-level fusion.

For each seed, a logistic regression is trained on the members' ASVspoof 2019 LA
**dev** scores (bona fide vs spoof) and applied unchanged to the test sets. Runs of
the same seed are fused, so the result is mean ± std over 3 fused systems, the same
as every other row. The test sets never influence the fusion weights.

Members (default `auto`): the best spectrogram model by mean degraded-dev EER (the
Phase 2 selection metric) among the LFCC models, AASIST, and the SSL model. Every
pair is also reported, to show what each member adds.

Usage:
    python scripts/ensemble/fuse_scores.py                       # auto members
    python scripts/ensemble/fuse_scores.py --members e4a_rawboost e5c_aasist e7a_wavlm_crnn
Writes experiments/fusion/fusion.json and .md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adfd.metrics import eer_from_labels
from adfd.paths import REPO_ROOT, load_paths

EXP = REPO_ROOT / "experiments"
SEEDS = (42, 43, 44)
SPECTRO = ["e4a_rawboost", "e5a_lcnn", "e6a_crnn_seres", "e6b_crnn_seres_asp", "e6c_crnn_light_gru"]
AUTO_REST = ["e5c_aasist", "e7a_wavlm_crnn"]
CONDS = ["clean", "mp3_51k", "mp3_19k", "opus_84k", "opus_10k", "telephone", "white_20db", "white_10db",
         "white_5db", "pink_10db"]
# (column label, score-file stem, how to score)
SETS = [("ASVspoof eval", "scores_asvspoof2019_la_eval.csv", "eval"),
        ("In-the-Wild", "scores_in_the_wild_test.csv", "plain"),
        ("DEEP-VOICE (recording)", "scores_deep_voice_test.csv", "recording"),
        ("degraded eval, mean of 10", None, "robust")]


def degraded_dev_mean(prefix: str) -> float | None:
    vals = []
    for s in SEEDS:
        f = EXP / f"{prefix}_s{s}__robustdev" / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))["sets"]
        vals.append(np.mean([m[f"asvspoof2019_la_robustdev_{c}:test"]["eer_pct"] for c in CONDS]))
    return float(np.mean(vals)) if vals else None


def summarise(per_seed: list[dict]) -> dict:
    keys = per_seed[0].keys()
    return {"n": len(per_seed), "mean": {k: float(np.mean([r[k] for r in per_seed])) for k in keys},
            "std": {k: float(np.std([r[k] for r in per_seed], ddof=1)) if len(per_seed) > 1 else float("nan")
                    for k in keys}}


def scores(run_dir, stem: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / stem, keep_default_na=False)
    return df.set_index("utt_id")


def member_table(runs, stem):
    """Aligned (n, k) score matrix plus the shared metadata of the first member."""
    frames = [scores(r, stem) for r in runs]
    base = frames[0]
    for f in frames[1:]:
        if len(f) != len(base) or not f.index.equals(base.index):
            f = f.reindex(base.index)
            if f["score"].isna().any():
                raise ValueError(f"{stem}: members scored different clips")
    x = np.column_stack([f["score"].to_numpy(float) for f in [base] + frames[1:]])
    return x, base


def set_eer(df: pd.DataFrame, score: np.ndarray, how: str, paths) -> float:
    if how == "recording":
        rec = df.assign(s=score).groupby("recording_id").agg(s=("s", "mean"), label=("label", "first"))
        return 100 * eer_from_labels(rec["s"], rec["label"])[0]
    return 100 * eer_from_labels(score, df["label"])[0]


def eval_system(runs, model, paths) -> dict:
    """EER per test set for one system: `model` maps the (n, k) member scores to one score."""
    out = {}
    for label, stem, how in SETS:
        if how == "robust":
            if not all((r / f"scores_asvspoof2019_la_robust_{c}_test__robust.csv").exists() for r in runs for c in CONDS):
                out[label] = float("nan")
                continue
            vals = []
            for c in CONDS:
                x, df = member_table(runs, f"scores_asvspoof2019_la_robust_{c}_test__robust.csv")
                vals.append(set_eer(df, model(x), "plain", paths))
            out[label] = float(np.mean(vals))
        else:
            x, df = member_table(runs, stem)
            out[label] = set_eer(df, model(x), how, paths)
            if how == "eval":
                from adfd.evaluate import asvspoof_metrics

                m = asvspoof_metrics(df.reset_index().assign(score=model(x)), "eval",
                                     paths.dataset("asvspoof2019_la"))
                out["eval min t-DCF"] = m["min_tdcf"]
    return out


def fuse(prefixes, paths, seeds) -> dict:
    per_seed, weights = [], []
    for s in seeds:
        runs = [paths.runs_dir / f"{p}_s{s}" for p in prefixes]
        x, df = member_table(runs, "scores_asvspoof2019_la_dev.csv")
        y = (df["label"].to_numpy() == 0).astype(int)              # 1 = bona fide (higher score = real)
        lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        lr.fit(x, y)
        weights.append((lr[-1].coef_[0] / lr[0].scale_).round(4).tolist())
        res = eval_system(runs, lr.decision_function, paths)
        res["dev EER"] = 100 * eer_from_labels(lr.decision_function(x), df["label"])[0]
        per_seed.append(res)
    return {"members": list(prefixes), "weights_per_seed": weights, "per_seed": per_seed, **summarise(per_seed)}


def single(prefix, paths, seeds) -> dict:
    return summarise([eval_system([paths.runs_dir / f"{prefix}_s{s}"], lambda x: x[:, 0], paths) for s in seeds])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="*", default=["auto"])
    args = ap.parse_args()
    paths = load_paths()
    note = ""
    if args.members == ["auto"]:
        cands = {p: v for p in SPECTRO if (v := degraded_dev_mean(p)) is not None}
        best = min(cands, key=cands.get)
        members = [best] + AUTO_REST
        note = (f"Spectrogram member chosen by mean degraded-dev EER: **{best}** ({cands[best]:.2f}%; candidates: "
                + ", ".join(f"{p} {v:.2f}" for p, v in sorted(cands.items(), key=lambda kv: kv[1])) + ").")
    else:
        members = args.members
    has = lambda m, s: all((paths.runs_dir / f"{m}_s{s}" / f).exists() for f in  # noqa: E731
                           ("scores_asvspoof2019_la_dev.csv", "scores_asvspoof2019_la_eval.csv"))
    members = [m for m in members if any(has(m, s) for s in SEEDS)]
    seeds = [s for s in SEEDS if all(has(m, s) for m in members)]
    if len(members) < 2 or not seeds:
        print("fewer than 2 members share a scored seed; nothing to fuse")
        return 0

    systems = {m: single(m, paths, seeds) for m in members}
    for k in range(2, len(members) + 1):
        for combo in itertools.combinations(members, k):
            systems[" + ".join(combo)] = fuse(combo, paths, seeds)

    cols = [label for label, _, _ in SETS[:1]] + ["eval min t-DCF"] + [label for label, _, _ in SETS[1:]]
    lines = ["# P3.4 Score-level fusion", "",
             f"Logistic regression on ASVspoof **dev** scores, one fusion per seed (seeds {seeds}); EER in %, "
             "mean ± std when there are several seeds. Test sets are never used to fit the fusion.", ""]
    if note:
        lines += [note, ""]
    lines += ["| System | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in systems.items():
        cells = []
        for c in cols:
            d = 3 if "t-DCF" in c else 2
            m, sd = r["mean"][c], r["std"][c]
            cells.append("–" if np.isnan(m) else f"{m:.{d}f}" if np.isnan(sd) else f"{m:.{d}f} ± {sd:.{d}f}")
        lines.append(f"| {'**' + name + '**' if ' + ' in name else name} | " + " | ".join(cells) + " |")
    full = " + ".join(members)
    best_single = {c: min(systems[m]["mean"][c] for m in members) for c in cols if "t-DCF" not in c}
    gains = {c: best_single[c] - systems[full]["mean"][c] for c in best_single}
    lines += ["", "Full fusion vs best single member (positive = fusion better, EER points): "
              + "; ".join(f"{c} {g:+.2f}" for c, g in gains.items() if not np.isnan(g)) + "."]
    out = EXP / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion.json").write_text(json.dumps({"members": members, "seeds": seeds, "systems": systems, "gain_vs_best_single": gains},
                                                indent=2), encoding="utf-8")
    md = "\n".join(lines) + "\n"
    (out / "fusion.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
