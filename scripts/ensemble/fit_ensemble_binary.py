"""Fit the two-way (REAL / FAKE) merged detector: E7a + VoiceGuard improved + VoiceGuard v2.

Replaces the cautious three-way rule of scripts/ensemble/fit_ensemble.py (user decision, 2026-11-22:
the app must always answer REAL or FAKE, decided by all three models together).

Fusion (adfd.ensemble.EnsembleAnalyzer.fuse)
  1. member log-odds z_i (each member's own calibration, as in fit_ensemble.py), clipped to
     [-CLIP, +CLIP], so no single over-confident member can swamp the others;
  2. weighted sum s = sum_i w_i * clip(z_i), weights W_MIN <= w_i <= W_MAX summing to 1,
     so every member always counts;
  3. decision threshold t on s, and P(fake) = sigmoid(A * (s - t)) (class- and domain-
     balanced Platt slope), so FAKE <=> P(fake) >= 0.5 <=> s >= t; Realness = 1 - P(fake).
No single member decides (hard constraint on w and t together): when one member is at
+-CLIP and the other two at the opposite +-CLIP, the pair wins, with a margin:
    |t| <= CLIP * (1 - 2 * max_i w_i) - MARGIN
(otherwise a heavy member plus a shifted threshold can override two agreeing members).
Weights and t: grid search (weights step 0.05, t step 0.05) maximising the mean balanced
accuracy of the REAL/FAKE decision over the tuning domains, within that constraint.

Tuning data (the only data that sets any number)
  - ASVspoof 2019 LA dev, clean + 10 degraded dev conditions
  - DEEP-VOICE windows of the non-test persons (deep_voice_xd train + dev split:
    biden, linus, musk, obama, margot, ryan); no member was trained on DEEP-VOICE
Held-out reports: ASVspoof eval, DEEP-VOICE test persons (taylor, trump), In-the-Wild
(contaminated for VoiceGuard v2, which trained on most of it: not a clean test).

    python scripts/ensemble/fit_ensemble_binary.py [--no-write-config]
Writes runs/ensemble/ensemble.json (the previous file is kept as ensemble_gated.json)
and experiments/ensemble_binary/ensemble.{md,json}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from fit_ensemble import CONDS, MEMBERS, member_cal, table  # noqa: E402

from adfd.calibration import sigmoid  # noqa: E402
from adfd.metrics import eer_from_labels  # noqa: E402
from adfd.paths import REPO_ROOT, load_paths  # noqa: E402

CLIP = 4.0
W_MIN, W_MAX, STEP = 0.2, 0.45, 0.05
MARGIN = 0.2      # the agreeing pair must win by at least this much in s
OUT = REPO_ROOT / "experiments" / "ensemble_binary"


def dv_split(split_names: set[str]):
    z, y, meta = table("deep_voice_test")
    m = pd.read_csv(load_paths().manifest("deep_voice_xd"), keep_default_na=False).set_index("utt_id")
    keep = m.loc[meta.index, "split"].isin(split_names).to_numpy()
    return z[:, keep], y[keep], meta[keep]


def weight_grid():
    vals = np.round(np.arange(W_MIN, W_MAX + 1e-9, STEP), 4)
    for a, b in itertools.product(vals, vals):
        c = round(1 - a - b, 4)
        if W_MIN - 1e-9 <= c <= W_MAX + 1e-9:
            yield np.array([a, b, c])


def fused_s(z, w):
    return (np.asarray(w)[:, None] * np.clip(z, -CLIP, CLIP)).sum(0)


def binary_stats(p, y):
    fake = p >= 0.5
    return {"n": int(len(y)), "accuracy_pct": 100 * float((fake == (y == 1)).mean()),
            "balanced_accuracy_pct": 50 * float(fake[y == 1].mean() + (~fake)[y == 0].mean()),
            "real_called_fake_pct": 100 * float(fake[y == 0].mean()),
            "fake_called_real_pct": 100 * float((~fake)[y == 1].mean()),
            "eer_pct": 100 * float(eer_from_labels(-p, y)[0])}


def per_recording(p, y, meta):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    r = pd.DataFrame({"r": meta["recording_id"].to_numpy(), "z": z, "y": y}).groupby("r").agg(
        z=("z", "mean"), y=("y", "first"))
    return binary_stats(sigmoid(r["z"].to_numpy()), r["y"].to_numpy())


def main() -> None:
    global W_MIN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write-config", action="store_true")
    ap.add_argument("--w-min", type=float, default=0.2, help="smallest weight a member may get")
    args = ap.parse_args()
    W_MIN = args.w_min

    asv = [table("asvspoof2019_la_dev")] + [table(f"asvspoof2019_la_robustdev_{c}_test") for c in CONDS]
    tune = {"ASVspoof dev + degraded dev": (np.concatenate([t[0] for t in asv], 1),
                                            np.concatenate([t[1] for t in asv]), None),
            "DEEP-VOICE tuning persons": dv_split({"train", "dev"})}
    tests = {"ASVspoof eval": table("asvspoof2019_la_eval"),
             "DEEP-VOICE test persons": dv_split({"test"}),
             "In-the-Wild (VG v2 contaminated)": table("in_the_wild_test")}

    # --- weights and decision threshold, chosen together -----------------------------------
    # objective: mean over tuning domains of the balanced accuracy of the REAL/FAKE decision
    # (what the app shows), so neither a big lab dataset nor one class dominates.
    thr_grid = np.linspace(-CLIP, CLIP, 161)
    best = None
    for w in weight_grid():
        bacc = np.zeros_like(thr_grid)
        feasible = np.abs(thr_grid) <= CLIP * (1 - 2 * w.max()) - MARGIN + 1e-9
        if not feasible.any():
            continue
        for z, y, _ in tune.values():
            s = fused_s(z, w)
            fake = s[None, :] >= thr_grid[:, None]
            bacc += 0.5 * (fake[:, y == 1].mean(1) + (~fake[:, y == 0]).mean(1)) / len(tune)
        bacc[~feasible] = -1
        i = int(np.argmax(bacc))
        if best is None or bacc[i] > best[0]:
            best = (float(bacc[i]), w, float(thr_grid[i]))
    _, w, t = best

    # --- calibration: class- and domain-balanced Platt slope, centred on the threshold -----
    s_all, y_all, wt_all = [], [], []
    for z, y, _ in tune.values():
        s = fused_s(z, w)
        for c in (0, 1):
            k = y == c
            s_all.append(s[k]), y_all.append(y[k]), wt_all.append(np.full(k.sum(), 1.0 / k.sum()))
    s_all, y_all, wt_all = map(np.concatenate, (s_all, y_all, wt_all))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit((s_all - t)[:, None], y_all, sample_weight=wt_all)
    A = float(lr.coef_[0, 0])
    B = -A * t                                   # P(fake) = 0.5 exactly at the chosen threshold

    def prob(z, ww=w, a=A, b=B):
        return sigmoid(a * fused_s(z, ww) + b)

    # --- report: chosen fusion vs E7a alone vs equal weights, all at the 0.5 decision -----
    systems = {"merged (chosen)": prob}
    e7a = lambda z: sigmoid(z[0])                                           # noqa: E731
    systems["E7a alone (for comparison)"] = e7a
    eq = np.full(3, 1 / 3)
    s_eq = np.concatenate([fused_s(z, eq) for z, _, _ in tune.values()])
    systems["equal weights (for comparison)"] = lambda z: sigmoid(fused_s(z, eq) - np.median(s_eq))
    report = {"weights": w.tolist(), "clip": CLIP, "margin": MARGIN, "A": A, "B": B, "w_range": [W_MIN, W_MAX],
              "threshold_s": t, "tuning_mean_balanced_acc_pct": 100 * best[0], "systems": {}}
    for sname, f in systems.items():
        d = {}
        for name, (z, y, meta) in {**tune, **tests}.items():
            d[name] = binary_stats(f(z), y)
            if name.startswith("DEEP"):
                d[name + " (per recording)"] = per_recording(f(z), y, meta)
        report["systems"][sname] = d
    report["single_members_eer_pct"] = {
        run: {name: 100 * eer_from_labels(-z[i], y)[0] for name, (z, y, _) in {**tune, **tests}.items()}
        for i, (run, _) in enumerate(MEMBERS)}

    cfg = {"fusion": "clipped_weighted_logodds", "decision": "binary", "threshold": 0.5, "clip": CLIP,
           "A": A, "B": B, "bias": 0.0,
           "members": [{"run": r, "label": lab, "weight": float(wi), "a": member_cal(r)[0], "b": member_cal(r)[1]}
                       for (r, lab), wi in zip(MEMBERS, w)],
           "fitted_on": "ASVspoof 2019 LA dev (clean + 10 degraded) and DEEP-VOICE non-test persons",
           "weight_constraints": [W_MIN, W_MAX], "threshold_s": t, "margin": MARGIN,
           "no_single_member_rule": "|t| <= clip * (1 - 2 * max w) - margin"}
    if not args.no_write_config:
        d = load_paths().runs_dir / "ensemble"
        if (d / "ensemble.json").exists() and not (d / "ensemble_gated.json").exists():
            shutil.copy(d / "ensemble.json", d / "ensemble_gated.json")
        (d / "ensemble.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ensemble.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    md = markdown(report)
    (OUT / "ensemble.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))


def markdown(r):
    sets = list(next(iter(r["systems"].values())))
    L = ["# Merged detector, two-way REAL / FAKE decision", "",
         "Produced by `scripts/ensemble/fit_ensemble_binary.py`. Weights and calibration come from the tuning sets only "
         "(ASVspoof dev + degraded dev, DEEP-VOICE non-test persons). Decision: FAKE if merged P(fake) >= 0.5.", "",
         f"Weights (E7a, VoiceGuard improved, VoiceGuard v2): {', '.join(f'{v:.2f}' for v in r['weights'])} "
         f"(each constrained to {r['w_range'][0]}–{r['w_range'][1]}); member log-odds clipped to ±{r['clip']}; "
         f"calibration P = sigmoid({r['A']:.3f}·s + {r['B']:.3f}), i.e. FAKE when s ≥ {r['threshold_s']:.2f}. "
         f"Constraint: two members that agree at full confidence always beat the third (margin {r['margin']}).", "",
         "**In-the-Wild is not a clean test**: VoiceGuard v2 was trained on most of it.", "",
         "## Accuracy at the 0.5 decision (%)", "",
         "| System | Set | n | accuracy | balanced acc. | real → FAKE | fake → REAL | EER |",
         "|---|---|---|---|---|---|---|---|"]
    for s, d in r["systems"].items():
        for name in sets:
            t = d[name]
            L.append(f"| {s} | {name} | {t['n']:,} | {t['accuracy_pct']:.1f} | {t['balanced_accuracy_pct']:.1f} | "
                     f"{t['real_called_fake_pct']:.1f} | {t['fake_called_real_pct']:.1f} | {t['eer_pct']:.2f} |")
    L += ["", "## EER (%) of each member alone", "", "| Member | " + " | ".join(r["single_members_eer_pct"][
        next(iter(r["single_members_eer_pct"]))]) + " |", "|---" * (1 + len(next(iter(
            r["single_members_eer_pct"].values())))) + "|"]
    for run, d in r["single_members_eer_pct"].items():
        L.append(f"| {run} | " + " | ".join(f"{v:.2f}" for v in d.values()) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
