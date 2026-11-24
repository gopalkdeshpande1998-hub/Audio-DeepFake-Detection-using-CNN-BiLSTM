"""Fit and evaluate the merged detector (adfd.ensemble): E7a + VoiceGuard improved + VoiceGuard v2.

Tuning data (the only data that sets any number): ASVspoof 2019 LA dev, clean plus the
10 degraded dev conditions. Test sets are scored once with the frozen configuration.

Member calibration
  E7a        its own Platt scaling (runs/e7a_wavlm_crnn_s42/reliability.json, dev data)
  VoiceGuard native sigmoid of the logit. Both were trained with class-balanced loss
             on their own data; ASVspoof dev cannot calibrate them (they are near
             chance on it), and the test sets must not.
Fusion candidates (all compared on the tuning data, then on the tests)
  e7a only | equal-weight mean of member log-odds | logistic regression on tuning data
Decision rule (adfd.ensemble.EnsembleAnalyzer.decide): fused thresholds t_real / t_fake
chosen so at most `--target` % of real tuning clips are called Fake and of fake tuning
clips are called Real, plus the agreement gate (min_agree members vote Fake) and the
veto (no member votes Fake with P >= veto_p for a Real verdict).

    python scripts/ensemble/fit_ensemble.py [--fusion equal|lr|e7a] [--target 2]
Writes runs/ensemble/ensemble.json and experiments/ensemble/ensemble.{md,json}
"""

from __future__ import annotations

import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from adfd.calibration import sigmoid
from adfd.metrics import auc, eer_from_labels
from adfd.models.voiceguard import VOICEGUARD_MODELS
from adfd.paths import REPO_ROOT, load_paths

MEMBERS = [("e7a_wavlm_crnn_s42", "WavLM + CNN-BiLSTM-Att (E7a)"),
           ("vg_improved", "VoiceGuard improved"),
           ("vg_v2", "VoiceGuard v2")]
CONDS = ["clean", "mp3_51k", "mp3_19k", "opus_84k", "opus_10k", "telephone", "white_20db", "white_10db",
         "white_5db", "pink_10db"]
TESTS = [("ASVspoof eval", "asvspoof2019_la_eval"), ("In-the-Wild", "in_the_wild_test"),
         ("DEEP-VOICE (windows)", "deep_voice_test")]
# --tests xd: In-the-Wild restricted to the held-out TEST speakers (scripts/data/build_extra_data.py),
# plus the extra Speech_Dataset test set. Needed once a member has trained on In-the-Wild (E7b).
TESTS_XD = [("ASVspoof eval", "asvspoof2019_la_eval"), ("In-the-Wild (test speakers)", "in_the_wild_xd_test"),
            ("DEEP-VOICE (windows)", "deep_voice_test"), ("Speech_Dataset", "speech_dataset_test")]
OUT = REPO_ROOT / "experiments" / "ensemble"


def read(run, stem):
    d = load_paths().runs_dir / run
    for f in (d / f"scores_{stem}.csv", d / f"scores_{stem}__robustdev.csv", *sorted(d.glob(f"scores_{stem}__*.csv"))):
        if f.exists():
            return pd.read_csv(f, keep_default_na=False).set_index("utt_id")
    raise FileNotFoundError(f"{run}: no scores for {stem}")


def member_cal(run):
    if run in VOICEGUARD_MODELS:
        return 1.0, 0.0
    c = json.loads((load_paths().runs_dir / run / "reliability.json").read_text(encoding="utf-8"))["calibration"]
    return c["a"], c["b"]


def table(stem):
    """(members, n) member log-odds (calibrated), labels, metadata frame."""
    frames = [read(r, stem) for r, _ in MEMBERS]
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    z = []
    for (run, _), f in zip(MEMBERS, frames):
        a, b = member_cal(run)
        z.append(a * (-f.loc[idx, "score"].to_numpy(float)) + b)
    return np.array(z), frames[0].loc[idx, "label"].to_numpy(int), frames[0].loc[idx]


def decide(p, member_p, cfg):
    votes_fake = (member_p >= cfg["vote_fake"]).sum(0)
    veto = (member_p >= cfg["veto_p"]).any(0)
    fake = (p >= cfg["t_fake"]) & (votes_fake >= cfg["min_agree"])
    real = (p <= cfg["t_real"]) & ~veto
    return fake, real


def stats(fake, real, y):
    dec = fake | real
    err = (fake & (y == 0)) | (real & (y == 1))
    return {"n": int(len(y)), "decided_pct": 100 * dec.mean(),
            "error_among_decided_pct": 100 * err[dec].mean() if dec.any() else float("nan"),
            "real_called_fake_pct": 100 * fake[y == 0].mean(), "fake_called_real_pct": 100 * real[y == 1].mean(),
            "fake_detected_pct": 100 * fake[y == 1].mean(), "real_confirmed_pct": 100 * real[y == 0].mean()}


def fused(z, w, bias):
    return sigmoid((np.asarray(w)[:, None] * z).sum(0) + bias)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fusion", choices=["equal", "lr", "e7a"], default="equal")
    ap.add_argument("--target", type=float, default=2.0, help="max %% of each tuning class on the wrong side")
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--veto-p", type=float, default=0.95)
    ap.add_argument("--main", default="e7a_wavlm_crnn_s42", help="the project model in the ensemble")
    ap.add_argument("--main-label", default="WavLM + CNN-BiLSTM-Att (E7a)")
    ap.add_argument("--tests", choices=["orig", "xd"], default="orig")
    ap.add_argument("--tag", help="write experiments/ensemble_<tag>/ instead of experiments/ensemble/")
    ap.add_argument("--no-write-config", action="store_true",
                    help="report only; leave runs/ensemble/ensemble.json (the app's ensemble) unchanged")
    args = ap.parse_args()
    global OUT
    MEMBERS[0] = (args.main, args.main_label)
    if args.tag:
        OUT = REPO_ROOT / "experiments" / f"ensemble_{args.tag}"
    tests_spec = TESTS_XD if args.tests == "xd" else TESTS

    tune = [table("asvspoof2019_la_dev")] + [table(f"asvspoof2019_la_robustdev_{c}_test") for c in CONDS]
    zt = np.concatenate([t[0] for t in tune], axis=1)
    yt = np.concatenate([t[1] for t in tune])
    tests = {name: table(stem) for name, stem in tests_spec}
    k = len(MEMBERS)

    fusions = {"e7a": (np.eye(k)[0], 0.0), "equal": (np.full(k, 1.0 / k), 0.0)}
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(zt.T, yt)
    fusions["lr"] = (lr.coef_[0], float(lr.intercept_[0]))

    report = {"members": [m for m, _ in MEMBERS], "target_pct": args.target, "fusions": {}}
    for fname, (w, b) in fusions.items():
        p = fused(zt, w, b)
        rows = {"weights": [float(v) for v in w], "bias": b}
        rows["eer_pct"] = {"tuning (dev + degraded dev)": 100 * eer_from_labels(-p, yt)[0]}
        for name, (z, y, meta) in tests.items():
            pt = fused(z, w, b)
            rows["eer_pct"][name] = 100 * eer_from_labels(-pt, y)[0]
            if name.startswith("DEEP"):
                rec = pd.DataFrame({"r": meta["recording_id"].to_numpy(), "z": np.log(pt / (1 - pt + 1e-12) + 1e-12),
                                    "y": y}).groupby("r").agg(z=("z", "mean"), y=("y", "first"))
                rows["eer_pct"]["DEEP-VOICE (recording)"] = 100 * eer_from_labels(-rec["z"], rec["y"])[0]
        report["fusions"][fname] = rows
    # single members, for reference
    report["single_members_eer_pct"] = {}
    for i, (run, _) in enumerate(MEMBERS):
        d = {"tuning (dev + degraded dev)": 100 * eer_from_labels(-zt[i], yt)[0]}
        for name, (z, y, _) in tests.items():
            d[name] = 100 * eer_from_labels(-z[i], y)[0]
        report["single_members_eer_pct"][run] = d

    # --- decision rule on the chosen fusion ---------------------------------------------
    w, b = fusions[args.fusion]
    p = fused(zt, w, b)
    t_fake = float(np.percentile(p[yt == 0], 100 - args.target))
    t_real = float(np.percentile(p[yt == 1], args.target))
    if t_real >= t_fake:
        t_real = t_fake = float(-eer_from_labels(-p, yt)[1])
    cfg = {"fusion": args.fusion, "bias": b, "t_real": t_real, "t_fake": t_fake, "vote_fake": 0.5, "vote_real": 0.5,
           "min_agree": args.min_agree, "veto_p": args.veto_p,
           "members": [{"run": r, "label": lab, "weight": float(wi), "a": member_cal(r)[0], "b": member_cal(r)[1]}
                       for (r, lab), wi in zip(MEMBERS, w)],
           "fitted_on": "ASVspoof 2019 LA dev, clean + 10 degraded dev conditions",
           "target_pct": args.target}
    member_p_t = sigmoid(zt)
    report["decision"] = {"config": {k: v for k, v in cfg.items() if k != "members"}, "sets": {}}
    variants = {"fused thresholds only": dict(cfg, min_agree=0, veto_p=1.01),
                "+ agreement gate and veto (used)": cfg}
    for vname, vc in variants.items():
        sets = {"tuning (dev + degraded dev)": stats(*decide(p, member_p_t, vc), yt)}
        for name, (z, y, _) in tests.items():
            sets[name] = stats(*decide(fused(z, w, b), sigmoid(z), vc), y)
        report["decision"]["sets"][vname] = sets
    # every member alone at its own 0.5 threshold, for comparison (no abstention)
    report["members_at_0_5"] = {}
    for i, (run, _) in enumerate(MEMBERS):
        d = {}
        for name, (z, y, _) in tests.items():
            q = sigmoid(z[i])
            d[name] = stats(q >= 0.5, q < 0.5, y)
        report["members_at_0_5"][run] = d

    ens_dir = load_paths().runs_dir / "ensemble"
    if not args.no_write_config:
        ens_dir.mkdir(parents=True, exist_ok=True)
        (ens_dir / "ensemble.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ensemble.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    md = markdown(report, cfg)
    (OUT / "ensemble.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))


def markdown(r, cfg):
    sets = list(next(iter(r["single_members_eer_pct"].values()))) + ["DEEP-VOICE (recording)"]
    L = [f"# Merged detector: {r['members'][0]} + VoiceGuard improved + VoiceGuard v2", "",
         "All numbers below are produced by `scripts/ensemble/fit_ensemble.py`. Tuning data: ASVspoof 2019 LA dev "
         "(clean + 10 degraded conditions). Test sets never set any weight or threshold.", "",
         "**Contamination warning:** VoiceGuard v2 was trained on about 79% of In-the-Wild (speaker split; the "
         "split files are not available). Its In-the-Wild numbers, and those of any fusion that includes it, are "
         "**not test results**. ASVspoof eval and DEEP-VOICE are clean for all three members (as far as is known: "
         "the training data of VoiceGuard improved is not recorded).", "",
         "## EER (%) of each member and fusion", "",
         "| System | " + " | ".join(sets) + " |", "|---" * (len(sets) + 1) + "|"]
    for run, d in r["single_members_eer_pct"].items():
        L.append(f"| {run} | " + " | ".join(f"{d[s]:.2f}" if s in d else "–" for s in sets) + " |")
    for f, d in r["fusions"].items():
        L.append(f"| fusion: {f} (w = {', '.join(f'{v:.2f}' for v in d['weights'])}) | " +
                 " | ".join(f"{d['eer_pct'][s]:.2f}" if s in d["eer_pct"] else "–" for s in sets) + " |")
    c = r["decision"]["config"]
    L += ["", f"## Decision rule (fusion = {c['fusion']})", "",
          f"Fake if fused P(fake) ≥ {c['t_fake']:.3f} and at least {c['min_agree']} members vote fake; Real if "
          f"fused P(fake) ≤ {c['t_real']:.3f} and no member has P(fake) ≥ {c['veto_p']}; otherwise Uncertain "
          f"(thresholds: ≤ {r['target_pct']}% of each tuning class on the wrong side).", "",
          "| Rule | Set | decided % | error among decided % | real → Fake % (false detection) | "
          "fake → Real % | fakes caught % | reals confirmed % |", "|---|---|---|---|---|---|---|---|"]
    for v, ss in r["decision"]["sets"].items():
        for s, t in ss.items():
            L.append(f"| {v} | {s} | {t['decided_pct']:.1f} | {t['error_among_decided_pct']:.2f} | "
                     f"{t['real_called_fake_pct']:.2f} | {t['fake_called_real_pct']:.2f} | "
                     f"{t['fake_detected_pct']:.1f} | {t['real_confirmed_pct']:.1f} |")
    L += ["", "## For comparison: each member alone, plain 0.5 threshold, no abstention", "",
          "| Member | Set | error % | real → Fake % | fake → Real % |", "|---|---|---|---|---|"]
    for run, ss in r["members_at_0_5"].items():
        for s, t in ss.items():
            L.append(f"| {run} | {s} | {t['error_among_decided_pct']:.2f} | {t['real_called_fake_pct']:.2f} | "
                     f"{t['fake_called_real_pct']:.2f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
