"""Error analysis of the app's detector (plan P4.6) from the stored per-utterance scores.

Default (--model e7a): WavLM + CNN-BiLSTM-Att (E7a) alone, the only model the app runs since 2026-11-23;
writes experiments/error_analysis_e7a/ (the trust table the app reads). --model ensemble rebuilds the
retired three-model analysis in experiments/error_analysis/ for reference.

No audio is decoded and no model is run: the per-utterance scores of the three members
(runs/<member>/scores_<set>.csv) are fused exactly as the app does (runs/ensemble/ensemble.json)
and the two-way decision at P(fake) = 0.5 is analysed on the held-out sets:

  - ASVspoof 2019 LA eval: errors per attack (A07-A19) and per duration bin;
  - In-the-Wild: errors per speaker (contaminated for VoiceGuard v2, reported for analysis only);
  - every set: how often the verdict is right for each member-agreement pattern and each
    confidence band. The app uses this table to say how trustworthy a given verdict has been;
  - the worst false accepts (fakes called REAL) and false rejects (real called FAKE).

    python scripts/error_analysis.py                    # E7a (the app's model)
    python scripts/error_analysis.py --model ensemble   # retired merged detector
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_ensemble  # noqa: E402
from fit_ensemble import MEMBERS, table  # noqa: E402

from adfd.calibration import sigmoid  # noqa: E402
from adfd.metrics import eer_from_labels  # noqa: E402
from adfd.paths import REPO_ROOT, load_paths  # noqa: E402

OUT = REPO_ROOT / "experiments" / "error_analysis"
OUT_E7A = REPO_ROOT / "experiments" / "error_analysis_e7a"
E7A = [("e7a_wavlm_crnn_s42", "WavLM + CNN-BiLSTM-Att (E7a)")]
SETS = [("ASVspoof eval", "asvspoof2019_la_eval"), ("In-the-Wild", "in_the_wild_test")]
CONF_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5001]          # |P(fake) - 0.5|
CONF_NAMES = ["very low (50±10%)", "low (60–70% / 30–40%)", "medium (70–80% / 20–30%)",
              "high (80–90% / 10–20%)", "very high (≥90% / ≤10%)"]
DUR_BINS = [0, 2, 3, 4, 6, 1e9]
DUR_NAMES = ["< 2 s", "2–3 s", "3–4 s", "4–6 s", "≥ 6 s"]


def fuse(z: np.ndarray, cfg: dict) -> np.ndarray:
    w = np.array([m["weight"] for m in cfg["members"]])[:, None]
    s = (w * np.clip(z, -cfg["clip"], cfg["clip"])).sum(0)
    return sigmoid(cfg["A"] * s + cfg["B"])


def rates(fake: np.ndarray, y: np.ndarray) -> dict:
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {"n": int(len(y)), "n_real": n0, "n_fake": n1,
            "accuracy_pct": 100 * float((fake == (y == 1)).mean()) if len(y) else None,
            "real_called_fake_pct": 100 * float(fake[y == 0].mean()) if n0 else None,
            "fake_called_real_pct": 100 * float((~fake)[y == 1].mean()) if n1 else None}


def reliability_tables(p: np.ndarray, member_p: np.ndarray, y: np.ndarray) -> dict:
    fake = p >= 0.5
    correct = fake == (y == 1)
    votes = (member_p >= 0.5).sum(0)
    agree = np.where(fake, votes, len(member_p) - votes)       # members that share the final verdict
    conf = np.digitize(np.abs(p - 0.5), CONF_BINS) - 1
    by_agree = {int(k): {"n": int((agree == k).sum()), "accuracy_pct": 100 * float(correct[agree == k].mean())}
                for k in range(len(member_p) + 1) if (agree == k).any()}
    by_conf = {CONF_NAMES[k]: {"n": int((conf == k).sum()), "accuracy_pct": 100 * float(correct[conf == k].mean())}
               for k in range(len(CONF_NAMES)) if (conf == k).any()}
    grid = {}
    for k in range(len(member_p) + 1):
        for c in range(len(CONF_NAMES)):
            m = (agree == k) & (conf == c)
            if m.sum() >= 20:
                grid[f"{k}|{c}"] = {"n": int(m.sum()), "accuracy_pct": 100 * float(correct[m].mean())}
    return {"by_agreement": by_agree, "by_confidence": by_conf, "grid": grid}


def worst(meta: pd.DataFrame, p: np.ndarray, member_p: np.ndarray, y: np.ndarray, label: int, k: int = 12):
    """label 1: fakes with the lowest P(fake) (false accepts); 0: reals with the highest (false rejects)."""
    idx = np.where(y == label)[0]
    idx = idx[np.argsort(p[idx])][:k] if label == 1 else idx[np.argsort(-p[idx])][:k]
    rows = []
    for i in idx:
        r = meta.iloc[i]
        rows.append({"utt_id": str(meta.index[i]), "speaker": str(r["speaker"]), "attack": str(r["attack"]),
                     "duration_s": round(float(r["end_s"] - r["start_s"]), 2), "p_fake_pct": round(100 * float(p[i]), 1),
                     **{f"{run}_pct": round(100 * float(member_p[j, i]), 1) for j, (run, _) in enumerate(MEMBERS)}})
    return rows


def main() -> None:
    global MEMBERS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["e7a", "ensemble"], default="e7a")
    single = ap.parse_args().model == "e7a"
    out_dir = OUT_E7A if single else OUT
    if single:
        MEMBERS = fit_ensemble.MEMBERS = E7A          # table() reads fit_ensemble.MEMBERS at call time
        cfg = None
    else:
        cfg = json.loads((load_paths().runs_dir / "ensemble" / "ensemble.json").read_text(encoding="utf-8"))
    report: dict = {"decision": ("FAKE if E7a's calibrated P(fake) >= 0.5" if single
                                 else "FAKE if merged P(fake) >= 0.5"),
                    "model": "e7a" if single else "ensemble",
                    "members": [r for r, _ in MEMBERS], "confidence_bins": CONF_NAMES, "sets": {}}
    for name, stem in SETS:
        z, y, meta = table(stem)
        member_p = sigmoid(z)
        p = member_p[0] if single else fuse(z, cfg)
        fake = p >= 0.5
        d = {"overall": rates(fake, y), "eer_pct": 100 * float(eer_from_labels(-p, y)[0]),
             "reliability": reliability_tables(p, member_p, y),
             "false_accepts": worst(meta, p, member_p, y, 1), "false_rejects": worst(meta, p, member_p, y, 0)}
        dur = (meta["end_s"] - meta["start_s"]).to_numpy(float)
        db = np.digitize(dur, DUR_BINS) - 1
        d["by_duration"] = {DUR_NAMES[k]: rates(fake[db == k], y[db == k]) for k in range(len(DUR_NAMES))
                            if (db == k).any()}
        if stem.startswith("asvspoof"):
            att = meta["attack"].to_numpy(str)
            d["by_attack"] = {}
            for a in sorted(set(att) - {"-"}):
                k = att == a
                e7a_eer = 100 * eer_from_labels(-z[0][k | (att == "-")], y[k | (att == "-")])[0]
                d["by_attack"][a] = {**rates(fake[k], y[k]), "merged_eer_pct":
                                     100 * float(eer_from_labels(-p[k | (att == "-")], y[k | (att == "-")])[0]),
                                     "e7a_eer_pct": float(e7a_eer)}
        if stem.startswith("in_the_wild"):
            spk = meta["speaker"].to_numpy(str)
            rows = {}
            for s in sorted(set(spk)):
                k = spk == s
                if k.sum() >= 30:
                    rows[s] = rates(fake[k], y[k])
            d["by_speaker"] = dict(sorted(rows.items(), key=lambda kv: kv[1]["accuracy_pct"]))   # hardest first
        report["sets"][name] = d
        print(f"{name}: n={len(y)} acc={d['overall']['accuracy_pct']:.1f}%", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "error_analysis.md").write_text(markdown(report), encoding="utf-8")


def fmt(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


def markdown(r: dict) -> str:
    single = r.get("model") == "e7a"
    title = ("WavLM + CNN-BiLSTM-Att (E7a), the app's only model" if single
             else "the retired merged detector")
    out = [f"# Error analysis of {title} (P4.6)", "",
           "Produced by `scripts/error_analysis.py` from the stored per-utterance scores; decision: "
           f"{r['decision']}." + ("" if single else " In-the-Wild is contaminated for VoiceGuard v2 (trained on most "
                                     "of it)."), ""]
    for name, d in r["sets"].items():
        o = d["overall"]
        out += [f"## {name}", "", f"n = {o['n']:,} · accuracy {fmt(o['accuracy_pct'])}% · real → FAKE "
                f"{fmt(o['real_called_fake_pct'])}% · fake → REAL {fmt(o['fake_called_real_pct'])}% · "
                f"EER {d['eer_pct']:.2f}%", ""]
        if not single:
            out += ["**Accuracy by how many members share the final verdict**", "",
                    "| members agreeing | n | accuracy % |", "|---|---|---|"]
            out += [f"| {k} of 3 | {v['n']:,} | {v['accuracy_pct']:.1f} |"
                    for k, v in d["reliability"]["by_agreement"].items()]
            out.append("")
        out += [f"**Accuracy by confidence of the {'model' if single else 'merged'} probability**", "",
                "| confidence | n | accuracy % |",
                "|---|---|---|"]
        out += [f"| {k} | {v['n']:,} | {v['accuracy_pct']:.1f} |" for k, v in d["reliability"]["by_confidence"].items()]
        if "by_attack" in d:
            out += ["", "**Per attack** (fake → REAL at 0.5; EER against all bona fide eval)", "",
                    ("| attack | n | fake → REAL % | EER % |" if single else "| attack | n | fake → REAL % | merged EER % | E7a EER % |"),
                    "|---|---|---|---|" if single else "|---|---|---|---|---|"]
            out += [f"| {a} | {v['n']:,} | {fmt(v['fake_called_real_pct'])} | {v['merged_eer_pct']:.2f} |"
                    + ("" if single else f" {v['e7a_eer_pct']:.2f} |") for a, v in d["by_attack"].items()]
        out += ["", "**By clip duration**", "", "| duration | n | accuracy % | real → FAKE % | fake → REAL % |",
                "|---|---|---|---|---|"]
        out += [f"| {k} | {v['n']:,} | {fmt(v['accuracy_pct'])} | {fmt(v['real_called_fake_pct'])} | "
                f"{fmt(v['fake_called_real_pct'])} |" for k, v in d["by_duration"].items()]
        if "by_speaker" in d:
            out += ["", "**Hardest In-the-Wild speakers** (lowest accuracy, ≥ 30 clips)", "",
                    "| speaker | n | accuracy % |", "|---|---|---|"]
            out += [f"| {s} | {v['n']} | {v['accuracy_pct']:.1f} |" for s, v in list(d["by_speaker"].items())[:10]]
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
