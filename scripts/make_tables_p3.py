"""Phase 3 main table: every model's EER on the three test sets, min t-DCF, robustness,
size and speed, plus the model chosen for the app.

All trained models use the Phase 2 protocol (silence-trimmed, RawBoost + gain) unless
marked; rows average all seeds (mean ± std, n shown). The app model is chosen on dev
data only: lowest mean EER over the 10 degraded ASVspoof dev conditions (the Phase 2
selection metric), clean dev EER as the tie-break. Test columns are reported only.

Usage:
    python scripts/make_tables_p3.py
Writes experiments/tables/phase3.md and phase3.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

from adfd.paths import REPO_ROOT

EXP = REPO_ROOT / "experiments"
SEEDS = (42, 43, 44)
CONDS = ["clean", "mp3_51k", "mp3_19k", "opus_84k", "opus_10k", "telephone", "white_20db", "white_10db",
         "white_5db", "pink_10db"]
ROWS = [  # (group, label, prefix)
    ("Phase 2 reference", "E1 CNN-BiLSTM, MFCC, untrimmed, no augmentation (v1 paper model)", "e1_cnn_bilstm_mfcc"),
    ("Phase 2 reference", "E4a CNN-BiLSTM, LFCC (Phase 2 selection)", "e4a_rawboost"),
    ("P3.1 published baselines", "E5a LCNN, LFCC", "e5a_lcnn"),
    ("P3.1 published baselines", "E5b RawNet2, raw waveform", "e5b_rawnet2"),
    ("P3.1 published baselines", "E5c AASIST, raw waveform", "e5c_aasist"),
    ("P3.2 CNN-RNN upgrades", "E6a SE-residual CNN, time-major BiLSTM", "e6a_crnn_seres"),
    ("P3.2 CNN-RNN upgrades", "E6b E6a + attentive stats pooling (CNN-BiLSTM-Att)", "e6b_crnn_seres_asp"),
    ("P3.2 CNN-RNN upgrades", "E6c light SE-res CNN, BiGRU, attentive stats pooling", "e6c_crnn_light_gru"),
    ("P3.3 pretrained SSL front end", "E7a WavLM-Base+ (fine-tuned) + CNN-BiLSTM-Att", "e7a_wavlm_crnn"),
]


def load(prefix: str, seed: int, tag: str = ""):
    f = EXP / (f"{prefix}_s{seed}__{tag}" if tag else f"{prefix}_s{seed}") / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8"))["sets"] if f.exists() else None


def collect(prefix: str) -> dict:
    cols = {k: [] for k in ("dev", "eval", "tdcf", "itw", "dv", "deg_eval", "deg_dev")}
    for s in SEEDS:
        m = load(prefix, s)
        if m is None:
            continue
        cols["dev"].append(m["asvspoof2019_la:dev"]["eer_pct"])
        if "asvspoof2019_la:eval" in m:
            cols["eval"].append(m["asvspoof2019_la:eval"]["eer_pct"])
            cols["tdcf"].append(m["asvspoof2019_la:eval"]["min_tdcf"])
        if "in_the_wild:test" in m:
            cols["itw"].append(m["in_the_wild:test"]["eer_pct"])
        if "deep_voice:test" in m:
            cols["dv"].append(m["deep_voice:test"]["per_recording"]["eer_pct"])
        for tag, key, pre in (("robust", "deg_eval", "asvspoof2019_la_robust_"),
                              ("robustdev", "deg_dev", "asvspoof2019_la_robustdev_")):
            r = load(prefix, s, tag)
            if r and all(f"{pre}{c}:test" in r for c in CONDS):
                cols[key].append(float(np.mean([r[f"{pre}{c}:test"]["eer_pct"] for c in CONDS])))
    return cols


def fmt(v, digits=2):
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"


def main() -> int:
    eff_f = EXP / "efficiency" / "efficiency.json"
    eff = json.loads(eff_f.read_text(encoding="utf-8")) if eff_f.exists() else {}
    data, lines = {}, [
        "# Phase 3 results", "",
        "EER in %, mean ± std over seeds (n = seeds; n = 1 rows are the shortened Phase 3 prototype runs, see the "
        "plan). All trained models use the Phase 2 protocol "
        "(silence-trimmed audio, RawBoost + gain) except E1. Degraded = mean EER over 10 codec/noise/channel "
        "conditions of the ASVspoof eval (reported) or dev (selection) subset. Speed = seconds of audio "
        "scored per second (see `experiments/efficiency/efficiency.md`).", "",
        "| Model | n | dev | eval | eval min t-DCF | In-the-Wild | DEEP-VOICE (recording) | degraded eval | "
        "degraded dev | parameters | GPU × real time | CPU × real time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    group = None
    for g, label, prefix in ROWS:
        c = collect(prefix)
        if not c["dev"]:
            continue
        if g != group:
            lines.append(f"| *{g}* |  |  |  |  |  |  |  |  |  |  |  |")
            group = g
        e = eff.get(prefix, {})
        data[prefix] = {"label": label, **{k: v for k, v in c.items()}, "efficiency": e}
        lines.append(f"| {label} | {len(c['dev'])} | {fmt(c['dev'])} | {fmt(c['eval'])} | {fmt(c['tdcf'], 3)} | "
                     f"{fmt(c['itw'])} | {fmt(c['dv'])} | {fmt(c['deg_eval'])} | {fmt(c['deg_dev'])} | "
                     f"{e['params']:,} | {e['gpu_audio_s_per_s']:,.0f} | {e['cpu_audio_s_per_s']:,.1f} |"
                     if e else
                     f"| {label} | {len(c['dev'])} | {fmt(c['dev'])} | {fmt(c['eval'])} | {fmt(c['tdcf'], 3)} | "
                     f"{fmt(c['itw'])} | {fmt(c['dv'])} | {fmt(c['deg_eval'])} | {fmt(c['deg_dev'])} | – | – | – |")

    fus_f = EXP / "fusion" / "fusion.json"
    if fus_f.exists():
        fus = json.loads(fus_f.read_text(encoding="utf-8"))
        full = " + ".join(fus["members"])
        r = fus["systems"][full]
        mm, sd = r["mean"], r["std"]
        cell = lambda k, d=2: ("–" if np.isnan(mm[k]) else f"{mm[k]:.{d}f}" if np.isnan(sd[k])  # noqa: E731
                               else f"{mm[k]:.{d}f} ± {sd[k]:.{d}f}")
        lines += [f"| *P3.4 score fusion (dev-trained logistic regression)* |  |  |  |  |  |  |  |  |  |  |  |",
                  f"| Fusion: {full} | {r['n']} | {cell('dev EER')} | {cell('ASVspoof eval')} | "
                  f"{cell('eval min t-DCF', 3)} | {cell('In-the-Wild')} | {cell('DEEP-VOICE (recording)')} | "
                  f"{cell('degraded eval, mean of 10')} | – | sum of members | – | – |"]
        lines += ["", "Fusion details, including every pair: `experiments/fusion/fusion.md`."]

    # app model: dev data only
    cands = {p: (np.mean(d["deg_dev"]), np.mean(d["dev"])) for p, d in data.items()
             if d["deg_dev"] and p != "e1_cnn_bilstm_mfcc"}
    if cands:
        best = min(cands, key=lambda p: cands[p])
        ranking = sorted(cands.items(), key=lambda kv: kv[1])
        lines += ["", f"**Model for the app: {data[best]['label']}**, chosen on dev data only: lowest mean "
                      f"degraded-dev EER ({cands[best][0]:.2f}%; clean dev {cands[best][1]:.2f}%). Ranking: "
                  + ", ".join(f"{p} {v[0]:.2f}" for p, v in ranking) + "."]
        data["_app_model"] = best
    out = EXP / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3.json").write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    md = "\n".join(lines) + "\n"
    (out / "phase3.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
