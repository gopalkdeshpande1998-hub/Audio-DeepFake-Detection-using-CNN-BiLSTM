"""Which frequency bands the mel-spectrogram models rely on, per attack and per dataset (plan P4.5).

For a stratified sample of clips (ASVspoof 2019 LA eval: bona fide and each attack A07-A19;
In-the-Wild: real and fake), computes for each 128-band log-mel model (VoiceGuard improved,
VoiceGuard v2) two maps over the input mel spectrogram:
  - gradient x input saliency (adfd.inference.Analyzer.saliency)
  - Grad-CAM on the last convolution (adfd.interpret.grad_cam)
and reports the mean share of each map in the low (< 1 kHz), mid (1-4 kHz) and high (4-8 kHz)
bands (adfd.forensics.BANDS). A uniform map would give each band its share of the mel axis.

    python scripts/evaluation/band_analysis.py [--n 40] [--device cpu]
Writes experiments/band_analysis/band_analysis.{json,md,png}. Runs on the CPU in a few
minutes; In-the-Wild is contaminated for VoiceGuard v2 (trained on most of it).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from adfd.data.audio_cache import AudioCache
from adfd.forensics import BANDS, band_shares
from adfd.inference import Analyzer
from adfd.interpret import grad_cam
from adfd.paths import REPO_ROOT, configure_caches, load_paths

OUT = REPO_ROOT / "experiments" / "band_analysis"
MODELS = ["vg_improved", "vg_v2"]


def groups(n: int, seed: int = 0) -> list[tuple[str, str, pd.DataFrame]]:
    paths = load_paths()
    out = []
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    ev = asv[asv["split"] == "eval"]
    for att in ["-"] + sorted(set(ev["attack"]) - {"-"}):
        g = ev[ev["attack"] == att]
        out.append(("ASVspoof eval", "bona fide" if att == "-" else att, g.sample(min(n, len(g)), random_state=seed)))
    itw = pd.read_csv(paths.manifest("in_the_wild"), keep_default_na=False)
    for lab, name in ((0, "real"), (1, "fake")):
        g = itw[itw["label"] == lab]
        out.append(("In-the-Wild", name, g.sample(min(n, len(g)), random_state=seed)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="clips per group")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    configure_caches()
    paths = load_paths()
    caches = {"ASVspoof eval": AudioCache(paths.cache_dir, "asvspoof2019_la"),
              "In-the-Wild": AudioCache(paths.cache_dir, "in_the_wild")}
    gs = groups(args.n)
    rows = []
    for run in MODELS:
        an = Analyzer(paths.runs_dir / run, args.device)
        for ds, grp, df in gs:
            sal_sh, cam_sh = [], []
            for rid in df["recording_id"]:
                x = caches[ds].read(rid).astype(np.float32) / 32768.0
                sal = an.saliency(x)
                gc = grad_cam(an, x)
                n_frames = max(1, int(np.ceil(min(len(x), an.crop) / 256)))   # ignore the zero padding
                sal_sh.append([b["share_pct"] for b in band_shares(sal["map"][:, :n_frames])])
                cam_sh.append([b["share_pct"] for b in band_shares(gc["cam"][:, :n_frames])])
            s, c = np.mean(sal_sh, 0), np.mean(cam_sh, 0)
            rows.append({"model": run, "dataset": ds, "group": grp, "n": len(df),
                         **{f"saliency_{k}": float(v) for k, v in zip(("low", "mid", "high"), s)},
                         **{f"gradcam_{k}": float(v) for k, v in zip(("low", "mid", "high"), c)}})
            print(f"{run} {ds} {grp}: saliency low/mid/high {s.round(1)}  grad-cam {c.round(1)}", flush=True)
    bw = [100 * b["bandwidth_share_pct"] / 100 for b in band_shares(np.ones((128, 1)))]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"bands": [b[0] for b in BANDS], "mel_axis_share_pct": bw, "rows": rows, "n_per_group": args.n}
    (OUT / "band_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "band_analysis.md").write_text(markdown(res), encoding="utf-8")
    plot(res)


def markdown(res: dict) -> str:
    bw = res["mel_axis_share_pct"]
    out = ["# Which frequency bands the mel models rely on (P4.5)", "",
           f"Produced by `scripts/evaluation/band_analysis.py`, {res['n_per_group']} clips per group. Share (%) of gradient x "
           "input saliency and of Grad-CAM in each band of the 128-band mel input. A uniform map would give "
           f"low {bw[0]:.0f}% / mid {bw[1]:.0f}% / high {bw[2]:.0f}% (the bands' share of the mel axis). "
           "In-the-Wild is contaminated for VoiceGuard v2.", ""]
    for run in MODELS:
        out += [f"## {run}", "", "| dataset | group | n | saliency low | mid | high | Grad-CAM low | mid | high |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in res["rows"]:
            if r["model"] == run:
                out.append(f"| {r['dataset']} | {r['group']} | {r['n']} | {r['saliency_low']:.1f} | "
                           f"{r['saliency_mid']:.1f} | {r['saliency_high']:.1f} | {r['gradcam_low']:.1f} | "
                           f"{r['gradcam_mid']:.1f} | {r['gradcam_high']:.1f} |")
        out.append("")
    return "\n".join(out)


def plot(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(res["rows"])
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(12, 3.2 * len(MODELS)), sharex=True)
    for ax, run in zip(np.atleast_1d(axes), MODELS):
        d = df[df["model"] == run]
        lab = [f"{a[:3]} {g}" if a.startswith("In") else g for a, g in zip(d["dataset"], d["group"])]
        bottom = np.zeros(len(d))
        for k, col in zip(("low", "mid", "high"), ("#93c5fd", "#2563eb", "#1e3a8a")):
            v = d[f"saliency_{k}"].to_numpy()
            ax.bar(lab, v, bottom=bottom, color=col, label=k)
            bottom += v
        ax.set_ylabel("saliency share (%)")
        ax.set_title(run, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
    plt.setp(np.atleast_1d(axes)[-1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "band_analysis.png", dpi=120)


if __name__ == "__main__":
    main()
