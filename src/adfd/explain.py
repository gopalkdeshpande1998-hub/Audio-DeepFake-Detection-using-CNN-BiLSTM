"""Explain one audio file: calibrated verdict, OOD check, score timeline and saliency heatmap.

Usage:
    python -m adfd.explain path/to/audio.wav [--run runs/<model run>] [--out explanations/] [--device cpu]
Writes <out>/<file stem>/verdict.json, timeline.png and heatmap.png, and prints the verdict.
Without --run the app model is used (adfd.registry.default_run()).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adfd.inference import SR, Analyzer

INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e4e3df", "#fcfcfb"
LINE, FAKE_BAND, REAL_BAND = "#2a78d6", "#fbe3d6", "#dcf3ea"


def plot_timeline(res: dict, ax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    tl = res["timeline"]
    centres = [(w["start_s"] + w["end_s"]) / 2 for w in tl]
    p = [w["p_fake"] for w in tl]
    th = res.get("thresholds") or {"t_real": 0.5, "t_fake": 0.5}
    ax.axhspan(th["t_fake"], 1, color=FAKE_BAND, zorder=0)
    ax.axhspan(0, th["t_real"], color=REAL_BAND, zorder=0)
    ax.text(0.005, 0.97, "Fake", transform=ax.transAxes, va="top", color=INK_2, fontsize=8)
    ax.text(0.005, 0.03, "Real", transform=ax.transAxes, va="bottom", color=INK_2, fontsize=8)
    for w in res["suspicious_segments"]:
        ax.axvspan(w["start_s"], w["end_s"], color=FAKE_BAND, alpha=0.6, zorder=0)
    for w in (tl if len(tl) <= 3 else []):   # short clips: show each window's span, so one window is visible
        ax.hlines(w["p_fake"], w["start_s"], w["end_s"], color=LINE, alpha=0.35, linewidth=6, zorder=2)
    ax.plot(centres, p, color=LINE, marker="o" if len(p) < 40 else None, markersize=5, linewidth=2, zorder=3,
            markeredgecolor=SURFACE)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(res["duration_s"], 1e-3))
    ax.set_xlabel("time (s), window centre", color=INK_2)
    ax.set_ylabel("P(fake)", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Score over time: {res['verdict']} (P(fake) = {res['p_fake']:.2f})", loc="left", color=INK, fontsize=10)
    if fig is not None:
        fig.tight_layout()
    return fig


def plot_heatmap(seg: np.ndarray, sal: dict, source_s: float | None = None):
    """seg: the 4 s model input; source_s: length of the real audio in it (repeat seams are marked)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, hop_length=160, n_mels=80)(torch.from_numpy(seg).float())
    mel = (10 * torch.log10(mel + 1e-10)).numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), facecolor=SURFACE, sharex=False)
    t_end = len(seg) / SR
    axes[0].imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=(0, t_end, 0, 8))
    axes[0].set_ylabel("mel band (kHz approx.)", color=INK_2)
    axes[0].set_title("Log-mel spectrogram of the most suspicious window", loc="left", color=INK, fontsize=10)
    if "map" in sal:
        m = sal["map"]
        m = np.sqrt(np.clip(m / (np.percentile(m, 99) + 1e-12), 0, 1))   # robust scale, no single-pixel blow-out
        axes[1].imshow(m, origin="lower", aspect="auto", cmap="viridis", extent=(0, t_end, 0, m.shape[0]))
        n = m.shape[0]
        if n % 3 == 0 and sal["frontend"] in ("lfcc", "cqcc"):
            for k, lab in enumerate(("static", "delta", "delta-delta")):
                axes[1].axhline(k * n / 3, color=SURFACE, linewidth=0.6)
                axes[1].text(t_end * 0.995, (k + 0.5) * n / 3, lab, color=SURFACE, fontsize=7, ha="right", va="center")
        axes[1].set_ylabel(f"{sal['frontend'].upper()} coefficient", color=INK_2)
        axes[1].set_title(f"Saliency (|gradient × input|) on the model's {sal['frontend'].upper()} input",
                          loc="left", color=INK, fontsize=10)
    else:
        prof = np.asarray(sal["time_profile"], float)
        axes[1].fill_between(np.linspace(0, t_end, len(prof)), prof / (prof.max() + 1e-12), color=LINE, alpha=0.8)
        axes[1].set_ylabel("relative saliency", color=INK_2)
        axes[1].set_title(f"Saliency over time ({sal['frontend']} input has no frequency axis)", loc="left",
                          color=INK, fontsize=10)
    axes[1].set_xlabel("time in window (s)", color=INK_2)
    if source_s and source_s < t_end - 0.05:   # short clip: repeated to fill the window, as in training
        for k in np.arange(source_s, t_end, source_s):
            for ax in axes:
                ax.axvline(k, color="#ffffff", linewidth=1, linestyle=(0, (2, 2)))
        axes[0].text(0.995, 1.02, f"clip {source_s:.1f} s, repeated to {t_end:.0f} s (dashed = seams)",
                     transform=axes[0].transAxes, ha="right", va="bottom", color=INK_2, fontsize=8)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


REGION_COLOR = {"Fake": "#d92d20", "Real": "#16a34a", "Uncertain": "#eda100", "No speech": "#b9b7ae"}


ZONE_COLOR = {"Fake": "#d92d20", "Border": "#f59e0b", "Uncertain": "#f59e0b", "Real": "#16a34a"}   # labelled
BORDER = 10.0     # +-10 points around the decision line: drawn amber for reference, never a verdict


def plot_score_line(res: dict, show_members: bool = True):
    """Easy-to-read line graph for an ensemble result: chance of fake (%) per second on labelled zones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decision = 100 * res["thresholds"].get("decision", 0.5)
    ps = res["per_second"]
    c = np.array([(w["start_s"] + w["end_s"]) / 2 for w in ps])
    p = np.array([np.nan if w["p_fake"] is None else 100 * w["p_fake"] for w in ps], float)
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lo, hi = decision - BORDER, decision + BORDER
    for (y0, y1, z, lab) in ((hi, 100, "Fake", "FAKE zone"), (lo, hi, "Border", "near the line (visual only)"),
                             (0, lo, "Real", "REAL zone")):
        ax.axhspan(y0, y1, color=ZONE_COLOR[z], alpha=0.12, linewidth=0, zorder=0)
        ax.text(0.006, (y1 - 1.5) / 100, lab, transform=ax.transAxes, va="top", color=INK_2, fontsize=8.5,
                fontweight="bold")
    ax.axhline(decision, color=INK_2, linestyle=(0, (5, 3)), linewidth=1.2, zorder=2,
               label=f"decision threshold {decision:.0f}%")
    for sg in res.get("segments", []):
        ax.axvspan(sg["start_s"], sg["end_s"], color=ZONE_COLOR["Fake"], alpha=0.08, linewidth=0, zorder=0)
        for e in (sg["start_s"], sg["end_s"]):
            ax.axvline(e, color=ZONE_COLOR["Fake"], linewidth=0.9, alpha=0.7, zorder=1)
    if show_members:
        for m, ls, col in zip(res.get("members", []), [(0, (6, 3)), (0, (2, 2)), (0, (8, 3, 2, 3))],
                              ["#6f6e69", "#8a887f", "#a3a198"]):
            q = np.array([np.nan if v is None else 100 * v for v in m["per_second"]], float)
            ax.plot(c, q, color=col, linestyle=ls, linewidth=1.2, label=m["label"])
    ax.plot(c, p, color=LINE, linewidth=2.4, label="synthetic probability", zorder=3,
            solid_joinstyle="round", solid_capstyle="round", marker="o" if len(p) <= 60 else None, markersize=3.5)
    hot = p >= decision
    if np.any(hot):
        ax.scatter(c[hot], p[hot], s=26, color=ZONE_COLOR["Fake"], edgecolor=SURFACE, linewidth=1, zorder=4,
                   label="step above the threshold")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(res["duration_s"], 1e-3))
    ax.set_xlabel("time (s)", color=INK_2)
    ax.set_ylabel("synthetic probability (%)", color=INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    ax.set_title("Synthetic probability over time (per local window; red bands = flagged segments)", loc="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def plot_forensic(x: np.ndarray, res: dict):
    """Whole-file forensic view for an ensemble result (adfd.ensemble).

    Row 1: waveform, coloured by the Real / Uncertain / Fake region it falls in.
    Row 2: 128-band log-mel spectrogram (the VoiceGuard models' input), region bar on top,
           and the saliency of the analysed window outlined on it.
    Row 3: fused P(fake) per second with each member's score and the decision bands.
    """
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dur = len(x) / SR
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.6, 1.2]})
    regs = res.get("regions", [])
    # 1. waveform by region
    ax = axes[0]
    step = max(1, len(x) // 8000)
    t = np.arange(0, len(x), step) / SR
    env = x[::step]
    ax.plot(t, env, color="#9a9890", linewidth=0.5)
    for r in regs:
        m = (t >= r["start_s"]) & (t <= r["end_s"])
        ax.plot(t[m], env[m], color=REGION_COLOR[r["label"]], linewidth=0.6)
    ax.set_ylabel("amplitude", color=INK_2)
    ax.set_title("Where: waveform coloured by local analysis (red = suspected fake, green = real, grey = no speech)",
                 loc="left", color=INK, fontsize=10)
    # 2. mel spectrogram, region bar, saliency box
    ax = axes[1]
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=x.astype(np.float32), sr=SR, n_fft=1024,
                                                             hop_length=256, n_mels=128), ref=np.max)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=(0, dur, 0, 128), vmin=-80, vmax=0)
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=SR / 2)
    ticks = [0, 500, 1000, 2000, 4000, 8000]
    ax.set_yticks([int(np.argmin(np.abs(freqs - f))) for f in ticks], [f"{f / 1000:g}k" if f else "0" for f in ticks])
    ax.set_ylabel("frequency (Hz, mel scale)", color=INK_2)
    for r in regs:
        ax.add_patch(Rectangle((r["start_s"], 124), r["end_s"] - r["start_s"], 4, color=REGION_COLOR[r["label"]],
                               linewidth=0))
    sal = res.get("saliency")
    if sal and sal.get("map") is not None:
        m = np.abs(sal["map"])
        n_real = max(1, int(np.ceil(sal["source_s"] * SR / 256)))
        m = m[:, :n_real]
        m = m / (np.percentile(m, 99) + 1e-12)
        a = sal["start_s"]
        ax.contour(np.linspace(a, a + n_real * 256 / SR, m.shape[1]), np.arange(m.shape[0]),
                   np.clip(m, 0, 1), levels=[0.5], colors=["#5ad1ff"], linewidths=0.9)
        ax.axvspan(a, a + sal["source_s"], ymin=0, ymax=0.965, fill=False, edgecolor="#5ad1ff", linewidth=1.2)
        ax.text(a + 0.05, 2, "analysed window: outlines = time-frequency areas that drove the score",
                color="#5ad1ff", fontsize=7.5, va="bottom")
    ax.set_title("Log-mel spectrogram (128 bands) with region bar",
                 loc="left", color=INK, fontsize=10)
    # 3. score per second
    ax = axes[2]
    th = res["thresholds"]
    ax.axhspan(th["t_fake"], 1, color=FAKE_BAND, zorder=0)
    ax.axhspan(0, th["t_real"], color=REAL_BAND, zorder=0)
    for mm, ls in zip(res.get("members", []), ["-", "--", ":"]):
        c = [(w["start_s"] + w["end_s"]) / 2 for w in mm["timeline"]]
        ax.plot(c, [w["p_fake"] for w in mm["timeline"]], color="#8a887f", linestyle=ls, linewidth=1,
                label=mm["label"] + " (own windows)")
    if res.get("model_windows"):
        mw = res["model_windows"]
        ax.plot([(w["start_s"] + w["end_s"]) / 2 for w in mw], [w["p_fake"] for w in mw], color="#8a887f",
                linestyle="--", linewidth=1, label="file-level windows (4 s, hop 1 s)")
    ps = res["per_second"]
    ax.plot([(w["start_s"] + w["end_s"]) / 2 for w in ps],
            [np.nan if w["p_fake"] is None else w["p_fake"] for w in ps], color=LINE, linewidth=2.2,
            label="local windows")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(fake)", color=INK_2)
    ax.set_xlabel("time (s)", color=INK_2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    ax.set_title(f"Score over time: {res['verdict']} (P(fake) = {res['p_fake']:.2f})", loc="left",
                 color=INK, fontsize=10)
    for a_ in axes:
        a_.set_facecolor(SURFACE)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    axes[2].grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_xlim(0, max(dur, 1e-3))
    fig.tight_layout()
    return fig


def explain(path: str, run: str | None = None, out: str = "explanations", device: str = "cpu") -> dict:
    from adfd.registry import default_run

    d = Path(out) / Path(path).stem
    d.mkdir(parents=True, exist_ok=True)
    from adfd.single import MODEL_RUN, SingleModelAnalyzer

    if run is None or Path(run).name == MODEL_RUN:
        ens = SingleModelAnalyzer(device=device)
        x = Analyzer.load_audio(path)
        res = ens.analyze(x)
        plot_forensic(x, res).savefig(d / "forensic.png", dpi=140, facecolor=SURFACE)
        plot_score_line(res).savefig(d / "score_line.png", dpi=140, facecolor=SURFACE)
        res.pop("saliency", None)
        (d / "verdict.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        (d / "explanation.txt").write_text("\n".join(res["explanation"]["lines"]), encoding="utf-8")
        res["output_dir"] = str(d)
        return res
    an = Analyzer(run or default_run(), device)
    x = an.load_audio(path)
    res = an.analyze(x)
    (d / "verdict.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    plot_timeline(res).savefig(d / "timeline.png", dpi=140, facecolor=SURFACE)
    top = max(res["timeline"], key=lambda w: w["p_fake"])
    a, b = int(top["start_s"] * SR), int(top["end_s"] * SR)
    seg = an._prep(x[a:b])
    plot_heatmap(seg, an.saliency(x[a:b]), (b - a) / SR).savefig(d / "heatmap.png", dpi=140, facecolor=SURFACE)
    res["output_dir"] = str(d)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--run")
    ap.add_argument("--out", default="explanations")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from adfd.paths import configure_caches

    configure_caches()
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    res = explain(args.audio, args.run, args.out, args.device)
    print(f"Verdict: {res['verdict']}   P(fake) = {res['p_fake']:.3f}   model: {res['model']}")
    if res["out_of_distribution"]:
        print("Warning: this does not look like normal speech: " + "; ".join(res["ood_reasons"]))
    for w in res["suspicious_segments"]:
        print(f"  suspicious {w['start_s']:.1f}-{w['end_s']:.1f} s  P(fake) = {w['p_fake']:.2f}")
    for line in res.get("explanation", {}).get("lines", []):
        print("  " + line)
    print(f"Saved: {res['output_dir']}")


if __name__ == "__main__":
    main()
