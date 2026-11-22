"""P4.1-P4.3: fit calibration, three-way thresholds and OOD checks for one model (dev data only),
then report them on the test sets.

  calibration    Platt scaling on clean dev (+ degraded dev, when scored) spoof logits
  three-way      Real / Uncertain / Fake thresholds (<= 5% of each class wrong on that dev data)
  signal OOD     acceptable feature ranges from 1,000 bona fide dev recordings (shared by all models)
  embedding OOD  Mahalanobis distance on the penultimate layer, fitted on 1,000 dev clips
Reported: ECE before/after and reliability diagrams on dev, eval, In-the-Wild and
DEEP-VOICE; three-way coverage and errors; OOD flag rates on real test speech (false
alarms) and on synthetic non-speech (tones, noise, hum, music-like chords, silence).

Usage:
    python scripts/fit_reliability.py --run runs/e4a_rawboost_s42 [--device cpu]
Writes <run>/reliability.json (used by the app, API and explain.py) and
experiments/reliability/<run>.{json,md,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adfd.calibration import ece, fit_calibration, reliability, sigmoid, three_way_stats
from adfd.data.audio_cache import AudioCache
from adfd.inference import SR, Analyzer
from adfd.ood import EmbeddingOOD, check_signal, fit_signal_ranges, signal_features
from adfd.paths import REPO_ROOT, configure_caches, load_paths

EXP = REPO_ROOT / "experiments" / "reliability"
CONDS = ["clean", "mp3_51k", "mp3_19k", "opus_84k", "opus_10k", "telephone", "white_20db", "white_10db",
         "white_5db", "pink_10db"]
TEST_SETS = [("ASVspoof dev", "scores_asvspoof2019_la_dev.csv"), ("ASVspoof eval", "scores_asvspoof2019_la_eval.csv"),
             ("In-the-Wild", "scores_in_the_wild_test.csv"), ("DEEP-VOICE (windows)", "scores_deep_voice_test.csv")]


def synthetic(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(4 * SR) / SR
    pink = np.fft.irfft(np.fft.rfft(rng.standard_normal(len(t))) / np.sqrt(np.arange(1, len(t) // 2 + 2)), n=len(t))
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0)) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    hum = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in range(1, 6))
    clicks = np.zeros_like(t)
    clicks[:: SR // 4] = 1.0
    sig = {"tone 440 Hz": np.sin(2 * np.pi * 440 * t), "tone 1 kHz": np.sin(2 * np.pi * 1000 * t),
           "white noise": rng.standard_normal(len(t)), "pink noise": pink, "50 Hz hum": hum,
           "music-like chord": chord, "click train": clicks, "near silence": 1e-4 * rng.standard_normal(len(t))}
    return {k: (0.3 * v / (np.abs(v).max() + 1e-12)).astype(np.float32) for k, v in sig.items()}


def sample_recordings(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--native", action="store_true",
                    help="keep the checkpoint's own sigmoid and a 0.5 threshold (external models such as "
                         "VoiceGuard, which ASVspoof dev cannot calibrate); OOD checks are still fitted")
    args = ap.parse_args()
    paths = load_paths()
    configure_caches(paths)
    run = Path(args.run)
    an = Analyzer(run, args.device)
    an.calibration = an.emb_ood = None                       # fit from scratch
    report = {"run": run.name}

    # --- P4.1 / P4.2 calibration and thresholds ----------------------------------------
    dev = pd.read_csv(run / "scores_asvspoof2019_la_dev.csv", keep_default_na=False)
    parts, used = [dev], ["clean dev"]
    rd = [run / f"scores_asvspoof2019_la_robustdev_{c}_test__robustdev.csv" for c in CONDS]
    if all(f.exists() for f in rd):
        parts += [pd.read_csv(f, keep_default_na=False) for f in rd]
        used.append("10 degraded dev conditions")
    fit_df = pd.concat(parts, ignore_index=True)
    cal = fit_calibration(-fit_df["score"].to_numpy(), fit_df["label"].to_numpy(), " + ".join(used))
    if args.native:
        from adfd.calibration import Calibration

        cal = Calibration(1.0, 0.0, 0.5, 0.5, 0.5, "native sigmoid of the checkpoint (not refitted)", 0.0)
    report["calibration"] = cal.to_dict()
    report["sets"] = {}
    curves = {}
    for name, f in TEST_SETS:
        fp = run / f
        if not fp.exists():
            continue
        d = pd.read_csv(fp, keep_default_na=False)
        z, y = -d["score"].to_numpy(), d["label"].to_numpy()
        p_raw, p_cal = sigmoid(z), cal.prob_fake(z)
        report["sets"][name] = {"n": len(d), "ece_raw_pct": ece(p_raw, y), "ece_calibrated_pct": ece(p_cal, y),
                                "three_way": three_way_stats(p_cal, y, cal)}
        curves[name] = (reliability(p_raw, y), reliability(p_cal, y))

    # --- P4.3 OOD: signal ranges (shared), embedding distance (per model) ---------------
    asv = pd.read_csv(paths.manifest("asvspoof2019_la"), keep_default_na=False)
    cache = AudioCache(paths.cache_dir, "asvspoof2019_la")
    rng_f = EXP / "signal_ranges.json"
    if rng_f.exists():
        ranges = {k: tuple(v) for k, v in json.loads(rng_f.read_text(encoding="utf-8"))["ranges"].items()}
    else:
        bona = sample_recordings(asv[(asv["split"] == "dev") & (asv["label"] == 0)], 1000, 0)
        feats = [signal_features(cache.read(r) / 32768.0) for r in bona["recording_id"]]
        ranges = fit_signal_ranges(feats)
        EXP.mkdir(parents=True, exist_ok=True)
        rng_f.write_text(json.dumps({"fitted_on": "1,000 bona fide ASVspoof dev recordings", "ranges": ranges},
                                    indent=2), encoding="utf-8")
    report["signal_ranges"] = ranges

    devs = sample_recordings(asv[asv["split"] == "dev"], args.n_dev, 1)
    segs = []
    for r in devs["recording_id"]:
        x = cache.read(r) / 32768.0
        s, e = an.region(x)
        mid = s + max(0, (e - s - an.crop) // 2)
        segs.append(an._prep(x[mid:min(e, mid + an.crop)].astype(np.float32)))
    _, emb = an._score(segs)
    eo = EmbeddingOOD.fit(emb)
    report["embedding_ood"] = {"threshold": eo.threshold, "dim": int(emb.shape[1]), "fitted_on": f"{len(segs)} dev clips"}

    # false alarms on real test speech, and detection of synthetic non-speech
    an.signal_ranges, an.emb_ood, an.calibration = ranges, eo, cal
    fa = {}
    itw = pd.read_csv(paths.manifest("in_the_wild"), keep_default_na=False)
    ev = asv[asv["split"] == "eval"]
    for name, df, ds in (("ASVspoof eval, real", ev[ev["label"] == 0], "asvspoof2019_la"),
                         ("ASVspoof eval, fake", ev[ev["label"] == 1], "asvspoof2019_la"),
                         ("In-the-Wild, real", itw[itw["label"] == 0], "in_the_wild"),
                         ("In-the-Wild, fake", itw[itw["label"] == 1], "in_the_wild")):
        c = AudioCache(paths.cache_dir, ds)
        smp = sample_recordings(df, 200, 2)
        flags = {"signal": 0, "embedding": 0, "any": 0}
        for r in smp["recording_id"]:
            res = an.analyze((c.read(r) / 32768.0).astype(np.float32))
            sig = bool(check_signal(res["signal"], ranges))
            embf = res["embedding_distance"] > eo.threshold
            flags["signal"] += sig
            flags["embedding"] += embf
            flags["any"] += sig or embf
        fa[name] = {k: 100 * v / len(smp) for k, v in flags.items()} | {"n": len(smp)}
    report["ood_flag_pct_on_test_speech"] = fa
    synth = {}
    for name, x in synthetic().items():
        res = an.analyze(x)
        synth[name] = {"flagged": res["out_of_distribution"], "reasons": res["ood_reasons"],
                       "p_fake_raw": float(sigmoid(res["logit"])), "p_fake_calibrated": res["p_fake"],
                       "embedding_distance": res["embedding_distance"]}
    report["synthetic"] = synth

    # --- save ---------------------------------------------------------------------------
    rel = {"calibration": cal.to_dict(), "signal_ranges": ranges, "embedding_ood": eo.to_dict()}
    (run / "reliability.json").write_text(json.dumps(rel), encoding="utf-8")
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / f"{run.name}.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    plot(curves, EXP / f"{run.name}.png", run.name)
    md = markdown(report)
    (EXP / f"{run.name}.md").write_text(md, encoding="utf-8")
    sys.stdout.buffer.write(md.encode("utf-8"))
    return 0


def markdown(r: dict) -> str:
    c = r["calibration"]
    L = [f"# Reliability: {r['run']}", "",
         f"Calibration (Platt, balanced classes) fitted on {c['fitted_on']}: p_fake = sigmoid({c['a']:.3f}·z + {c['b']:.3f}). "
         f"Verdict: Real if p_fake ≤ {c['t_real']:.3f}, Fake if p_fake ≥ {c['t_fake']:.3f}, else Uncertain "
         f"(≤ {c['target_error_pct']:.0f}% of each dev class on the wrong side).", "",
         "| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | "
         "real → Fake % | fake → Real % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in r["sets"].items():
        t = s["three_way"]
        L.append(f"| {k} | {s['n']:,} | {s['ece_raw_pct']:.2f} | {s['ece_calibrated_pct']:.2f} | {t['pct_real']:.1f} | "
                 f"{t['pct_uncertain']:.1f} | {t['pct_fake']:.1f} | {t['error_pct_decided']:.2f} | "
                 f"{t['real_called_fake_pct']:.2f} | {t['fake_called_real_pct']:.2f} |")
    L += ["", f"Reliability diagrams: `experiments/reliability/{r['run']}.png`.", "",
          "## Out-of-distribution check", "",
          f"Embedding check: Mahalanobis distance on the {r['embedding_ood']['dim']}-d penultimate layer, threshold "
          f"{r['embedding_ood']['threshold']:.2f} (99th percentile of {r['embedding_ood']['fitted_on']}). "
          "Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).", "",
          "Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):", "",
          "| Set | n | signal check % | embedding check % | either % |", "|---|---|---|---|---|"]
    for k, v in r["ood_flag_pct_on_test_speech"].items():
        L.append(f"| {k} | {v['n']} | {v['signal']:.1f} | {v['embedding']:.1f} | {v['any']:.1f} |")
    L += ["", "Synthetic non-speech (should be flagged):", "",
          "| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |", "|---|---|---|---|---|---|"]
    for k, v in r["synthetic"].items():
        L.append(f"| {k} | {'yes' if v['flagged'] else '**no**'} | {'; '.join(v['reasons']) or '–'} | "
                 f"{v['p_fake_raw']:.3f} | {v['p_fake_calibrated']:.3f} | {v['embedding_distance']:.1f} |")
    return "\n".join(L) + "\n"


def plot(curves: dict, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e4e3df", "#fcfcfb"
    RAW, CAL = "#eb6834", "#2a78d6"
    fig, axes = plt.subplots(1, len(curves), figsize=(3.4 * len(curves), 3.6), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, (raw, calc)) in zip(axes, curves.items()):
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        for rows, color, marker, lab in ((raw, RAW, "o", "raw sigmoid"), (calc, CAL, "s", "calibrated")):
            xs = [r[0] for r in rows if not np.isnan(r[1])]
            ys = [r[1] for r in rows if not np.isnan(r[1])]
            ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("predicted P(fake)", color=INK_2)
        ax.grid(color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("observed fraction fake", color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Reliability diagrams: {title}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
