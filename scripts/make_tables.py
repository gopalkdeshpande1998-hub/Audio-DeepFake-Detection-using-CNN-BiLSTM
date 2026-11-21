"""Build the Phase 2 result tables from experiments/*/metrics.json.

Usage:
    python scripts/make_tables.py
Writes experiments/tables/phase2.md and the robustness figures
experiments/tables/robustness_{eval,dev}.png. Every row averages all seeds of a
model (mean ± std, n shown). Rows whose runs have not finished are skipped.
Model choices use ASVspoof dev data only: the feature by clean dev EER, the
augmentation by mean EER on the degraded dev subset. Test columns are reported,
never used for selection.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from adfd.data.degrade import CONDITIONS
from adfd.paths import REPO_ROOT

EXP = REPO_ROOT / "experiments"
DEV, EVAL, ITW, DV = "asvspoof2019_la:dev", "asvspoof2019_la:eval", "in_the_wild:test", "deep_voice:test"

FEATURE_RUNS = [  # (label, run folder prefix; seeds are averaged when several exist)
    ("MFCC-40 (E1, paper feature)", "e1_cnn_bilstm_mfcc"),
    ("LFCC-60", "e2a_lfcc"),
    ("CQCC-60", "e2b_cqcc"),
    ("Log-mel-80", "e2c_logmel"),
    ("Sinc filterbank (raw waveform)", "e2d_sinc"),
]
AUG_RUNS = [  # filled in once the augmentation runs exist
    ("LFCC-60, trimmed, no augmentation (E3)", "e3_trimmed"),
    ("+ RawBoost", "e4a_rawboost"),
    ("+ Codec", "e4b_codec"),
    ("+ RawBoost + codec", "e4c_rawboost_codec"),
    ("+ RawBoost + codec + resample + speed", "e4d_full_aug"),
]
ROBUST_MODELS = [  # (label, prefix, marker) for the degraded-set tables and figures
    ("E1 MFCC, untrimmed", "e1_cnn_bilstm_mfcc", "o"),
    ("E2a LFCC, untrimmed", "e2a_lfcc", "s"),
    ("E3 LFCC, trimmed", "e3_trimmed", "^"),
    ("E4a + RawBoost", "e4a_rawboost", "v"),
    ("E4b + codec", "e4b_codec", "<"),
    ("E4c + RawBoost + codec", "e4c_rawboost_codec", "D"),
    ("E4d + RawBoost + codec + resample + speed", "e4d_full_aug", "P"),
]
# Validated categorical palette (dataviz reference, light surface), assigned in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e4e3df", "#fcfcfb"


def load(run: str) -> dict | None:
    f = EXP / run / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def seeds(prefix: str, tag: str = "") -> list[dict]:
    out = []
    for d in sorted(EXP.glob(f"{prefix}_s*")):
        name = d.name
        if tag and not name.endswith(f"__{tag}"):
            continue
        if not tag and "__" in name:
            continue
        m = load(name)
        if m:
            out.append(m)
    return out


def cell(ms: list[dict], s: str, key="eer_pct", digits=2) -> str:
    vals = []
    for m in ms:
        if s in m["sets"]:
            v = m["sets"][s]
            v = v["per_recording"]["eer_pct"] if key == "per_rec" else v.get(key)
            if v is not None:
                vals.append(v)
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"


def results_table(rows, tag="") -> list[str]:
    lines = ["| Input / setting | seeds | dev EER | eval EER | eval min t-DCF | In-the-Wild EER | DEEP-VOICE EER (window / recording) |",
             "|---|---|---|---|---|---|---|"]
    for label, prefix in rows:
        if prefix is None:
            continue
        ms = seeds(prefix, tag)
        if not ms:
            continue
        lines.append(f"| {label} | {len(ms)} | {cell(ms, DEV)} | {cell(ms, EVAL)} | {cell(ms, EVAL, 'min_tdcf', 3)} | "
                     f"{cell(ms, ITW)} | {cell(ms, DV)} / {cell(ms, DV, 'per_rec')} |")
    return lines


FAMILIES = [  # (panel title, x labels, conditions in order) for the robustness figures
    ("MP3", ["clean", "51 kbps", "19 kbps"], ["clean", "mp3_51k", "mp3_19k"]),
    ("Opus", ["clean", "84 kbps", "10 kbps"], ["clean", "opus_84k", "opus_10k"]),
    ("White noise", ["clean", "20 dB", "10 dB", "5 dB"], ["clean", "white_20db", "white_10db", "white_5db"]),
    ("Other channels", ["clean", "telephone", "pink 10 dB"], ["clean", "telephone", "pink_10db"]),
]
PLOT_MODELS = ["e1_cnn_bilstm_mfcc", "e2a_lfcc", "e3_trimmed"]  # plus the selected augmentation


def select_augmentation(dev_rows: dict):
    """(prefix, mean EER) of the E4 run with the lowest mean degraded-dev EER, averaged over seeds."""
    cands = {}
    for p, (_, _, ps) in dev_rows.items():
        if p.startswith("e4"):
            v = ps.mean(axis=1)
            v = v[~np.isnan(v)]
            if len(v):
                cands[p] = float(v.mean())
    if not cands:
        return None
    best = min(cands, key=cands.get)
    return best, cands[best]


def plot_robustness(degraded: dict) -> list[str]:
    """Small multiples: EER against degradation level, one line per headline model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(CONDITIONS)
    best = select_augmentation(degraded.get("robustdev", {}))
    models = PLOT_MODELS + ([best[0]] if best else [])
    written = []
    for tag, fname, title in (("robust", "robustness_eval.png", "Degraded ASVspoof eval subset (reported only)"),
                              ("robustdev", "robustness_dev.png", "Degraded ASVspoof dev subset (used for selection)")):
        rows = degraded.get(tag, {})
        present = [m for m in models if m in rows]
        if not present:
            continue
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                             "xtick.color": INK_2, "ytick.color": INK_2})
        fig, axes = plt.subplots(1, len(FAMILIES), figsize=(13, 4.0), sharey=True, facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [len(f[2]) for f in FAMILIES]})
        for ax, (fam, xl, cs) in zip(axes, FAMILIES):
            ax.set_facecolor(SURFACE)
            idx = [conds.index(c) for c in cs]
            x = np.arange(len(cs))
            for m in present:
                label, marker, ps = rows[m]
                vals = ps[:, idx]
                mean = np.nanmean(vals, axis=0)
                sd = np.nanstd(vals, axis=0, ddof=1) if len(ps) > 1 else None
                ax.errorbar(x, mean, yerr=sd, color=SERIES[models.index(m)], marker=marker, markersize=8, linewidth=2,
                            linestyle="-" if fam != "Other channels" else "none", elinewidth=1, capsize=3,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label} (n={len(ps)})", zorder=3)
            ax.set_xticks(x, xl)
            ax.set_xlim(-0.4, len(cs) - 0.6)
            ax.set_title(fam, color=INK, fontsize=11, loc="left")
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            ax.axhline(50, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel("EER (%)")
        axes[0].set_ylim(0, 75)
        axes[0].text(-0.35, 51, "chance", color=INK_2, fontsize=9, ha="left", va="bottom")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=SERIES[models.index(m)], marker=rows[m][1], markersize=8, linewidth=2,
                          markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{rows[m][0]} (n={len(rows[m][2])})")
                   for m in present]
        fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.995, 1.0), labelcolor=INK)
        fig.suptitle(title, x=0.01, y=0.975, ha="left", color=INK, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = EXP / "tables" / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        written.append(f"experiments/tables/{fname}")
    return written


def main() -> int:
    out = ["# Phase 2 results", "",
           "EER in %. Mean ± std when several seeds exist. Every choice (best feature, best augmentation) "
           "is made on **ASVspoof dev EER only**; the test columns are reported, never used for selection.", ""]

    out += ["## P2.1 Input-feature ablation (same CNN-BiLSTM, untrimmed audio)", ""] + results_table(FEATURE_RUNS)

    out += ["", "## P2.2 Silence shortcut", ""]
    ctrl = EXP / "shortcuts" / "silence_control.md"
    if ctrl.exists():
        out += ["### Control: classifiers that see only silence and duration", ""]
        out += [l for l in ctrl.read_text(encoding="utf-8").splitlines()[5:] if l.strip()]
    out += ["", "### Models scored on silence-trimmed test audio", "",
            "| Model | trained on | scored on | seeds | dev EER | eval EER | eval min t-DCF | In-the-Wild EER |",
            "|---|---|---|---|---|---|---|---|"]
    untrimmed_tags = (("", "untrimmed"), ("trimmed", "trimmed"))
    for label, prefix, trained, tags in [
            ("E1 MFCC", "e1_cnn_bilstm_mfcc", "untrimmed", untrimmed_tags),
            ("E2a LFCC", "e2a_lfcc", "untrimmed", untrimmed_tags),
            ("E3 LFCC", "e3_trimmed", "trimmed", (("untrimmed", "untrimmed"), ("", "trimmed")))]:
        for tag, scored in tags:
            ms = seeds(prefix, tag)
            if ms:
                out.append(f"| {label} | {trained} | {scored} | {len(ms)} | {cell(ms, DEV)} | {cell(ms, EVAL)} | "
                           f"{cell(ms, EVAL, 'min_tdcf', 3)} | {cell(ms, ITW)} |")

    aug = results_table(AUG_RUNS)
    if len(aug) > 2:
        out += ["", "## P2.3 Augmentation ablation (LFCC-60, silence-trimmed)", ""] + aug

    conds = list(CONDITIONS)
    degraded = {}
    for tag, prefix, title, note in (
            ("robustdev", "asvspoof2019_la_robustdev_",
             "P2.3 Augmentation selection: EER (%) on a degraded ASVspoof **dev** subset",
             "Subset: 2,000 real + 770 per attack (A01–A06) from dev. Selection metric = mean over the 10 "
             "conditions (averaged over seeds); this is the only table used to choose the augmentation."),
            ("robust", "asvspoof2019_la_robust_",
             "P2.4 Robustness: EER (%) on a degraded ASVspoof eval subset",
             "Subset: 2,000 real + 770 per attack (A07–A19) from eval, identical audio for every model. "
             "Reported only.")):
        rows = {}
        for label, mprefix, marker in ROBUST_MODELS:
            ms = seeds(mprefix, tag)
            if not ms:
                continue
            per_seed = np.array([[m["sets"].get(f"{prefix}{c}:test", {}).get("eer_pct", np.nan) for c in conds]
                                 for m in ms], dtype=float)
            rows[mprefix] = (label, marker, per_seed)
        degraded[tag] = rows
        if not rows:
            continue
        out += ["", f"## {title}", "", note, "",
                "| Model | seeds | " + " | ".join(conds) + " | mean |", "|---|---|" + "---|" * (len(conds) + 1)]
        for mprefix, (label, _, ps) in rows.items():
            means = np.nanmean(ps, axis=0)
            row_means = ps.mean(axis=1)  # NaN if a seed misses a condition
            ok = ~np.isnan(row_means)
            if ok.any():
                mean_txt = f"**{row_means[ok].mean():.2f}**"
                if ok.sum() > 1:
                    mean_txt += f" ± {row_means[ok].std(ddof=1):.2f}"
            else:
                mean_txt = "–"
            out.append(f"| {label} | {len(ps)} | " + " | ".join("–" if np.isnan(v) else f"{v:.2f}" for v in means)
                       + f" | {mean_txt} |")
        if tag == "robustdev":
            best = select_augmentation(rows)
            if best:
                out += ["", f"**Selected augmentation: {rows[best[0]][0]}** (lowest mean degraded-dev EER, "
                            f"{best[1]:.2f}%, over {len(rows[best[0]][2])} seeds)."]
    if any(EXP.glob("*__robust*")):
        out += ["", "Conditions: " + "; ".join(f"`{c}` = {d}" for c, (d, _) in CONDITIONS.items())]
        figs = plot_robustness(degraded)
        if figs:
            out += ["", "Figures: " + ", ".join(f"`{f}`" for f in figs) + " (mean over seeds, bars = ± 1 std)."]

    for f, title in (("length_check.md", "P2.2 Length shortcut"), ("loudness_check.md", "P2.2 Loudness shortcut")):
        fp = EXP / "shortcuts" / f
        if fp.exists():
            body = fp.read_text(encoding="utf-8").splitlines()[1:]
            out += ["", f"## {title} (from `experiments/shortcuts/{f}`)"]
            out += [line.replace("## ", "### ", 1) for line in body]

    (EXP / "tables").mkdir(parents=True, exist_ok=True)
    text = "\n".join(out) + "\n"
    (EXP / "tables" / "phase2.md").write_text(text, encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
