"""Detector views: single-file analysis, batch scoring and the session history.

Imported only by the Detect and Batch pages, so the research pages open without loading PyTorch.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from adfd import registry, trust
from adfd.explain import plot_forensic, plot_heatmap, plot_score_line, plot_timeline
from adfd.inference import SR, load_audio
from adfd.paths import load_paths
from adfd.segments import SEVERITY_TEXT, fmt_ts
from ui import lab, signal
from ui.icons import icon
from ui.theme import (AMBER, BLUE, CYAN, GREEN, GREY, RED, VERDICT_COLOR, VERDICT_WORD, bar, chip, esc, file_card,
                      fmt_dur, fmt_size, loader_html, note, results_skeleton, ring, section, stepper_html, verdict_chip,
                      warn_html)

log = logging.getLogger("adfd.app")

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a", "aiff", "aif", "webm", "opus"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "avi", "webm"]
MAX_MB = 50
MAX_SECONDS = 600                                  # longer files: the first 10 minutes are analysed
DECISION = 50.0                                    # % : FAKE at or above, REAL below
BORDER = 10.0                                      # amber band +-10 around the line: visual reference only
LABEL_COLOR = {"Real": GREEN, "Fake": RED, "No speech": GREY}
LIMITS = ("Research prototype. The final verdict is always REAL or FAKE: the weighted combination of three models, "
          "decided at 50%. Every verdict can be wrong. On held-out tests about 1 in 9 real studio recordings "
          "(ASVspoof) are called FAKE, 1 in 5 ASVspoof fakes and most voice-conversion fakes (DEEP-VOICE) are "
          "called REAL; see the Error analysis page. Flagged regions are model-flagged suspicions, not proof of "
          "manipulation. Never use one verdict as evidence on its own.")


def gpu_available() -> bool:
    import torch

    return torch.cuda.is_available()


def default_device() -> str:
    """ADFD_DEVICE if set, else the GPU when PyTorch can see one, else the CPU."""
    import os

    env = os.environ.get("ADFD_DEVICE")
    if env:
        return env
    return "cuda" if gpu_available() else "cpu"


def device() -> str:
    return st.session_state.get("device") or default_device()


def device_label(dev: str | None = None) -> str:
    dev = dev or device()
    if dev.startswith("cuda") and gpu_available():
        import torch

        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


@st.cache_resource(show_spinner="Loading models…", max_entries=3)
def _load_analyzer(run: str, dev: str):
    return registry.make_analyzer(run, dev)


def get_analyzer(run: str):
    dev = device()
    try:
        return _load_analyzer(run, dev)
    except RuntimeError as e:                     # CUDA out of memory or driver error: stay usable on the CPU
        if dev == "cpu":
            raise
        log.warning("loading %s on %s failed (%r); falling back to the CPU", run, dev, e)
        _free_gpu()
        st.session_state["device"] = "cpu"
        st.warning("The models did not fit on the GPU, so they run on the CPU instead (slower).", icon="⚠️")
        return _load_analyzer(run, "cpu")


def _free_gpu() -> None:
    _load_analyzer.clear()
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_analysis(run: str, an, x: np.ndarray, **kw) -> dict:
    """an.analyze(x) with a CPU retry when the GPU runs out of memory mid-analysis."""
    import torch

    try:
        return an.analyze(x, **kw)
    except torch.OutOfMemoryError:
        log.warning("CUDA out of memory; retrying on the CPU")
        _free_gpu()
        st.session_state["device"] = "cpu"
        st.warning("The GPU ran out of memory on this file, so it was analysed on the CPU instead.", icon="⚠️")
        kw.pop("progress", None)
        return _load_analyzer(run, "cpu").analyze(x, **kw)


def to_wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def fig_png(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def demo_files() -> dict[str, Path]:
    """DEEP-VOICE demonstration clips, if the dataset is on this machine (never shipped with the repo)."""
    try:
        root = load_paths().dataset("deep_voice")
    except Exception:
        return {}
    for base in [root, *root.parents[:3]]:
        for d in base.rglob("DEMONSTRATION"):
            files = sorted(p for p in d.glob("*") if p.suffix.lower().lstrip(".") in AUDIO_TYPES)
            if files:
                return {p.name: p for p in files}
    return {}


# ---------------------------------------------------------------------------------------
# verdict, meter, members
# ---------------------------------------------------------------------------------------
def consensus(res: dict) -> tuple[int, int, str]:
    """(models sharing the final verdict, number of models, status word)."""
    ms = res.get("members") or []
    agree = sum(m["vote"] == res["verdict"] for m in ms)
    status = "Unanimous" if agree == len(ms) else "Majority" if 2 * agree > len(ms) else "Split"
    return agree, len(ms), status


def verdict_html(res: dict, source: str, elapsed: float | None = None, t: dict | None = None) -> str:
    """The large final-verdict panel: word, probability ring, agreement, trust, timing and the REAL/SYNTHETIC meter."""
    fake = res["verdict"] == "Fake"
    c = VERDICT_COLOR[res["verdict"]]
    tint = "rgba(248,113,113,.10)" if fake else "rgba(52,211,153,.09)"
    word = VERDICT_WORD[res["verdict"]]
    p_syn = 100 * res["p_fake"]
    p_ver = p_syn if fake else 100 - p_syn
    meaning = ("Likely synthetic or voice-converted speech." if fake else "Likely genuine (bona fide) speech.")
    facts = []
    if res.get("members"):
        agree, n, status = consensus(res)
        pips = "".join(f"<i style='--c:{VERDICT_COLOR[m['vote']]};--d:{.15 * i:.2f}s'></i>"
                       for i, m in enumerate(res["members"]))
        facts.append(("Model agreement", f"{agree}/{n} <span class='vg-pips'>{pips}</span>"))
        facts.append(("Consensus", chip(status, "real" if status == "Unanimous" else "amber",
                                        "check" if status == "Unanimous" else "alert")))
    if t:
        lvl = t["level"]
        kind = "real" if lvl in ("high", "very high") else "amber" if lvl == "medium" else "syn"
        facts.append(("Held-out trust", chip(f"{lvl} confidence", kind)))
    if elapsed is not None:
        facts.append(("Analysis time", f"{elapsed:.1f} s"))
    facts.append(("Duration", fmt_dur(res["duration_s"])))
    fx = "".join(f"<div><div class='fk'>{esc(k)}</div><div class='fv'>{v}</div></div>" for k, v in facts)
    ic = icon("shield-alert" if fake else "shield-check", 14)
    return (f"<div class='vg-verdict' style='--c:{c};--tint:{tint}'>"
            f"{ring(p_ver, c, 132, 'probability', 10)}"
            f"<div><div class='k'><span style='color:{c}'>{ic}</span>Final verdict · {esc(source)}</div>"
            f"<div class='w'>{word}</div><div class='d'>{meaning} The ensemble gives this recording a synthetic score "
            f"of <b>{p_syn:.1f}%</b>; it is called SYNTHETIC at {DECISION:.0f}% or more.</div>"
            f"{meter_html(res['p_fake'])}</div>"
            f"<div class='vg-facts'>{fx}</div></div>")


def meter_html(p_fake: float) -> str:
    fake = round(100 * p_fake, 1)
    real = round(100 - fake, 1)
    return (f"<div class='vg-meter'><div class='lab'><span style='color:{GREEN}'>REAL <b>{real:.1f}%</b></span>"
            f"<span style='color:{RED}'>SYNTHETIC <b>{fake:.1f}%</b></span></div>"
            f"<div class='vg-track' role='img' aria-label='Real {real:.1f} percent, synthetic {fake:.1f} percent'>"
            f"<div class='r' style='--w:{real}%'></div><div class='f' style='--w:{fake}%'></div>"
            f"<div class='mid' title='Decision line, 50%'></div></div></div>")


def member_cards(res: dict) -> None:
    """One card per ensemble member: ring, REAL / SYNTHETIC bars, its own call and whether it sides with the verdict."""
    ms = res["members"]
    cols = st.columns(len(ms), gap="medium")
    for i, (c, m) in enumerate(zip(cols, ms)):
        p = 100 * m["p_fake"]
        col = VERDICT_COLOR[m["vote"]]
        agrees = m["vote"] == res["verdict"]
        d = f"{.08 + .1 * i:.2f}s"
        status = (f"<span class='st' style='color:{GREEN}'><span class='vg-dot'></span>Agrees with verdict</span>"
                  if agrees else f"<span class='st' style='color:{AMBER}'><span class='vg-dot'></span>Dissents</span>")
        bars = "".join(f"<div class='vg-mbar'><div class='l'><span>{lb}</span><b>{v:.1f}%</b></div>"
                       f"<div class='t'><div style='--w:{v:.1f}%;--c:{cc};--d:{d}'></div></div></div>"
                       for lb, v, cc in (("REAL", 100 - p, GREEN), ("SYNTHETIC", p, RED)))
        with c:
            st.markdown(
                f"<div class='vg-model' style='--d:{d}'><div class='top'><div><div class='nm'>{esc(m['label'])}</div>"
                f"<div class='sub'>Weight {100 * m['weight']:.0f}% in the ensemble · {len(m.get('timeline') or [])} "
                f"window(s) scored</div></div>{verdict_chip(m['vote'])}</div>"
                f"<div class='mid'>{ring(p, col, 76, 'synthetic', 7, small=True)}<div class='bars'>{bars}</div></div>"
                f"<div class='foot'>{status}<span>Own call at 50%</span></div></div>", unsafe_allow_html=True)


def comparison_html(res: dict) -> str:
    """REAL vs SYNTHETIC share of every model and of the ensemble, on one 100% scale with the 50% line."""
    rows = [(m["label"], f"weight {100 * m['weight']:.0f}%", m["p_fake"], False) for m in res["members"]]
    rows.append(("Ensemble", "final verdict", res["p_fake"], True))
    out = ["<div class='vg-legend2' style='margin-bottom:14px'>"
           f"<span><i style='background:{GREEN}'></i>REAL score</span>"
           f"<span><i style='background:{RED}'></i>SYNTHETIC score</span>"
           "<span><i style='background:transparent;border-left:2px dashed #cbd5e1;border-radius:0'></i>"
           "50% decision line</span></div><div class='vg-cmp'>"]
    for i, (nm, sub, p, ens) in enumerate(rows):
        f, r = 100 * p, 100 - 100 * p
        d = f"{.1 * i:.1f}s"
        lab_r = f"{r:.0f}%" if r >= 9 else ""
        lab_f = f"{f:.0f}%" if f >= 9 else ""
        out.append(f"<div class='row{' ens' if ens else ''}'><div class='nm'>{esc(nm)}<span>{esc(sub)}</span></div>"
                   f"<div class='bar'><div class='seg r' style='--w:{r:.2f}%;--d:{d}' "
                   f"data-tip='{esc(nm)} · REAL {r:.1f}%'>{lab_r}</div>"
                   f"<div class='seg f' style='--w:{f:.2f}%;--d:{d}' data-tip='{esc(nm)} · SYNTHETIC {f:.1f}%'>"
                   f"{lab_f}</div><div class='thr'></div></div></div>")
    out.append("<div class='axis'><span></span><div><span>0%</span><span>25%</span><span>50%</span><span>75%</span>"
               "<span>100%</span></div></div></div>")
    return "".join(out)


def models_panel(res: dict) -> None:
    agree, n, status = consensus(res)
    section("Three-model AI verdict", "Each model scores the whole file on its own; the ensemble weighs all three.",
            "layers", f"{agree} of {n} models agree · {status.lower()}")
    member_cards(res)
    if status != "Unanimous":
        diss = [m for m in res["members"] if m["vote"] != res["verdict"]]
        names = "; ".join(f"<b style='color:inherit'>{esc(m['label'])}</b> calls it {VERDICT_WORD[m['vote']]} on its own "
                          f"({100 * m['p_fake']:.0f}% synthetic)" for m in diss)
        st.markdown(warn_html("The models disagree on this recording",
                              f"{names}, while the ensemble says {VERDICT_WORD[res['verdict']]}. On held-out data, "
                              "verdicts where the models disagree were right less often; check the trust panel and "
                              "listen to the flagged regions before relying on it."), unsafe_allow_html=True)


TRUST_SETS = {"ASVspoof eval": "Studio TTS, unseen attacks", "In-the-Wild": "Internet audio (partly seen by one model)"}


def trust_of(res: dict) -> dict:
    return trust.assess(res["p_fake"], [m["p_fake"] for m in res.get("members", [])] or [res["p_fake"]])


def trust_html(t: dict) -> str:
    col = {"very low": RED, "low": RED, "medium": AMBER, "high": GREEN, "very high": GREEN}[t["level"]]
    bars = ""
    for name, sub in TRUST_SETS.items():
        s = t["sets"].get(name)
        if s:
            c = GREEN if s["accuracy_pct"] >= 90 else AMBER if s["accuracy_pct"] >= 70 else RED
            bars += bar(f"{sub} · n = {s['n']:,}", s["accuracy_pct"], c)
    basis = next((s["basis"] for s in t["sets"].values()), None)
    weak = [(TRUST_SETS.get(k, k), v["accuracy_pct"]) for k, v in t["sets"].items() if v["accuracy_pct"] < 50]
    advice = t["advice"]
    if weak:
        advice += " But on " + ", ".join(f"{n[0].lower() + n[1:]} ({a:.0f}% right)" for n, a in weak) + (
            " verdicts like this were usually wrong, so if the recording could be of that kind, do not rely on it.")
    body = (f"<div style='font-size:.85rem;margin:8px 0 4px'>How often verdicts like this were right on held-out "
            f"test data ({basis}: {t['agree']} of {t['n_members']} models agreeing, {t['level']} confidence):</div>"
            + bars) if bars else ("<div style='font-size:.85rem;margin-top:6px'>Run <code>python "
                                  "scripts/error_analysis.py</code> to add held-out accuracy here.</div>")
    return (f"<div class='adfd-trust'><div style='font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;"
            f"opacity:.75'>How far to trust it</div><div class='lvl' style='color:{col}'>{t['level'].capitalize()} "
            f"confidence</div><div style='font-size:.92rem'>{html.escape(advice)}</div>{body}</div>")


def trust_panel_html(t: dict) -> str:
    """On-screen trust panel: the same lookup as trust_html, in the workstation style."""
    lvl = t["level"]
    col = {"very low": RED, "low": RED, "medium": AMBER, "high": GREEN, "very high": GREEN}[lvl]
    bars = ""
    for name, sub in TRUST_SETS.items():
        s = t["sets"].get(name)
        if s:
            c = GREEN if s["accuracy_pct"] >= 90 else AMBER if s["accuracy_pct"] >= 70 else RED
            bars += bar(f"{sub} · n = {s['n']:,}", s["accuracy_pct"], c)
    basis = next((s["basis"] for s in t["sets"].values()), None)
    weak = [(TRUST_SETS.get(k, k), v["accuracy_pct"]) for k, v in t["sets"].items() if v["accuracy_pct"] < 50]
    advice = t["advice"]
    if weak:
        advice += " But on " + ", ".join(f"{n[0].lower() + n[1:]} ({a:.0f}% right)" for n, a in weak) + (
            " verdicts like this were usually wrong, so if the recording could be of that kind, do not rely on it.")
    body = (f"<div style='font-size:.8rem;color:#8b98ad;margin:12px 0 6px'>How often verdicts like this were right "
            f"on held-out test data ({esc(basis)}; {t['agree']} of {t['n_members']} models agreeing):</div>{bars}"
            if bars else "<div style='font-size:.8rem;color:#8b98ad;margin-top:10px'>No held-out accuracy table on "
                         "this machine. Run <code>python scripts/error_analysis.py</code> to add it.</div>")
    return (f"<div class='vg-panel' style='height:100%'><div style='display:flex;align-items:center;gap:10px;"
            f"justify-content:space-between'><div style='display:flex;align-items:center;gap:10px;font-weight:600'>"
            f"<span style='color:{col}'>{icon('gauge', 18)}</span>How far to trust it</div>"
            f"<span class='vg-chip' style='color:{col}'>{esc(lvl)} confidence</span></div>"
            f"<div style='font-size:.88rem;color:#c4cdd9;margin-top:10px;line-height:1.55'>{esc(advice)}</div>"
            f"{body}</div>")


def legend_html() -> str:
    items = [(GREEN, "REAL: local synthetic score below 50%"),
             (RED, "SYNTHETIC: local synthetic score 50% or more (suspected region)"),
             (AMBER, f"amber band ({DECISION - BORDER:.0f}–{DECISION + BORDER:.0f}%): close to the line, shown for "
                     "reference only; still classified by the 50% line"),
             (GREY, "no speech: not scored")]
    return "<div class='adfd-legend'>" + "".join(
        f"<span><span class='sw' style='background:{c}'></span>{html.escape(t)}</span>" for c, t in items) + "</div>"


# ---------------------------------------------------------------------------------------
# time charts (Altair: tooltips and hover)
# ---------------------------------------------------------------------------------------
def step_label(p) -> str:
    return "No speech" if p is None else ("Fake" if 100 * p >= DECISION else "Real")


def waveform_chart(x: np.ndarray, res: dict):
    """Waveform envelope over the whole recording, each slice coloured by the local analysis."""
    import altair as alt

    n = len(x)
    nb = int(min(1600, max(200, n // 64)))
    edges = np.linspace(0, n, nb + 1).astype(int)
    lo = np.array([x[a:b].min() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    hi = np.array([x[a:b].max() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    t0, t1 = edges[:-1] / SR, edges[1:] / SR
    ps = res["per_second"]
    starts = np.array([w["start_s"] for w in ps])
    idx = np.clip(np.searchsorted(starts, (t0 + t1) / 2, side="right") - 1, 0, len(ps) - 1)
    p = [ps[i]["p_fake"] for i in idx]
    df = pd.DataFrame({"t0": t0, "t1": t1, "lo": lo, "hi": hi, "label": [step_label(q) for q in p],
                       "time": [fmt_ts((a + b) / 2) for a, b in zip(t0, t1)],
                       "fake_pct": [None if q is None else round(100 * q, 1) for q in p]})
    peak = float(max(np.abs(x).max(), 1e-3))
    dur = n / SR
    xs = alt.X("t0:Q", title="time (s)", scale=alt.Scale(domain=[0, dur], nice=False))
    ys = alt.Y("lo:Q", title="amplitude", scale=alt.Scale(domain=[-peak, peak]), axis=alt.Axis(labels=False))
    layers = []
    seg = pd.DataFrame([{"a": s["start_s"], "b": s["end_s"], "name": f"S{i}",
                         "span": f"{fmt_ts(s['start_s'])}–{fmt_ts(s['end_s'])}", "sev": s["severity"],
                         "p": round(100 * s["p_fake"], 1)} for i, s in enumerate(res.get("segments", []), 1)])
    if len(seg):
        layers.append(alt.Chart(seg).mark_rect(color=RED, opacity=0.08).encode(
            x="a:Q", x2="b:Q", tooltip=[alt.Tooltip("name:N", title="segment"), alt.Tooltip("span:N", title="time"),
                                         alt.Tooltip("p:Q", title="mean synthetic %"), alt.Tooltip("sev:N", title="severity")]))
    layers.append(alt.Chart(df).mark_bar(binSpacing=0).encode(
        x=xs, x2="t1:Q", y=ys, y2="hi:Q",
        color=alt.Color("label:N", scale=alt.Scale(domain=list(LABEL_COLOR), range=list(LABEL_COLOR.values())),
                        legend=None),
        tooltip=[alt.Tooltip("time:N", title="time"), alt.Tooltip("label:N", title="local analysis"),
                 alt.Tooltip("fake_pct:Q", title="local synthetic %")]))
    return alt.layer(*layers).properties(height=170)


def timeline_chart(res: dict, show_members: bool):
    """Merged fake probability (%) over time from the local windows, on REAL / FAKE zones."""
    import altair as alt

    ps = res["per_second"]
    dur = res["duration_s"]
    rows, run = [], 0
    for i, w in enumerate(ps):
        if w["p_fake"] is None:
            run += 1
        rows.append({"t": (w["start_s"] + w["end_s"]) / 2, "time": f"{fmt_ts(w['start_s'])}–{fmt_ts(w['end_s'])}",
                     "merged": None if w["p_fake"] is None else 100 * w["p_fake"], "run": run,
                     "here": step_label(w["p_fake"]),
                     **{m["label"]: (None if m["per_second"][i] is None else 100 * m["per_second"][i])
                        for m in res["members"]}})
    df = pd.DataFrame(rows)
    xs = alt.X("t:Q", title="time (s)", scale=alt.Scale(domain=[0, max(dur, 1e-3)], nice=False))
    ysc = alt.Scale(domain=[0, 100], nice=False)
    zones = pd.DataFrame([{"y0": DECISION + BORDER, "y1": 100, "c": RED, "z": "SYNTHETIC zone"},
                          {"y0": DECISION - BORDER, "y1": DECISION + BORDER, "c": AMBER, "z": "near the line"},
                          {"y0": 0, "y1": DECISION - BORDER, "c": GREEN, "z": "REAL zone"}])
    y_axis = alt.Axis(values=[0, 20, 40, 50, 60, 80, 100], labelOverlap=False, labelFontWeight=alt.ExprRef(
        "datum.value == 50 ? 'bold' : 'normal'"), labelExpr=f"datum.value == {DECISION:g} ? "
        f"'{DECISION:g} ▸ threshold' : datum.label")
    layers = [alt.Chart(zones).mark_rect(opacity=0.10).encode(
                  y=alt.Y("y0:Q", scale=ysc, title="synthetic score (%)", axis=y_axis), y2="y1:Q",
                  color=alt.Color("c:N", scale=None, legend=None)),
              alt.Chart(zones).mark_text(align="left", baseline="top", dx=6, dy=4, fontSize=11, fontWeight="bold",
                                         opacity=0.75).encode(
                  x=alt.value(0), y=alt.Y("y1:Q", scale=ysc, axis=y_axis, title="synthetic score (%)"), text="z:N",
                  color=alt.Color("c:N", scale=None, legend=None))]
    seg = pd.DataFrame([{"a": s["start_s"], "b": s["end_s"], "name": f"S{i}", "sev": s["severity"],
                         "start": fmt_ts(s["start_s"]), "end": fmt_ts(s["end_s"]), "p": round(100 * s["p_fake"], 1)}
                        for i, s in enumerate(res.get("segments", []), 1)])
    if len(seg):
        tip = [alt.Tooltip("name:N", title="segment"), alt.Tooltip("start:N", title="begins"),
               alt.Tooltip("end:N", title="ends"), alt.Tooltip("p:Q", title="mean synthetic %"),
               alt.Tooltip("sev:N", title="severity")]
        layers.append(alt.Chart(seg).mark_rect(color=RED, opacity=0.07).encode(x="a:Q", x2="b:Q", tooltip=tip))
        edges = pd.concat([seg.assign(e=seg["a"]), seg.assign(e=seg["b"])])
        layers.append(alt.Chart(edges).mark_rule(color=RED, strokeWidth=1.2, strokeDash=[2, 2]).encode(
            x="e:Q", tooltip=tip))
        layers.append(alt.Chart(seg).mark_text(color=RED, fontWeight="bold", fontSize=11, align="left", dx=3,
                                               baseline="top", dy=2).encode(x="a:Q", y=alt.value(0), text="name:N"))
    thr = pd.DataFrame({"y": [DECISION], "t": [max(dur, 1e-3)], "txt": [f"decision threshold {DECISION:.0f}%"]})
    layers.append(alt.Chart(thr).mark_rule(strokeDash=[6, 4], strokeWidth=1.5, color="#94a3b8").encode(
        y=alt.Y("y:Q", scale=ysc, axis=y_axis, title="synthetic score (%)")))
    if show_members:
        names = [m["label"] for m in res["members"]]
        long = df.melt(id_vars=["t", "run"], value_vars=names, var_name="model", value_name="pct")
        layers.append(alt.Chart(long).mark_line(strokeWidth=1.4, opacity=0.9).encode(
            x=xs, y=alt.Y("pct:Q", scale=ysc, axis=y_axis, title="synthetic score (%)"), detail="run:N",
            color=alt.Color("model:N", scale=alt.Scale(domain=names, range=["#e2e8f0", "#fbbf24", "#94a3b8"]),
                            legend=alt.Legend(title="each model on its own", orient="bottom")),
            strokeDash=alt.StrokeDash("model:N", scale=alt.Scale(domain=names, range=[[6, 3], [2, 2], [8, 3, 2, 3]]),
                                      legend=None)))
    merged = df.dropna(subset=["merged"])
    layers.append(alt.Chart(merged).mark_line(strokeWidth=2.6, color=BLUE, interpolate="monotone").encode(
        x=xs, y=alt.Y("merged:Q", scale=ysc, axis=y_axis, title="synthetic score (%)"), detail="run:N"))
    layers.append(alt.Chart(merged).mark_point(filled=True, size=36, opacity=1).encode(
        x=xs, y=alt.Y("merged:Q", scale=ysc, axis=y_axis, title="synthetic score (%)"),
        color=alt.Color("here:N", scale=alt.Scale(domain=["Real", "Fake"], range=[GREEN, RED]), legend=None)))
    hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    tips = [alt.Tooltip("time:N", title="time"), alt.Tooltip("here:N", title="local result"),
            alt.Tooltip("merged:Q", title="ensemble synthetic %", format=".1f")]
    tips += [alt.Tooltip(f"{m['label']}:Q", title=f"{m['label']} %", format=".0f") for m in res["members"]]
    layers.append(alt.Chart(df).mark_rule(strokeWidth=14, color="#000", opacity=0).encode(
        x=xs, tooltip=tips).add_params(hover))
    layers.append(alt.Chart(df).mark_rule(color="#94a3b8", strokeWidth=1).encode(x=xs).transform_filter(hover))
    return alt.layer(*layers).properties(height=300)


# ---------------------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------------------
def segment_rows(res: dict) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(res.get("segments", []), 1):
        ev = s.get("evidence") or {}
        key = next((a["statement"] for a in ev.get("acoustic", []) if a["kind"] != "reference"), None) or \
            next((a["statement"] for a in ev.get("acoustic", [])), None) or \
            "Model features only (no measured acoustic difference)"
        rows.append({"segment": f"S{i}", "start": fmt_ts(s["start_s"]), "end": fmt_ts(s["end_s"]),
                     "duration (s)": round(s["duration_s"], 1), "fake probability (%)": round(100 * s["p_fake"], 1),
                     "peak (%)": round(100 * s["p_fake_peak"], 1), "severity": s["severity"],
                     "main evidence": key.split(". Possible reading")[0]})
    return pd.DataFrame(rows)


def segments_panel(x: np.ndarray, res: dict) -> None:
    segs = res.get("segments", [])
    section("Flagged segments", "Where the local analysis crosses 50%, with the evidence for each region.", "search",
            f"{len(segs)} region(s)")
    if not segs:
        msg = "No part of the recording reaches 50% in the local analysis."
        if res["verdict"] == "Fake":
            msg += (" The SYNTHETIC verdict comes from evidence spread over the whole file (each model's full-length "
                    "windows), not from one region.")
        note(esc(msg), "check" if res["verdict"] == "Real" else "info")
        return
    if res["verdict"] == "Real":
        note("The recording as a whole scores REAL, but the regions below reach 50% locally. Listen to them; they are "
             "model-flagged suspicions, not findings.", "search")
        st.markdown("")
    st.dataframe(segment_rows(res), hide_index=True, width="stretch", column_config={
        "fake probability (%)": st.column_config.ProgressColumn("synthetic score (%)", min_value=0, max_value=100,
                                                                format="%.1f%%"),
        "main evidence": st.column_config.TextColumn(width="large")})
    st.caption(f"Severity by the segment's mean ensemble synthetic score ({SEVERITY_TEXT}). Timestamps come from "
               f"{res['local_window_s']:.0f}-s analysis windows every {res['step_s']:.1f} s, so boundaries are "
               f"accurate to about ±{res['local_window_s'] / 2:.0f} s.")
    for i, s in enumerate(segs, 1):
        ev = s.get("evidence") or {}
        with st.expander(f"S{i} · {fmt_ts(s['start_s'])}–{fmt_ts(s['end_s'])} · {s['duration_s']:.1f} s · "
                         f"{100 * s['p_fake']:.0f}% synthetic · {s['severity']} severity", expanded=(i == 1)):
            a, b = int(s["start_s"] * SR), int(s["end_s"] * SR)
            st.audio(to_wav_bytes(x[a:b]), format="audio/wav")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Model evidence** (what flagged this segment)")
                st.markdown("\n".join(f"- {html.escape(t)}" for t in ev.get("model", [])) or "- n/a")
            with c2:
                st.markdown("**Acoustic measurements** (model-free; possible readings, not proof)")
                ac = ev.get("acoustic", [])
                st.markdown("\n".join(f"- {html.escape(a_['statement'])}" for a_ in ac) or
                            "- No measured property stands out for this segment.")
                for n_ in ev.get("notes", []):
                    st.caption(n_)


# ---------------------------------------------------------------------------------------
# full result views
# ---------------------------------------------------------------------------------------
MEAS_NAMES = {
    "hf_energy_pct": ("Energy above 4 kHz", "%"), "bandwidth_khz": ("Effective bandwidth", "kHz"),
    "spectral_flux": ("Frame-to-frame spectral change", ""), "hf_flatness": ("Noisiness of 4-8 kHz band", ""),
    "pitch_variation_st": ("Pitch variation", "semitones"), "pitch_jitter_st": ("Pitch jitter", "semitones"),
    "pause_floor_db": ("Pause noise floor vs speech", "dB"),
}


def _reference() -> dict | None:
    from adfd import forensics

    return forensics.load_reference()


def percentile_of(k: str, v, ref: dict | None) -> float | None:
    """Percentile of value v among the real reference recordings (same lookup as adfd.forensics.observations)."""
    if ref is None or v is None or not np.isfinite(v) or k not in ref.get("percentiles", {}):
        return None
    return float(np.interp(v, np.array(ref["percentiles"][k]), np.array(ref["grid"])))


def measurement_table(res: dict) -> pd.DataFrame:
    ref = _reference()
    rows = []
    for k, (name, unit) in MEAS_NAMES.items():
        v = res["measurements"].get(k)
        ok = v is not None and np.isfinite(v)
        pct = percentile_of(k, v, ref) if ok else None
        flag = "n/a" if pct is None else "unusual (low)" if pct <= 5 else "unusual (high)" if pct >= 95 else "normal"
        rows.append({"measurement": name, "value": round(float(v), 3) if ok else None, "unit": unit,
                     "percentile among real speech": None if pct is None else round(pct), "reading": flag})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# explainability center
# ---------------------------------------------------------------------------------------
def _severity(pct: float | None) -> tuple[str, str]:
    if pct is None:
        return "No reference", GREY
    if pct <= 1 or pct >= 99:
        return "Rare", RED
    if pct <= 5 or pct >= 95:
        return "Unusual", AMBER
    return "Typical", GREEN


def _cue_text(k: str, v, pct: float | None) -> str:
    from adfd import forensics

    if v is None or not np.isfinite(v):
        return "Not measurable for this clip (too little speech or no pauses)."
    if pct is None:
        return "No real-speech reference table on this machine, so the value cannot be placed."
    cue = forensics.CUES.get(k)
    if pct <= 5 or pct >= 95:
        side = "lower" if pct <= 5 else "higher"
        rare = pct if pct <= 5 else 100 - pct
        txt = (cue[2] if pct <= 5 else cue[3]) if cue else None
        share = "over 99" if rare < 1 else f"{100 - rare:.0f}"
        base = f"{side.capitalize()} than {share}% of real reference recordings."
        return base + (f" Possible reading: {txt}." if txt and k not in forensics.TABLE_ONLY else "")
    return f"At the {pct:.0f}th percentile, inside the central 90% of real reference recordings."


def _gauge(pct: float | None, color: str) -> str:
    if pct is None:
        return ""
    return (f"<div class='vg-pct' title='Percentile among real reference speech: {pct:.0f}'><div class='band'></div>"
            f"<div class='dot' style='--p:{max(0.0, min(pct, 100.0)):.1f}%;--c:{color}'></div></div>"
            "<div class='vg-pct-l'><span>0</span><span>percentile among real speech (green: 5th–95th)</span>"
            "<span>100</span></div>")


def _mini(vals: list, max_bars: int = 48) -> str:
    """Tiny bar strip of 0-1 scores, coloured by side of the 50% line (None = no speech)."""
    if not vals:
        return ""
    n = len(vals)
    step = max(1, int(np.ceil(n / max_bars)))
    out = []
    for i in range(0, n, step):
        grp = [v for v in vals[i:i + step] if v is not None]
        if not grp:
            out.append(f"<i style='height:8%;--c:{GREY}'></i>")
            continue
        p = float(np.mean(grp))
        out.append(f"<i style='height:{max(6.0, 100 * p):.0f}%;--c:{RED if p >= .5 else GREEN}' "
                   f"title='{100 * p:.0f}% synthetic'></i>")
    return "<div class='vg-mini' aria-hidden='true'>" + "".join(out) + "</div>"


def _xcard(title: str, ico: str, sev: tuple[str, str], value: str, text: str, extra: str = "", details: str = "",
           d: int = 0) -> str:
    label, col = sev
    det = f"<details><summary>Details</summary><div>{details}</div></details>" if details else ""
    return (f"<div class='vg-x' style='--c:{col};--d:{.06 * d:.2f}s'><div class='top'><div class='tt'>"
            f"<span class='ib'>{icon(ico, 16)}</span>{esc(title)}</div><span class='vg-sev'>"
            f"<span class='vg-dot'></span>{esc(label)}</span></div>{value}{extra}"
            f"<div class='txt'>{esc(text)}</div>{det}</div>")


def _val(v, unit: str, fmt: str = "{:.2f}") -> str:
    if v is None or not np.isfinite(v):
        return "<div class='val na'>Not available</div>"
    return f"<div class='val'>{fmt.format(v)}<small>{esc(unit)}</small></div>"


def _secondary(meas: dict, P: dict, k: str, nm: str, unit: str) -> str:
    v = meas.get(k)
    if v is None or not np.isfinite(v):
        return f"{nm}: not available"
    return f"{nm}: <b>{v:.3f} {unit}</b>" + (f" (percentile {P[k]:.0f})" if P.get(k) is not None else "")


def feature_cards(res: dict) -> list[str]:
    meas = res.get("measurements") or {}
    ref = _reference()
    P = {k: percentile_of(k, meas.get(k), ref) for k in MEAS_NAMES}
    cards = []
    # 1 spectral irregularities
    k = "spectral_flux"
    s = _severity(P[k])
    cards.append(_xcard("Spectral irregularities", "activity", s, _val(meas.get(k), "flux"),
                        _cue_text(k, meas.get(k), P[k]), _gauge(P[k], s[1]),
                        "How much the spectrum changes from one 32-ms frame to the next during speech. Over-smoothed "
                        "spectra are typical of text-to-speech acoustic models; abrupt jumps can come from splicing.<br>"
                        + _secondary(meas, P, "hf_flatness", "Noisiness of the 4–8 kHz band", "") + "<br>"
                        + esc(_cue_text("hf_flatness", meas.get("hf_flatness"), P["hf_flatness"])), 0))
    # 2 frequency characteristics
    k = "bandwidth_khz"
    s = _severity(P[k])
    bands = res.get("bands") or []
    top = max(bands, key=lambda b: b["share_pct"] / max(b["bandwidth_share_pct"], 1e-6)) if bands else None
    btxt = (f"The mel-spectrogram model relied most on the {esc(top['band'])} band ({top['share_pct']:.0f}% of the "
            f"evidence on {top['bandwidth_share_pct']:.0f}% of the mel bands) in the decisive window."
            if top else "No band analysis for this result.")
    cards.append(_xcard("Frequency characteristics", "waves", s, _val(meas.get(k), "kHz bandwidth", "{:.1f}"),
                        _cue_text(k, meas.get(k), P[k]), _gauge(P[k], s[1]),
                        "Effective bandwidth: the highest frequency whose long-term level is within 50 dB of the "
                        "peak. Resampling, telephone channels and low-rate vocoders cut it early.<br>" + btxt, 1))
    # 3 temporal patterns
    k = "pitch_variation_st"
    s = _severity(P[k])
    steps = [w["p_fake"] for w in res.get("per_second", [])]
    cards.append(_xcard("Temporal patterns", "clock", s, _val(meas.get(k), "semitones pitch spread"),
                        _cue_text(k, meas.get(k), P[k]), _gauge(P[k], s[1]) + _mini(steps),
                        "Pitch variation is the spread of the voice's pitch over voiced frames; flat intonation is "
                        "common in synthetic or converted speech.<br>"
                        + _secondary(meas, P, "pitch_jitter_st", "Frame-to-frame pitch jitter", "semitones") + "<br>"
                        "The bar strip shows the ensemble's local synthetic score across the recording, left to "
                        "right (red at or above 50%).", 2))
    # 4 energy distribution
    k = "hf_energy_pct"
    s = _severity(P[k])
    cards.append(_xcard("Energy distribution", "chart", s, _val(meas.get(k), "% of energy above 4 kHz", "{:.1f}"),
                        _cue_text(k, meas.get(k), P[k]), _gauge(P[k], s[1]),
                        "Share of speech energy above 4 kHz. Band-limited vocoders and heavy compression remove it; "
                        "some neural vocoders add high-band hiss.<br>"
                        + _secondary(meas, P, "pause_floor_db", "Noise floor in pauses vs speech", "dB")
                        + "<br>The pause floor is shown for reference only: MP3 encoders, noise suppression and "
                          "editing also silence pauses, so it never feeds an explanation.", 3))
    # 5 audio consistency (from the local timeline)
    scored = [p for p in steps if p is not None]
    if not res.get("time_resolved", False) or len(scored) < 2:
        sev5 = ("Single window", GREY)
        v5 = "<div class='val na'>Too short for time-local analysis</div>"
        t5 = "The clip fits in one analysis window, so there is no timeline to compare across."
        extra5 = ""
    else:
        share = 100 * float(np.mean(np.array(scored) >= .5))
        spread = 100 * (max(scored) - min(scored))
        sev5 = ("Consistent", GREEN) if share == 0 else ("Mostly flagged", RED) if share >= 50 else ("Mixed", AMBER)
        v5 = f"<div class='val'>{share:.0f}%<small>of scored time at ≥ 50%</small></div>"
        t5 = (f"Local synthetic scores range over {spread:.0f} points across {len(scored)} scored steps "
              f"({min(scored) * 100:.0f}%–{max(scored) * 100:.0f}%).")
        hist, _ = np.histogram(scored, bins=10, range=(0, 1))
        extra5 = "<div class='vg-mini' aria-hidden='true'>" + "".join(
            f"<i style='height:{max(4.0, 100 * h / max(hist.max(), 1)):.0f}%;--c:{RED if i >= 5 else GREEN}' "
            f"title='{10 * i}–{10 * i + 10}%: {h} step(s)'></i>" for i, h in enumerate(hist)) + "</div>"
    if res.get("out_of_distribution"):
        sev5 = ("Speech check warning", AMBER)
        t5 += " The speech checks flagged the input: " + "; ".join(res["ood_reasons"]) + "."
    cards.append(_xcard("Audio consistency", "layers", sev5, v5, t5, extra5,
                        "Share of the recording whose local ensemble score crosses 50%, and how much the local score "
                        "moves. The histogram counts scored steps by synthetic score, 0% on the left to 100% on the "
                        "right. A recording that is real in one part and synthetic in another shows up as mixed.", 4))
    # 6 model agreement
    ms = res.get("members") or []
    agree, n, status = consensus(res)
    ps = [m["p_fake"] for m in ms]
    spread = 100 * (max(ps) - min(ps)) if ps else 0.0
    sev6 = ("Unanimous", GREEN) if status == "Unanimous" else ("Majority", AMBER) if status == "Majority" else (
        "Split", RED)
    bars6 = "<div class='vg-mini' aria-hidden='true' style='height:34px;gap:6px'>" + "".join(
        f"<i style='height:{max(6.0, 100 * p):.0f}%;--c:{VERDICT_COLOR[m['vote']]}' "
        f"title='{esc(m['label'])}: {100 * p:.0f}% synthetic'></i>" for m, p in zip(ms, ps)) + "</div>"
    cards.append(_xcard("Model agreement", "scale", sev6,
                        f"<div class='val'>{agree}/{n}<small>models share the verdict</small></div>",
                        f"The three own scores lie {spread:.0f} points apart "
                        f"({', '.join(f'{100 * p:.0f}%' for p in ps)} synthetic).", bars6,
                        "<ul>" + "".join(f"<li>{esc(m['label'])}: {100 * m['p_fake']:.1f}% synthetic, calls it "
                                         f"{VERDICT_WORD[m['vote']]} (weight {100 * m['weight']:.0f}%)</li>"
                                         for m in ms)
                        + "</ul>On held-out data, unanimous verdicts were right more often than split ones.", 5))
    return cards


def ltas_chart(x: np.ndarray):
    import altair as alt

    d = signal.ltas(x, SR)
    df = pd.DataFrame({"kHz": np.array(d["freq_hz"]) / 1000, "dB": np.maximum(np.array(d["db"]), -90)})
    area = alt.Chart(df).mark_area(
        line={"color": CYAN, "strokeWidth": 1.6}, interpolate="monotone",
        color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                           stops=[alt.GradientStop(color="rgba(34,211,238,0.0)", offset=0),
                                  alt.GradientStop(color="rgba(34,211,238,0.35)", offset=1)])).encode(
        x=alt.X("kHz:Q", title="frequency (kHz)", scale=alt.Scale(domain=[0, SR / 2000], nice=False)),
        y=alt.Y("dB:Q", title="average level (dB re max)", scale=alt.Scale(domain=[-90, 0])),
        tooltip=[alt.Tooltip("kHz:Q", format=".2f", title="frequency kHz"), alt.Tooltip("dB:Q", format=".1f")])
    mark = pd.DataFrame({"kHz": [4.0], "t": ["high band ▸"]})
    rule = alt.Chart(mark).mark_rule(color=AMBER, strokeDash=[4, 4]).encode(x="kHz:Q")
    txt = alt.Chart(mark).mark_text(color=AMBER, align="left", dx=5, baseline="top", fontSize=11).encode(
        x="kHz:Q", y=alt.value(4), text="t:N")
    return alt.layer(area, rule, txt).properties(height=230)


def bands_chart(res: dict):
    import altair as alt

    rows = []
    for b in res["bands"]:
        rows.append({"band": b["band"], "what": "share of model evidence", "pct": b["share_pct"]})
        rows.append({"band": b["band"], "what": "share of mel bands", "pct": b["bandwidth_share_pct"]})
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("band:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)), x=alt.X("pct:Q", title="%"),
        yOffset=alt.YOffset("what:N", sort=["share of model evidence", "share of mel bands"]),
        color=alt.Color("what:N", scale=alt.Scale(domain=["share of model evidence", "share of mel bands"],
                                                   range=[CYAN, "#475569"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["band", "what", alt.Tooltip("pct:Q", format=".1f")]).properties(height=230)


def summary_html(res: dict) -> str:
    agree, n, _ = consensus(res)
    n_syn = sum(m["vote"] == "Fake" for m in res.get("members", []))
    segs = res.get("segments", [])
    flagged = sum(s["end_s"] - s["start_s"] for s in segs)
    obs = res.get("observations", [])
    bands = res.get("bands") or []
    top = max(bands, key=lambda b: b["share_pct"] / max(b["bandwidth_share_pct"], 1e-6))["band"] if bands else "–"
    rows = [("Final verdict", verdict_chip(res["verdict"])),
            ("Ensemble synthetic score", f"{100 * res['p_fake']:.1f}%"),
            ("Models voting SYNTHETIC", f"{n_syn} of {n}"),
            ("Flagged time", f"{flagged:.1f} s of {res['duration_s']:.1f} s" if segs else "none"),
            ("Most-relied band", esc(top)),
            ("Unusual measurements", f"{len(obs)} of {len(MEAS_NAMES) - 1}"),
            ("Speech check", chip("warning", "amber", "alert") if res["out_of_distribution"]
             else chip("passed", "real", "check"))]
    body = "".join(f"<div style='display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px "
                   f"solid #1f2a3c;font-size:.86rem'><span style='color:#8b98ad'>{k}</span><span style='color:#e6ebf2;"
                   f"font-weight:600;text-align:right' class='vg-num'>{v}</span></div>" for k, v in rows)
    lines = "".join(f"<li>{esc(ln[2:] if ln.startswith('• ') else ln)}</li>" for ln in res["explanation"]["lines"])
    return (f"<div class='vg-panel' style='height:100%'><div style='display:flex;align-items:center;gap:10px;"
            f"font-weight:600;margin-bottom:6px'><span style='color:{CYAN}'>{icon('brain', 18)}</span>Model prediction "
            f"summary</div>{body}<details style='margin-top:10px'><summary style='cursor:pointer;color:{CYAN};"
            f"font-size:.82rem'>Read the full reasoning</summary><ul style='font-size:.82rem;color:#8b98ad;"
            f"line-height:1.55;padding-left:1.1rem;margin-top:8px'>{lines}</ul></details></div>")


def explain_center(x: np.ndarray, res: dict) -> None:
    flagged = res["verdict"] == "Fake" or bool(res.get("segments"))
    sub = ("The evidence behind the SYNTHETIC verdict: where the models reacted and how the audio measures against "
           "real speech." if res["verdict"] == "Fake" else
           "This recording was called REAL. Below: what the models and the measurements saw"
           + (", including the regions that crossed 50% locally." if flagged else "."))
    section("Why was this audio flagged?", sub, "search", "model evidence + model-free measurements", big=True)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.altair_chart(waveform_chart(x, res), width="stretch")
        st.markdown("<div style='display:flex;justify-content:space-between;font-size:.75rem;color:#8b98ad;"
                    "margin:10px 2px 4px'><span>Spectrogram · 0–8 kHz, low at the bottom</span><span>whole recording, "
                    "left to right</span></div>", unsafe_allow_html=True)
        st.image(signal.spectrogram_png(x, SR), width="stretch")
    with right:
        st.markdown(summary_html(res), unsafe_allow_html=True)
    if res.get("bands"):
        c1, c2 = st.columns(2, gap="medium")
    else:
        c1, c2 = st.container(), None
    with c1:
        st.markdown("<div style='font-weight:600;font-size:.92rem;margin:18px 0 8px'>Frequency analysis · average "
                    "spectrum of the whole recording</div>", unsafe_allow_html=True)
        st.altair_chart(ltas_chart(x), width="stretch")
    if c2 is not None:
        with c2:
            st.markdown("<div style='font-weight:600;font-size:.92rem;margin:18px 0 8px'>Where the spectrogram model "
                        "looked · evidence by frequency band</div>", unsafe_allow_html=True)
            st.altair_chart(bands_chart(res), width="stretch")
    st.markdown("<div style='font-weight:600;font-size:.92rem;margin:22px 0 10px'>Feature analysis</div>",
                unsafe_allow_html=True)
    cards = feature_cards(res)
    for i in range(0, len(cards), 3):
        for c, h in zip(st.columns(3, gap="medium"), cards[i:i + 3]):
            c.markdown(h, unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.caption("Measurements are model-free and compared with 1,700 real reference recordings; they describe the audio "
               "and never change the verdict. Model evidence shows where a model reacted, not proof of manipulation.")


def ensemble_view(name: str, x: np.ndarray, res: dict, elapsed: float | None = None) -> bytes:
    t = trust_of(res)
    st.markdown(verdict_html(res, f"ensemble of {len(res['members'])} models", elapsed, t), unsafe_allow_html=True)
    if res["out_of_distribution"]:
        st.markdown(warn_html("This doesn't look like normal speech",
                              "So the verdict is much less reliable: " + esc("; ".join(res["ood_reasons"])) + "."),
                    unsafe_allow_html=True)
    models_panel(res)
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        section("Model comparison", "REAL and SYNTHETIC score of each model and of the ensemble.", "chart")
        st.markdown(f"<div class='vg-panel'>{comparison_html(res)}</div>", unsafe_allow_html=True)
        st.caption("The ensemble clips each model's log-odds (so no over-confident model dominates) and weighs them; "
                   "two models that agree always outweigh the third. Hover a bar for its exact value.")
    with c2:
        section("Reliability", "Held-out accuracy of verdicts like this one.", "gauge")
        st.markdown(trust_panel_html(t), unsafe_allow_html=True)

    section("Spectrogram lab", "Play, zoom, select and inspect. Top strip: the ensemble's local score; dashed red: "
            "flagged segments.", "audio-waveform", f"{res['n_local_windows_scored']} local windows scored")
    lab.render([lab.lane(x, SR, name, "main", res)], key=f"lab_{name}")

    explain_center(x, res)

    section("Synthetic score over time", "Ensemble score of the local windows, on REAL and SYNTHETIC zones.",
            "activity")
    show = st.toggle("Show each model's own line", value=False, key=f"members_{name}")
    st.altair_chart(timeline_chart(res, show), width="stretch")
    if not res["time_resolved"]:
        note("This clip is too short for time-local analysis (one analysis window), so the timeline has a single "
             "value. Upload a longer recording to see changes over time.")
    else:
        st.caption(f"Each point is the ensemble synthetic score of the {res['local_window_s']:.0f}-s windows covering "
                   f"that moment, scored by all three models (one point every {res['step_s']:.1f} s; window-level "
                   "calibration). Red points are at or above the 50% line, green below. Hover for each model's value. "
                   "The file verdict uses each model's full-length windows, so a file can be SYNTHETIC without any "
                   "single region crossing 50%, or REAL with a few flagged regions.")
    segments_panel(x, res)

    section("Advanced forensics", "Full reasoning, saliency, measurements and model internals.", "cpu")
    with st.expander("Why: full explanation", expanded=False):
        body = []
        for ln in res["explanation"]["lines"]:
            body.append("    - " + ln[2:] if ln.startswith("• ") else "- " + ln)
        st.markdown("\n".join(body))
    with st.expander("Forensic view: spectrogram and saliency"):
        with st.spinner("Drawing the forensic view…"):
            fpng = fig_png(plot_forensic(x, res))
        st.image(fpng, width="stretch")
        st.caption("Top: waveform coloured by the local analysis. Middle: the 128-band log-mel spectrogram the "
                   "VoiceGuard models read; the bar on top repeats the regions, and the cyan outline marks the "
                   "time-frequency areas that moved the score most in the decisive window (gradient × input). Bottom: "
                   "the merged local score and each model's own full-window scores. Saliency shows where a model "
                   "looked, not proof of manipulation.")
    with st.expander("Whole-file acoustic measurements compared with real speech"):
        st.dataframe(measurement_table(res), hide_index=True, width="stretch")
        st.caption("Percentile among 1,700 real reference recordings (ASVspoof train/dev bona fide, clean and with "
                   "codec, noise and telephone degradations). Values below the 5th or above the 95th percentile are "
                   "reported as observations; they never change the verdict.")
    if res.get("bands"):
        with st.expander("Which frequency bands the mel-spectrogram model relied on"):
            st.dataframe(pd.DataFrame([{"band": b["band"], "share of evidence %": round(b["share_pct"], 1),
                                        "share of mel bands %": round(b["bandwidth_share_pct"], 1)}
                                       for b in res["bands"]]), hide_index=True, width="stretch")
    with st.expander("Inside the models: Grad-CAM, attention over time, WavLM layers"):
        internals_panel(name, x, res)
    return fpng


def _decisive(tl: list[dict], fake: bool) -> dict:
    return max(tl, key=lambda w: w["p_fake"]) if fake else min(tl, key=lambda w: w["p_fake"])


def internals_panel(name: str, x: np.ndarray, res: dict) -> None:
    """Grad-CAM for each spectrogram member, attention and layer weights for the WavLM member (P4.5)."""
    from adfd import interpret

    if not st.toggle("Compute (a few seconds per model)", key=f"internals_{name}_{res['p_fake']:.6f}"):
        st.caption("Grad-CAM shows which parts of the input the last convolution layer used to push the score towards "
                   "FAKE; the attention curve shows which moments the WavLM model weighted most when summarising "
                   "the window.")
        return
    ens = get_analyzer(registry.ENSEMBLE)
    fake = res["verdict"] == "Fake"
    for m in res["members"]:
        an = ens.analyzers[m["run"]] if hasattr(ens, "analyzers") else None
        if an is None:
            continue
        w = _decisive(m["timeline"], fake)
        a, b = int(w["start_s"] * SR), int(w["end_s"] * SR)
        st.markdown(f"**{m['label']}** · decisive window {fmt_ts(w['start_s'])}–{fmt_ts(w['end_s'])} "
                    f"({100 * w['p_fake']:.0f}% fake)")
        try:
            with st.spinner(f"Grad-CAM for {m['label']}…"):
                gc = interpret.grad_cam(an, x[a:b])
            if gc is not None:
                spectral = not m["run"].startswith("e7")
                st.image(fig_png(interpret.plot_grad_cam(gc, f"Grad-CAM, {m['label']}", spectral)), width="stretch")
            att = interpret.attention(an, x[a:b])
            if att is not None:
                st.image(fig_png(interpret.plot_attention(att, f"Attention over time, {m['label']}")), width="stretch")
            lw = interpret.layer_weights(an)
            if lw is not None:
                st.image(fig_png(interpret.plot_layer_weights(lw)), width="stretch")
        except Exception as e:  # noqa: BLE001
            log.warning("internals for %s failed: %r", m["run"], e)
            st.info(f"Could not compute the internal views for this model ({e.__class__.__name__}).")
    st.caption("Bright Grad-CAM areas pushed the score towards FAKE; for the WavLM model the vertical axis is the "
               "128 learned features, not frequency. The dashed lines mark a uniform weight. These show where a model "
               "looked, not proof of manipulation.")


def ensemble_report(name: str, res: dict, fpng: bytes) -> str:
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    line_png = fig_png(plot_score_line(res, show_members=True))
    fake = res["verdict"] == "Fake"
    color = RED if fake else GREEN
    lines = "".join(f"<li>{html.escape(ln.lstrip('• '))}</li>" for ln in res["explanation"]["lines"])
    members = "".join(f"<tr><td>{html.escape(m['label'])}</td><td>{100 * m['weight']:.0f}%</td>"
                      f"<td>{100 * m['p_fake']:.1f}%</td><td>{m['vote'].upper()}</td></tr>" for m in res["members"])
    segs = ""
    for i, s in enumerate(res.get("segments", []), 1):
        ev = s.get("evidence") or {}
        items = "".join(f"<li>{html.escape(t)}</li>" for t in ev.get("model", []))
        items += "".join(f"<li>{html.escape(a['statement'])}</li>" for a in ev.get("acoustic", []))
        segs += (f"<tr><td>S{i}</td><td>{fmt_ts(s['start_s'])}</td><td>{fmt_ts(s['end_s'])}</td>"
                 f"<td>{s['duration_s']:.1f}</td><td>{100 * s['p_fake']:.1f}%</td><td>{s['severity']}</td>"
                 f"<td><ul>{items}</ul></td></tr>")
    segs = (f"<table><tr><th>#</th><th>start</th><th>end</th><th>duration (s)</th><th>fake prob.</th>"
            f"<th>severity</th><th>evidence</th></tr>{segs}</table>" if segs
            else "<p>No part of the recording reaches 50% in the local analysis.</p>")
    real_pct, fake_pct = 100 * (1 - res["p_fake"]), 100 * res["p_fake"]
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Deepfake report: {html.escape(name)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1f1f1e}}
img{{max-width:100%}} table{{border-collapse:collapse;margin:.5rem 0}} td,th{{border:1px solid #ccc;padding:4px 8px;
vertical-align:top}} .v{{border:2px solid {color};border-radius:12px;padding:.8rem 1rem}}
.m{{display:flex;justify-content:space-between;height:20px;border-radius:10px;overflow:hidden;background:#eee}}
ul{{margin:0;padding-left:1.1rem}} .adfd-trust{{border:1px solid #ccc;border-radius:12px;padding:.8rem 1rem}}
.adfd-trust .lvl{{font-size:1.2rem;font-weight:800}} .adfd-bar{{height:8px;border-radius:4px;background:#eee;
overflow:hidden;margin:3px 0 8px}} .adfd-bar>div{{height:100%;background:var(--c)}}</style></head><body>
<h1>Audio deepfake report</h1>
<p>File: <b>{html.escape(name)}</b> ({res['duration_s']:.1f} s) · {datetime.now():%Y-%m-%d %H:%M} ·
model: {registry.ENSEMBLE_LABEL}</p>
<div class='v'><div style='font-size:2rem;font-weight:800;color:{color}'>{'FAKE' if fake else 'REAL'}</div>
Fake probability <b>{fake_pct:.1f}%</b> (FAKE at ≥ 50%)
<p style='display:flex;justify-content:space-between;margin:.6rem 0 .3rem'><span style='color:{GREEN}'>Realness
{real_pct:.1f}%</span><span style='color:{RED}'>Fakeness {fake_pct:.1f}%</span></p>
<div class='m'><div style='width:{real_pct:.1f}%;background:{GREEN}'></div><div style='width:{fake_pct:.1f}%;
background:{RED}'></div></div></div>
<h2>How far to trust it</h2>{trust_html(trust_of(res))}
<h2>The three models</h2><table><tr><th>model</th><th>weight</th><th>fake probability</th><th>on its own</th></tr>
{members}</table>
<h2>Fake probability over time</h2><img src='data:image/png;base64,{b64(line_png)}'>
<h2>Flagged segments</h2>{segs}
<h2>Why</h2><ul>{lines}</ul>
<h2>Forensic view</h2><img src='data:image/png;base64,{b64(fpng)}'>
<h2>Whole-file acoustic measurements</h2>{measurement_table(res).to_html(index=False)}
<h2>Limitations</h2><p>{LIMITS}</p></body></html>"""


@st.cache_data(show_spinner=False, max_entries=8)
def decode(data: bytes) -> np.ndarray:
    return load_audio(data)


def prepare(name: str, data: bytes) -> np.ndarray | None:
    """Decode and check one input; show a clear message and return None when it cannot be analysed."""
    try:
        x = decode(data)
    except Exception as e:  # noqa: BLE001
        log.warning("decode failed for %s: %r", name, e)
        st.error(f"Could not read **{name}** as audio ({e.__class__.__name__}). Save it as WAV, FLAC, MP3, OGG or "
                 "MP4 and try again.", icon="🛑")
        return None
    if len(x) < SR // 4:
        st.error("The clip is shorter than 0.25 s; there is nothing to analyse.", icon="🛑")
        return None
    if not np.isfinite(x).all() or float(np.abs(x).max()) < 1e-4:
        st.error("The file decodes to silence, so there is no speech to analyse.", icon="🛑")
        return None
    if len(x) > MAX_SECONDS * SR:
        st.warning(f"The recording is {len(x) / SR / 60:.1f} min long; only its first {MAX_SECONDS // 60} minutes "
                   "are analysed. Cut the part you care about for a full look.", icon="✂️")
        x = x[: MAX_SECONDS * SR]
    elif len(x) < 2 * SR:
        st.warning("Under 2 s of audio: there is little to go on, so the verdict is weaker than usual.", icon="⚠️")
    return x


def remember(name: str, run: str, res: dict, key: str) -> None:
    """Add one analysis to this browser session's history (once per file and model)."""
    hist = st.session_state.setdefault("history", [])
    if any(h["key"] == key for h in hist):
        return
    t = trust_of(res)
    row = {"key": key, "time": datetime.now().strftime("%H:%M:%S"), "file": name,
           "model": registry.available().get(run, run), "verdict": res["verdict"].upper(),
           "fake_pct": round(100 * res["p_fake"], 1), "confidence": t["level"],
           "models_agreeing": f"{t['agree']} of {t['n_members']}" if res.get("members") else "–",
           "duration_s": round(res["duration_s"], 1), "flagged_segments": len(res.get("segments") or
                                                                              res.get("suspicious_segments", [])),
           "speech_check": "warning" if res["out_of_distribution"] else "ok"}
    hist.insert(0, row)


def clear_input() -> None:
    """Forget the current input: the page's uploader, recorder and demo picker take a fresh key."""
    st.session_state["input_n"] = st.session_state.get("input_n", 0) + 1
    st.session_state["vg_toast"] = ("File removed", ":material/delete:")


def _estimate(dur: float) -> tuple[int, int]:
    # measured on the project laptop's CPU: ~0.65 s (cool) to ~3.4 s (hot, throttled) per second of
    # audio up to 1 min; longer files grow more slowly (the local timeline is capped at 120 windows)
    eff = dur if dur <= 60 else 60 + 0.4 * (dur - 60)
    on_gpu = device().startswith("cuda") and gpu_available()
    f_lo, f_hi = (0.08, 0.5) if on_gpu else (0.65, 3.5)
    return max(5, 5 * round(f_lo * eff / 5)), max(15, 5 * round(f_hi * eff / 5))


def analyse_one(name: str, data: bytes, run: str, compare: bool, step_ph=None) -> None:
    """Upload -> preview -> analyze -> results, for one recording."""
    step_ph = step_ph or st.empty()
    x = prepare(name, data)
    if x is None:
        step_ph.markdown(stepper_html(1), unsafe_allow_html=True)
        return
    ens = run == registry.ENSEMBLE
    key = ("analysis", run, hashlib.sha1(data if ens else x.tobytes()).hexdigest())
    done = key in st.session_state
    step_ph.markdown(stepper_html(5 if done else 2), unsafe_allow_html=True)

    ext = Path(name).suffix.lstrip(".").upper() or "AUDIO"
    tag = verdict_chip(st.session_state[key]["verdict"]) if done else chip("Ready to analyze", "cyan", "zap")
    fc, bc = st.columns([5, 1], vertical_alignment="center")
    fc.markdown(file_card(name, [("Size", fmt_size(len(data))), ("Format", ext), ("Duration", fmt_dur(len(x) / SR)),
                                 ("Analysed as", "16 kHz mono")], tag=tag), unsafe_allow_html=True)
    bc.button("Remove", icon=":material/close:", on_click=clear_input, key=f"rm_{key[2]}", width="stretch",
              help="Clear this file and start again")

    if not done:
        section("Preview", "Listen to the recording and inspect it before you analyze it.", "audio-waveform",
                fmt_dur(len(x) / SR))
        lab.render([lab.lane(x, SR, name)], key=f"prev_{key[2]}")
        lo, hi = _estimate(len(x) / SR)
        go = st.button("Analyze recording", type="primary", icon=":material/graphic_eq:", key=f"go_{key[2]}")
        st.caption(f"Runs {'all three models' if ens else 'the selected model'} on the {device_label()}; usually "
                   f"{lo}–{hi} s, plus ~15 s the first time while the models load. Nothing leaves this computer.")
        if not go:
            return
        step_ph.markdown(stepper_html(3), unsafe_allow_html=True)
        box = st.empty()
        with box.container():
            st.markdown(loader_html(f"Analyzing {name}",
                                    f"{len(x) / SR:.0f} s of audio on the {device_label()} · usually {lo}–{hi} s. "
                                    "The bar shows the real progress of the models."), unsafe_allow_html=True)
            pb = st.progress(0.0, text="Starting…")
            st.markdown("<div style='height:8px'></div>" + results_skeleton(), unsafe_allow_html=True)
        t0 = time.perf_counter()
        try:
            an = get_analyzer(run)
            if ens:
                res = run_analysis(run, an, x, progress=lambda f, t: pb.progress(f, text=t))
            else:
                pb.progress(0.3, text="Scoring with the selected model…")
                res = run_analysis(run, an, x)
        except Exception as e:  # noqa: BLE001
            log.exception("analysis failed for %s", name)
            box.empty()
            st.error(f"The analysis stopped with an error ({e.__class__.__name__}: {e}). Try again; if it repeats, "
                     "convert the file to 16 kHz WAV or pick a single model in the sidebar.", icon="🛑")
            return
        st.session_state[key] = res
        st.session_state[("elapsed",) + key] = time.perf_counter() - t0
        st.session_state["vg_toast"] = (f"Analysis complete: {VERDICT_WORD[res['verdict']]}", ":material/task_alt:")
        st.rerun()                     # redraw without the progress widgets

    res = st.session_state[key]
    elapsed = st.session_state.get(("elapsed",) + key)
    remember(name, run, res, key[2])
    if not ens:
        return analyse_single(name, x, run, compare, elapsed)
    fpng = ensemble_view(name, x, res, elapsed)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.download_button("Download report (HTML)", ensemble_report(name, res, fpng), icon=":material/download:",
                       file_name=f"{Path(name).stem}_deepfake_report.html", mime="text/html", on_click="ignore",
                       type="primary")


def report_html(name: str, res: dict, timeline_png: bytes, heatmap_png: bytes | None, others: pd.DataFrame | None,
                model_label: str) -> str:
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    color = RED if res["verdict"] == "Fake" else GREEN
    segs = "".join(f"<li>{fmt_ts(w['start_s'])}–{fmt_ts(w['end_s'])}: fake probability {100 * w['p_fake']:.0f}%</li>"
                   for w in res["suspicious_segments"]) or "<li>no window at or above 50%</li>"
    ood = ("<p style='color:#b35c00'><b>Out-of-distribution warning:</b> " + "; ".join(res["ood_reasons"]) + "</p>"
           if res["out_of_distribution"] else "<p>Input looks like speech (signal and embedding checks passed).</p>")
    table = others.to_html(index=False, float_format=lambda v: f"{v:.3f}") if others is not None else ""
    heat = f"<h2>Spectrogram and saliency</h2><img src='data:image/png;base64,{b64(heatmap_png)}'>" if heatmap_png else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Deepfake report: {html.escape(name)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f1f1e}}
img{{max-width:100%}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.v{{border:2px solid {color};border-radius:12px;padding:.6rem 1rem}}</style></head><body>
<h1>Audio deepfake report</h1>
<p>File: <b>{html.escape(name)}</b> ({res['duration_s']:.1f} s) · {datetime.now():%Y-%m-%d %H:%M} · model: {model_label} (<code>{res['model']}</code>)</p>
<div class='v'><h2 style='margin:0;color:{color}'>{res['verdict'].upper()}</h2>Fake probability: <b>{100 * res['p_fake']:.1f}%</b>,
realness {100 * (1 - res['p_fake']):.1f}% (FAKE at ≥ 50%)</div>
{ood}<h2>Score over time</h2><img src='data:image/png;base64,{b64(timeline_png)}'>
<h3>Most suspicious windows</h3><ul>{segs}</ul>{heat}
{'<h2>All models</h2>' + table if table else ''}
<h2>Limitations</h2><p>{LIMITS}</p></body></html>"""


def analyse_single(name: str, x: np.ndarray, run: str, compare: bool, elapsed: float | None = None) -> None:
    an = get_analyzer(run)
    key = ("analysis", run, hashlib.sha1(x.tobytes()).hexdigest())
    if key not in st.session_state:
        try:
            with st.spinner("Analysing…"):
                st.session_state[key] = run_analysis(run, an, x)
        except Exception as e:  # noqa: BLE001
            log.exception("analysis failed for %s", name)
            st.error(f"The analysis stopped with an error ({e.__class__.__name__}: {e}).", icon="🛑")
            return
    res = st.session_state[key]
    remember(name, run, res, key[2])
    st.markdown(verdict_html(res, registry.available().get(run, "single model"), elapsed), unsafe_allow_html=True)
    if res["out_of_distribution"]:
        st.markdown(warn_html("This doesn't look like normal speech", "So the verdict is much less reliable: "
                              + esc("; ".join(res["ood_reasons"])) + "."), unsafe_allow_html=True)
    section("Spectrogram lab", "Play, zoom, select and inspect the recording.", "audio-waveform")
    lab.render([lab.lane(x, SR, name)], key=f"lab_{name}")
    section("Score over time", "The selected model's windows and the most suspicious regions.", "activity")
    left, right = st.columns([1, 2])
    with left:
        st.caption(f"{res['duration_s']:.1f} s · analysed {res['analysed_region_s'][0]:.1f}–"
                   f"{res['analysed_region_s'][1]:.1f} s · {len(res['timeline'])} window(s)"
                   f"{'' if res['calibrated'] else ' · uncalibrated model'}")
    with right:
        tl_png = fig_png(plot_timeline(res))
        st.image(tl_png, width="stretch")
        if res["suspicious_segments"]:
            st.markdown("**Most suspicious windows**")
            for w in res["suspicious_segments"]:
                a, b = int(w["start_s"] * SR), int(w["end_s"] * SR)
                st.write(f"{fmt_ts(w['start_s'])}–{fmt_ts(w['end_s'])} · fake probability {100 * w['p_fake']:.0f}%")
                st.audio(to_wav_bytes(x[a:b]), format="audio/wav")
    heat_png = None
    with st.expander("Spectrogram and saliency heatmap (most suspicious window)"):
        top = max(res["timeline"], key=lambda w: w["p_fake"])
        a, b = int(top["start_s"] * SR), int(top["end_s"] * SR)
        with st.spinner("Computing saliency…"):
            heat_png = fig_png(plot_heatmap(an._prep(x[a:b]), an.saliency(x[a:b]), (b - a) / SR))
        st.image(heat_png, width="stretch")
        st.caption("Bright areas changed the model's score the most (gradient × input). This shows where the model "
                   "looked, not proof of manipulation.")
    with st.expander("Inside the model: Grad-CAM and attention (most suspicious window)"):
        from adfd import interpret

        try:
            gc = interpret.grad_cam(an, x[a:b])
            if gc is not None:
                st.image(fig_png(interpret.plot_grad_cam(gc, spectral=an.frontend_name != "ssl")), width="stretch")
            att = interpret.attention(an, x[a:b])
            if att is not None:
                st.image(fig_png(interpret.plot_attention(att)), width="stretch")
            lw = interpret.layer_weights(an)
            if lw is not None:
                st.image(fig_png(interpret.plot_layer_weights(lw)), width="stretch")
            if gc is None and att is None and lw is None:
                st.info("This model has no convolution, attention pooling or SSL layer weights to show.")
        except Exception as e:  # noqa: BLE001
            log.warning("internals for %s failed: %r", run, e)
            st.info(f"Could not compute the internal views for this model ({e.__class__.__name__}).")
    others = None
    if compare:
        rows = []
        for r, label in registry.available().items():
            if r == registry.ENSEMBLE:
                continue
            o = run_analysis(r, get_analyzer(r), x)
            rows.append({"model": label, "verdict": o["verdict"].upper(), "fake probability %": 100 * o["p_fake"],
                         "out of distribution": o["out_of_distribution"]})
        others = pd.DataFrame(rows)
        st.markdown("**All models**")
        st.dataframe(others, hide_index=True, width="stretch")
        counts = others["verdict"].value_counts()
        st.caption(f"{counts.iloc[0]} of {len(others)} models say **{counts.index[0]}**.")
    st.download_button("Download report (HTML)", report_html(name, res, tl_png, heat_png, others,
                                                              registry.available().get(run, run)),
                       file_name=f"{Path(name).stem}_deepfake_report.html", mime="text/html", on_click="ignore")


def batch(files: list[tuple[str, bytes]], run: str) -> pd.DataFrame:
    an = get_analyzer(run)
    rows = []
    pb = st.progress(0.0, text="Scoring…")
    for i, (name, data) in enumerate(files, 1):
        pb.progress((i - 1) / len(files), text=f"Scoring {name} ({i} of {len(files)})…")
        try:
            x = load_audio(data)
            if len(x) < SR // 4:
                raise ValueError("shorter than 0.25 s")
            cut = len(x) > MAX_SECONDS * SR
            x = x[: MAX_SECONDS * SR]
            res = run_analysis(run, an, x, explain=False) if run == registry.ENSEMBLE else run_analysis(run, an, x)
            t = trust_of(res)
            row = {"file": name, "verdict": res["verdict"].upper(), "fake_pct": round(100 * res["p_fake"], 2),
                   "real_pct": round(100 * (1 - res["p_fake"]), 2), "confidence": t["level"],
                   "duration_s": round(res["duration_s"], 2), "first_10_min_only": cut,
                   "out_of_distribution": res["out_of_distribution"], "ood_reasons": "; ".join(res["ood_reasons"])}
            if run == registry.ENSEMBLE:
                row["models_agreeing"] = f"{t['agree']} of {t['n_members']}"
                row.update({f"{m['run']}_fake_pct": round(100 * m["p_fake"], 2) for m in res["members"]})
                row["flagged_segments"] = "; ".join(f"{fmt_ts(s['start_s'])}-{fmt_ts(s['end_s'])} "
                                                    f"({100 * s['p_fake']:.0f}%, {s['severity']})"
                                                    for s in res["segments"])
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            log.warning("batch: %s failed: %r", name, e)
            rows.append({"file": name, "verdict": "ERROR", "fake_pct": None, "real_pct": None, "confidence": None,
                         "duration_s": None, "out_of_distribution": None,
                         "ood_reasons": f"could not analyse: {e.__class__.__name__}: {e}"})
    pb.progress(1.0, text=f"Scored {len(files)} of {len(files)}")
    return pd.DataFrame(rows)


def unpack(uploads) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Files and ZIP members as (name, bytes), plus messages about anything skipped."""
    out, skipped = [], []
    for u in uploads:
        data = u.getvalue()
        if u.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        fn = info.filename
                        if info.is_dir() or "__MACOSX" in fn or Path(fn).name.startswith("."):
                            continue
                        if fn.lower().rsplit(".", 1)[-1] not in AUDIO_TYPES:
                            skipped.append(f"{u.name}/{fn}: not an audio file")
                        elif info.file_size > MAX_MB * 2**20:
                            skipped.append(f"{u.name}/{fn}: larger than {MAX_MB} MB")
                        else:
                            out.append((fn, z.read(info)))
            except zipfile.BadZipFile:
                skipped.append(f"{u.name}: not a valid ZIP archive")
        else:
            out.append((u.name, data))
    return out, skipped
