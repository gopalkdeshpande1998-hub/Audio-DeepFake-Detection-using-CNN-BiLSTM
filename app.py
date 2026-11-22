"""Audio deepfake detector: Streamlit app (v2, PyTorch).

    streamlit run app.py

Default model: the merged detector (adfd.ensemble: WavLM E7a + VoiceGuard improved +
VoiceGuard v2). Per file: one final REAL / FAKE verdict from the weighted combination of
all three models (never "Uncertain"), a realness / fakeness meter, each model's own score,
a waveform coloured by time-local analysis (red = suspected fake, green = real), the merged
fake probability over time, and a table of flagged segments with model evidence and
acoustic measurements. Also: a forensic spectrogram view with saliency, acoustic
measurements against real reference speech, an HTML report, and a warning when the input
does not look like speech. Input: upload (audio or video), microphone recording, or a demo
clip. Batch mode scores many files (or a ZIP) into a CSV. Models: see adfd.registry;
inference runs on the CPU.
"""

from __future__ import annotations

import base64
import html
import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from adfd.paths import configure_caches, load_paths

configure_caches()
logging.getLogger("streamlit").setLevel(logging.ERROR)

from adfd import registry  # noqa: E402
from adfd.explain import plot_forensic, plot_heatmap, plot_score_line, plot_timeline  # noqa: E402
from adfd.inference import SR, load_audio  # noqa: E402
from adfd.segments import SEVERITY_TEXT, fmt_ts  # noqa: E402

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a", "aiff", "aif", "webm", "opus"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "avi", "webm"]
MAX_MB = 50
DECISION = 50.0                                    # % : FAKE at or above, REAL below
BORDER = 10.0                                      # amber band +-10 around the line: visual reference only
GREEN, RED, AMBER, GREY, BLUE = "#16a34a", "#dc2626", "#f59e0b", "#a8a29e", "#2563eb"
LABEL_COLOR = {"Real": GREEN, "Fake": RED, "No speech": GREY}
LIMITS = ("Research prototype. The final verdict is always REAL or FAKE: the weighted combination of three models, "
          "decided at 50%. Every verdict can be wrong. On held-out tests about 1 in 9 real studio recordings "
          "(ASVspoof) are called FAKE, 1 in 5 ASVspoof fakes and most voice-conversion fakes (DEEP-VOICE) are "
          "called REAL; see "
          "experiments/ensemble_binary/ensemble.md. Flagged regions are model-flagged suspicions, not proof of "
          "manipulation. Never use one verdict as evidence on its own.")

st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="wide")

CSS = f"""
<style>
.adfd-card{{border:1px solid rgba(127,127,127,.25);border-radius:12px;padding:14px 16px;height:100%}}
.adfd-verdict{{border-radius:14px;padding:16px 20px;border:2px solid var(--c);background:var(--bg)}}
.adfd-verdict .k{{font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;opacity:.75}}
.adfd-verdict .v{{font-size:2.6rem;font-weight:800;line-height:1.1;color:var(--c);margin:2px 0 4px}}
.adfd-verdict .s{{font-size:.95rem}}
.adfd-meter-labels{{display:flex;justify-content:space-between;font-size:.95rem;margin:14px 0 6px}}
.adfd-meter-labels .r{{color:{GREEN}}} .adfd-meter-labels .f{{color:{RED}}}
.adfd-meter-labels .dom{{font-weight:800}}
.adfd-pill{{display:inline-block;font-size:.72rem;font-weight:700;padding:1px 8px;border-radius:999px;
  color:#fff;margin:0 6px;vertical-align:middle;white-space:nowrap}}
.adfd-meter-labels>span{{white-space:nowrap}}
.adfd-meter{{display:flex;justify-content:space-between;height:24px;border-radius:12px;overflow:hidden;
  background:rgba(127,127,127,.18)}}
.adfd-meter .r,.adfd-meter .f{{height:100%;width:var(--w);animation:adfdgrow 1.2s cubic-bezier(.2,.8,.2,1) both}}
.adfd-meter .r{{background:{GREEN}}} .adfd-meter .f{{background:{RED}}}
@keyframes adfdgrow{{from{{width:0}}to{{width:var(--w)}}}}
.adfd-member .name{{font-weight:700;font-size:.95rem}} .adfd-member .w{{font-size:.78rem;opacity:.7}}
.adfd-member .p{{font-size:1.6rem;font-weight:800;margin:6px 0 4px}}
.adfd-mini{{height:8px;border-radius:4px;background:{GREEN}40;overflow:hidden}}
.adfd-mini>div{{height:100%;background:{RED}}}
.adfd-legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.85rem;margin:4px 0 2px}}
.adfd-legend span.sw{{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:6px;
  vertical-align:-2px}}
</style>
"""


@st.cache_resource(show_spinner="Loading models…")
def get_analyzer(run: str):
    return registry.make_analyzer(run, "cpu")


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
def verdict_html(res: dict, source: str) -> str:
    fake = res["verdict"] == "Fake"
    c = RED if fake else GREEN
    bg = "rgba(220,38,38,.08)" if fake else "rgba(22,163,74,.08)"
    word = "FAKE" if fake else "REAL"
    meaning = "likely synthetic or voice-converted speech" if fake else "likely genuine (bona fide) speech"
    return (f"<div class='adfd-verdict' style='--c:{c};--bg:{bg}'><div class='k'>Final verdict · {source}</div>"
            f"<div class='v'>{word}</div><div class='s'>{meaning}. Fake probability "
            f"<b>{100 * res['p_fake']:.1f}%</b> (FAKE at ≥ {DECISION:.0f}%).</div></div>")


def meter_html(p_fake: float) -> str:
    fake = round(100 * p_fake, 1)
    real = round(100 - fake, 1)
    dom_r, dom_f = ("dom", "") if real > fake else ("", "dom")
    pill = lambda c: f"<span class='adfd-pill' style='background:{c}'>dominant</span>"  # noqa: E731
    return (f"<div class='adfd-meter-labels'><span class='r {dom_r}'>Realness {real:.1f}%"
            f"{pill(GREEN) if dom_r else ''}</span><span class='f {dom_f}'>{pill(RED) if dom_f else ''}"
            f"Fakeness {fake:.1f}%</span></div>"
            f"<div class='adfd-meter' role='img' aria-label='Realness {real:.1f} percent, fakeness {fake:.1f} percent'>"
            f"<div class='r' style='--w:{real}%'></div><div class='f' style='--w:{fake}%'></div></div>")


def member_cards(res: dict) -> None:
    cols = st.columns(len(res["members"]))
    for c, m in zip(cols, res["members"]):
        p = 100 * m["p_fake"]
        col = RED if m["vote"] == "Fake" else GREEN
        with c:
            st.markdown(
                f"<div class='adfd-card adfd-member'><div class='name'>{html.escape(m['label'])}</div>"
                f"<div class='w'>weight in the merged score: {100 * m['weight']:.0f}%</div>"
                f"<div class='p' style='color:{col}'>{p:.1f}% fake</div>"
                f"<div class='adfd-mini'><div style='width:{p:.1f}%'></div></div>"
                f"<div style='margin-top:8px;font-size:.85rem'>On its own: "
                f"<span class='adfd-pill' style='background:{col};margin-left:2px'>{m['vote'].upper()}</span></div>"
                "</div>", unsafe_allow_html=True)


def legend_html() -> str:
    items = [(GREEN, "REAL: local fake probability below 50%"),
             (RED, "FAKE: local fake probability 50% or more (suspected region)"),
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
                                         alt.Tooltip("p:Q", title="mean fake %"), alt.Tooltip("sev:N", title="severity")]))
    layers.append(alt.Chart(df).mark_bar(binSpacing=0).encode(
        x=xs, x2="t1:Q", y=ys, y2="hi:Q",
        color=alt.Color("label:N", scale=alt.Scale(domain=list(LABEL_COLOR), range=list(LABEL_COLOR.values())),
                        legend=None),
        tooltip=[alt.Tooltip("time:N", title="time"), alt.Tooltip("label:N", title="local analysis"),
                 alt.Tooltip("fake_pct:Q", title="local fake %")]))
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
    zones = pd.DataFrame([{"y0": DECISION + BORDER, "y1": 100, "c": RED, "z": "FAKE zone"},
                          {"y0": DECISION - BORDER, "y1": DECISION + BORDER, "c": AMBER, "z": "near the line"},
                          {"y0": 0, "y1": DECISION - BORDER, "c": GREEN, "z": "REAL zone"}])
    y_axis = alt.Axis(values=[0, 20, 40, 50, 60, 80, 100], labelOverlap=False, labelFontWeight=alt.ExprRef(
        "datum.value == 50 ? 'bold' : 'normal'"), labelExpr=f"datum.value == {DECISION:g} ? "
        f"'{DECISION:g} ▸ threshold' : datum.label")
    layers = [alt.Chart(zones).mark_rect(opacity=0.10).encode(
                  y=alt.Y("y0:Q", scale=ysc, title="fake probability (%)", axis=y_axis), y2="y1:Q",
                  color=alt.Color("c:N", scale=None, legend=None)),
              alt.Chart(zones).mark_text(align="left", baseline="top", dx=6, dy=4, fontSize=11, fontWeight="bold",
                                         opacity=0.75).encode(
                  x=alt.value(0), y=alt.Y("y1:Q", scale=ysc, axis=y_axis, title="fake probability (%)"), text="z:N",
                  color=alt.Color("c:N", scale=None, legend=None))]
    seg = pd.DataFrame([{"a": s["start_s"], "b": s["end_s"], "name": f"S{i}", "sev": s["severity"],
                         "start": fmt_ts(s["start_s"]), "end": fmt_ts(s["end_s"]), "p": round(100 * s["p_fake"], 1)}
                        for i, s in enumerate(res.get("segments", []), 1)])
    if len(seg):
        tip = [alt.Tooltip("name:N", title="segment"), alt.Tooltip("start:N", title="begins"),
               alt.Tooltip("end:N", title="ends"), alt.Tooltip("p:Q", title="mean fake %"),
               alt.Tooltip("sev:N", title="severity")]
        layers.append(alt.Chart(seg).mark_rect(color=RED, opacity=0.07).encode(x="a:Q", x2="b:Q", tooltip=tip))
        edges = pd.concat([seg.assign(e=seg["a"]), seg.assign(e=seg["b"])])
        layers.append(alt.Chart(edges).mark_rule(color=RED, strokeWidth=1.2, strokeDash=[2, 2]).encode(
            x="e:Q", tooltip=tip))
        layers.append(alt.Chart(seg).mark_text(color=RED, fontWeight="bold", fontSize=11, align="left", dx=3,
                                               baseline="top", dy=2).encode(x="a:Q", y=alt.value(0), text="name:N"))
    thr = pd.DataFrame({"y": [DECISION], "t": [max(dur, 1e-3)], "txt": [f"decision threshold {DECISION:.0f}%"]})
    layers.append(alt.Chart(thr).mark_rule(strokeDash=[6, 4], strokeWidth=1.5, color="#57534e").encode(
        y=alt.Y("y:Q", scale=ysc, axis=y_axis, title="fake probability (%)")))
    if show_members:
        names = [m["label"] for m in res["members"]]
        long = df.melt(id_vars=["t", "run"], value_vars=names, var_name="model", value_name="pct")
        layers.append(alt.Chart(long).mark_line(strokeWidth=1.4, opacity=0.9).encode(
            x=xs, y=alt.Y("pct:Q", scale=ysc, axis=y_axis, title="fake probability (%)"), detail="run:N",
            color=alt.Color("model:N", scale=alt.Scale(domain=names, range=["#57534e", "#78716c", "#a8a29e"]),
                            legend=alt.Legend(title="each model on its own", orient="bottom")),
            strokeDash=alt.StrokeDash("model:N", scale=alt.Scale(domain=names, range=[[6, 3], [2, 2], [8, 3, 2, 3]]),
                                      legend=None)))
    merged = df.dropna(subset=["merged"])
    layers.append(alt.Chart(merged).mark_line(strokeWidth=2.6, color=BLUE, interpolate="monotone").encode(
        x=xs, y=alt.Y("merged:Q", scale=ysc, axis=y_axis, title="fake probability (%)"), detail="run:N"))
    layers.append(alt.Chart(merged).mark_point(filled=True, size=36, opacity=1).encode(
        x=xs, y=alt.Y("merged:Q", scale=ysc, axis=y_axis, title="fake probability (%)"),
        color=alt.Color("here:N", scale=alt.Scale(domain=["Real", "Fake"], range=[GREEN, RED]), legend=None)))
    hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    tips = [alt.Tooltip("time:N", title="time"), alt.Tooltip("here:N", title="local result"),
            alt.Tooltip("merged:Q", title="merged fake %", format=".1f")]
    tips += [alt.Tooltip(f"{m['label']}:Q", title=f"{m['label']} %", format=".0f") for m in res["members"]]
    layers.append(alt.Chart(df).mark_rule(strokeWidth=14, color="#000", opacity=0).encode(
        x=xs, tooltip=tips).add_params(hover))
    layers.append(alt.Chart(df).mark_rule(color="#57534e", strokeWidth=1).encode(x=xs).transform_filter(hover))
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
    st.markdown("#### Where and why: flagged segments")
    if not segs:
        msg = "No part of the recording reaches 50% in the local analysis."
        if res["verdict"] == "Fake":
            msg += (" The FAKE verdict comes from evidence spread over the whole file (each model's full-length "
                    "windows), not from one region.")
        if res["verdict"] == "Real":
            st.success(msg, icon="✅")
        else:
            st.info(msg)
        return
    if res["verdict"] == "Real":
        st.info("The recording as a whole scores REAL, but the regions below reach 50% locally. Listen to them; they "
                "are model-flagged suspicions, not findings.", icon="🔎")
    st.dataframe(segment_rows(res), hide_index=True, width="stretch", column_config={
        "fake probability (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        "main evidence": st.column_config.TextColumn(width="large")})
    st.caption(f"Severity by the segment's mean merged fake probability ({SEVERITY_TEXT}). Timestamps come from "
               f"{res['local_window_s']:.0f}-s analysis windows every {res['step_s']:.1f} s, so boundaries are "
               f"accurate to about ±{res['local_window_s'] / 2:.0f} s.")
    for i, s in enumerate(segs, 1):
        ev = s.get("evidence") or {}
        with st.expander(f"S{i} · {fmt_ts(s['start_s'])}–{fmt_ts(s['end_s'])} · {s['duration_s']:.1f} s · "
                         f"{100 * s['p_fake']:.0f}% fake · {s['severity']} severity", expanded=(i == 1)):
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


def measurement_table(res: dict) -> pd.DataFrame:
    from adfd import forensics

    ref = forensics.load_reference()
    rows = []
    for k, (name, unit) in MEAS_NAMES.items():
        v = res["measurements"].get(k)
        ok = v is not None and np.isfinite(v)
        pct = float(np.interp(v, ref["grid"], ref["percentiles"][k])) if ref and ok else None
        flag = "n/a" if pct is None else "unusual (low)" if pct <= 5 else "unusual (high)" if pct >= 95 else "normal"
        rows.append({"measurement": name, "value": round(float(v), 3) if ok else None, "unit": unit,
                     "percentile among real speech": None if pct is None else round(pct), "reading": flag})
    return pd.DataFrame(rows)


def ensemble_view(name: str, x: np.ndarray, res: dict) -> bytes:
    top_l, top_r = st.columns([5, 4], gap="large")
    with top_l:
        st.markdown(verdict_html(res, "combined result of 3 models"), unsafe_allow_html=True)
        st.markdown(meter_html(res["p_fake"]), unsafe_allow_html=True)
    with top_r:
        st.audio(to_wav_bytes(x), format="audio/wav")
        st.caption(f"{name} · {res['duration_s']:.1f} s · {res['n_local_windows_scored']} local windows scored")
        if res["out_of_distribution"]:
            st.warning("**This doesn't look like normal speech**, so the verdict is much less reliable: "
                       + "; ".join(res["ood_reasons"]) + ".", icon="🚫")
        st.markdown(f"<div style='font-size:.9rem'>{html.escape(res['explanation']['lines'][1])}</div>",
                    unsafe_allow_html=True)
    st.markdown("#### The three models")
    member_cards(res)
    st.caption("Each model scores the whole file with its own windows and calibration. The merged score clips each "
               "model's log-odds (so no single over-confident model dominates) and weights them; two models that "
               "agree always outweigh the third.")

    st.markdown("#### Waveform: where the audio sounds real or fake")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.altair_chart(waveform_chart(x, res), width="stretch")
    st.markdown("#### Fake probability over time")
    show = st.toggle("Show each model's own line", value=False, key=f"members_{name}")
    st.altair_chart(timeline_chart(res, show), width="stretch")
    if not res["time_resolved"]:
        st.info("This clip is too short for time-local analysis (one analysis window), so the timeline has a single "
                "value. Upload a longer recording to see changes over time.")
    else:
        st.caption(f"Each point is the merged fake probability of the {res['local_window_s']:.0f}-s windows covering "
                   f"that moment, scored by all three models (one point every {res['step_s']:.1f} s; window-level "
                   "calibration). Red points are at or above the 50% line, green below. Hover for each model's value. "
                   "The file verdict above uses each model's full-length windows, so a file can be FAKE without any "
                   "single region crossing 50%, or REAL with a few flagged regions.")
    segments_panel(x, res)

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
    return fpng


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
ul{{margin:0;padding-left:1.1rem}}</style></head><body>
<h1>Audio deepfake report</h1>
<p>File: <b>{html.escape(name)}</b> ({res['duration_s']:.1f} s) · {datetime.now():%Y-%m-%d %H:%M} ·
model: {registry.ENSEMBLE_LABEL}</p>
<div class='v'><div style='font-size:2rem;font-weight:800;color:{color}'>{'FAKE' if fake else 'REAL'}</div>
Fake probability <b>{fake_pct:.1f}%</b> (FAKE at ≥ 50%)
<p style='display:flex;justify-content:space-between;margin:.6rem 0 .3rem'><span style='color:{GREEN}'>Realness
{real_pct:.1f}%</span><span style='color:{RED}'>Fakeness {fake_pct:.1f}%</span></p>
<div class='m'><div style='width:{real_pct:.1f}%;background:{GREEN}'></div><div style='width:{fake_pct:.1f}%;
background:{RED}'></div></div></div>
<h2>The three models</h2><table><tr><th>model</th><th>weight</th><th>fake probability</th><th>on its own</th></tr>
{members}</table>
<h2>Fake probability over time</h2><img src='data:image/png;base64,{b64(line_png)}'>
<h2>Flagged segments</h2>{segs}
<h2>Why</h2><ul>{lines}</ul>
<h2>Forensic view</h2><img src='data:image/png;base64,{b64(fpng)}'>
<h2>Whole-file acoustic measurements</h2>{measurement_table(res).to_html(index=False)}
<h2>Limitations</h2><p>{LIMITS}</p></body></html>"""


def analyse_one(name: str, data: bytes, run: str, compare: bool) -> None:
    try:
        x = load_audio(data)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read **{name}** as audio ({e.__class__.__name__}). Try WAV, FLAC, MP3, OGG or MP4.")
        return
    if len(x) < SR // 4:
        st.error("The clip is shorter than 0.25 s; there is nothing to analyse.")
        return
    if run != registry.ENSEMBLE:
        return analyse_single(name, x, run, compare)
    import hashlib

    key = ("analysis", run, hashlib.sha1(data).hexdigest())
    if key not in st.session_state:   # widgets rerun the script; analyse each file only once
        with st.spinner("Analysing with all three models (whole file and local windows)…"):
            st.session_state[key] = get_analyzer(run).analyze(x)
    res = st.session_state[key]
    fpng = ensemble_view(name, x, res)
    st.download_button("Download report (HTML)", ensemble_report(name, res, fpng),
                       file_name=f"{Path(name).stem}_deepfake_report.html", mime="text/html")


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


def analyse_single(name: str, x: np.ndarray, run: str, compare: bool) -> None:
    an = get_analyzer(run)
    with st.spinner("Analysing…"):
        res = an.analyze(x)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(verdict_html(res, "single model"), unsafe_allow_html=True)
        st.markdown(meter_html(res["p_fake"]), unsafe_allow_html=True)
        if res["out_of_distribution"]:
            st.warning("**This doesn't look like normal speech**, so the verdict is much less reliable: "
                       + "; ".join(res["ood_reasons"]) + ".", icon="🚫")
        st.audio(to_wav_bytes(x), format="audio/wav")
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
    others = None
    if compare:
        rows = []
        for r, label in registry.available().items():
            if r == registry.ENSEMBLE:
                continue
            o = get_analyzer(r).analyze(x)
            rows.append({"model": label, "verdict": o["verdict"].upper(), "fake probability %": 100 * o["p_fake"],
                         "out of distribution": o["out_of_distribution"]})
        others = pd.DataFrame(rows)
        st.markdown("**All models**")
        st.dataframe(others, hide_index=True, width="stretch")
        counts = others["verdict"].value_counts()
        st.caption(f"{counts.iloc[0]} of {len(others)} models say **{counts.index[0]}**.")
    st.download_button("Download report (HTML)", report_html(name, res, tl_png, heat_png, others,
                                                              registry.available().get(run, run)),
                       file_name=f"{Path(name).stem}_deepfake_report.html", mime="text/html")


def batch(files: list[tuple[str, bytes]], run: str) -> None:
    an = get_analyzer(run)
    rows = []
    bar = st.progress(0.0, text="Scoring…")
    for i, (name, data) in enumerate(files, 1):
        try:
            x = load_audio(data)
            res = an.analyze(x, explain=False) if run == registry.ENSEMBLE else an.analyze(x)
            row = {"file": name, "verdict": res["verdict"].upper(), "fake_pct": round(100 * res["p_fake"], 2),
                   "real_pct": round(100 * (1 - res["p_fake"]), 2), "duration_s": round(res["duration_s"], 2),
                   "out_of_distribution": res["out_of_distribution"], "ood_reasons": "; ".join(res["ood_reasons"])}
            if run == registry.ENSEMBLE:
                row.update({f"{m['run']}_fake_pct": round(100 * m["p_fake"], 2) for m in res["members"]})
                row["flagged_segments"] = "; ".join(f"{fmt_ts(s['start_s'])}-{fmt_ts(s['end_s'])} "
                                                    f"({100 * s['p_fake']:.0f}%, {s['severity']})"
                                                    for s in res["segments"])
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            rows.append({"file": name, "verdict": "error", "fake_pct": None, "real_pct": None, "duration_s": None,
                         "out_of_distribution": None, "ood_reasons": f"could not read: {e.__class__.__name__}"})
        bar.progress(i / len(files), text=f"Scored {i} of {len(files)}")
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name="deepfake_batch_results.csv",
                       mime="text/csv")


def unpack(uploads) -> list[tuple[str, bytes]]:
    out = []
    for u in uploads:
        data = u.getvalue()
        if u.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if not info.is_dir() and info.filename.lower().rsplit(".", 1)[-1] in AUDIO_TYPES:
                        out.append((info.filename, z.read(info)))
        else:
            out.append((u.name, data))
    return out


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    models = registry.available()
    st.title("🎙️ Audio Deepfake Detector")
    if not models:
        st.error("No trained models found in runs/. Train one first (see README).")
        return
    default = registry.default_run().name
    with st.sidebar:
        st.header("Model")
        names = list(models)
        run = st.selectbox("Detector", names, index=names.index(default) if default in names else 0,
                           format_func=lambda r: models[r])
        compare = st.toggle("Compare all models", value=False, help="Single-model mode: score the file with every "
                            "available model. The merged detector always shows its three models.")
        st.caption(LIMITS)
    single, many = st.tabs(["Analyse a file", "Batch (many files or a ZIP)"])
    with single:
        demos = demo_files()
        src = st.radio("Input", ["Upload audio or video", "Record with microphone", "Demo clip"], horizontal=True,
                       label_visibility="collapsed")
        if src == "Upload audio or video":
            up = st.file_uploader(f"Audio or video file (up to {MAX_MB} MB); for a video, its audio track is analysed",
                                  type=sorted(set(AUDIO_TYPES + VIDEO_TYPES)))
            if up is not None:
                if up.size > MAX_MB * 2**20:
                    st.error(f"File is larger than {MAX_MB} MB.")
                else:
                    analyse_one(up.name, up.getvalue(), run, compare)
        elif src == "Record with microphone":
            rec = st.audio_input("Record a voice sample (a few seconds of speech works best)")
            if rec is not None:
                analyse_one("microphone recording.wav", rec.getvalue(), run, compare)
        elif not demos:
            st.info("No demo clips on this machine (they are read from the DEEP-VOICE dataset folder).")
        else:
            demo = st.selectbox("DEEP-VOICE demonstration clip", ["(none)", *demos])
            if demo != "(none)":
                analyse_one(demo, demos[demo].read_bytes(), run, compare)
    with many:
        ups = st.file_uploader("Audio files or ZIP archives", type=AUDIO_TYPES + ["zip"], accept_multiple_files=True)
        if ups and st.button("Score all", type="primary"):
            files = unpack(ups)
            if files:
                batch(files, run)
            else:
                st.info("No audio files found.")


main()
