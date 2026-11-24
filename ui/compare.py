"""Compare A/B: a reference recording beside a suspected one.

Both files are decoded the same way as on the Detect page. The lab draws them on one time axis with linked or
independent playback; the feature chart compares plain measurements (ui.signal); the detector runs on both only when
asked, and its results are shared with Detect and History.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ui import lab, signal
from ui.icons import icon
from ui.theme import (AMBER, CYAN, GREEN, RED, VERDICT_COLOR, VERDICT_WORD, chip, empty, esc, file_card, fmt_dur,
                      fmt_size, note, ring, section, verdict_chip)

SLOTS = {"A": ("Reference recording", "A known or trusted voice sample.", "a", CYAN),
         "B": ("Suspected recording", "The recording you want to check.", "b", AMBER)}
FEATURES = [("rms_dbfs", "RMS energy", "dBFS", "{:.1f}", "Average loudness of the whole file."),
            ("centroid_hz", "Spectral centroid", "Hz", "{:.0f}", "Centre of mass of the spectrum: brightness."),
            ("rolloff_hz", "Spectral roll-off", "Hz", "{:.0f}", "Frequency below which 85% of the energy lies."),
            ("zcr", "Zero-crossing rate", "per sample", "{:.3f}", "How often the waveform crosses zero: noisiness."),
            ("flatness", "Spectral flatness", "0–1", "{:.3f}", "Noise-like (1) versus tonal (0) spectrum."),
            ("crest_db", "Crest factor", "dB", "{:.1f}", "Peak level above the average level: dynamics.")]


def _swap() -> None:
    s = st.session_state
    s["cmp_A"], s["cmp_B"] = s.get("cmp_B"), s.get("cmp_A")
    s["vg_toast"] = ("Swapped A and B", ":material/swap_horiz:")


def _reset() -> None:
    for k in ("cmp_A", "cmp_B"):
        st.session_state.pop(k, None)
    st.session_state["cmp_n"] = st.session_state.get("cmp_n", 0) + 1
    st.session_state["vg_toast"] = ("Comparison cleared", ":material/restart_alt:")


def _clear(slot: str) -> None:
    st.session_state.pop(f"cmp_{slot}", None)
    st.session_state["cmp_n"] = st.session_state.get("cmp_n", 0) + 1


def _key(data: bytes, run: str) -> tuple:
    return ("analysis", run, hashlib.sha1(data).hexdigest())


def _feature_html(fa: dict | None, fb: dict | None) -> str:
    cards = []
    for i, (k, nm, unit, fmt, what) in enumerate(FEATURES):
        va = fa.get(k) if fa else None
        vb = fb.get(k) if fb else None
        vals = [abs(v) for v in (va, vb) if v is not None and np.isfinite(v)]
        top = max(vals) if vals else 1.0
        rows = ""
        for tag, v, col in (("A", va, CYAN), ("B", vb, AMBER)):
            if v is None or not np.isfinite(v):
                rows += (f"<div class='vg-mbar'><div class='l'><span>{tag}</span><b style='color:#5d6b82'>–</b></div>"
                         f"<div class='t'></div></div>")
                continue
            if unit == "dBFS":                       # level: bar length grows with loudness on a -60..0 dBFS scale
                w = 100 * min(max((v + 60) / 60, 0.0), 1.0)
            else:
                w = 100 * abs(v) / top if top else 0
            rows += (f"<div class='vg-mbar' title='{tag}: {fmt.format(v)} {unit}'><div class='l'><span>{tag}</span>"
                     f"<b>{fmt.format(v)} <span style='color:#8b98ad;font-weight:400'>{esc(unit)}</span></b></div>"
                     f"<div class='t'><div style='--w:{w:.1f}%;--c:{col};--d:{.05 * i:.2f}s'></div></div></div>")
        diff = ""
        if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
            if unit in ("dBFS", "dB"):
                diff = f"B − A: {vb - va:+.1f} dB"
            elif abs(va) > 1e-9:
                diff = f"B vs A: {100 * (vb - va) / abs(va):+.0f}%"
        cards.append(f"<div class='vg-card' style='padding:16px'><div style='display:flex;justify-content:space-between;"
                     f"gap:8px'><h4 style='margin:0'>{esc(nm)}</h4><span style='font-size:.74rem;color:#8b98ad'>"
                     f"{esc(diff)}</span></div><p style='margin:2px 0 12px'>{esc(what)}</p>"
                     f"<div style='display:grid;gap:10px'>{rows}</div></div>")
    return cards


def _mfcc_chart(fa: dict | None, fb: dict | None):
    import altair as alt

    rows = []
    for tag, f in (("A · reference", fa), ("B · suspected", fb)):
        if f and f.get("mfcc_mean"):
            rows += [{"coefficient": i, "mean": v, "recording": tag} for i, v in enumerate(f["mfcc_mean"])]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=40, filled=True), strokeWidth=2,
                                   interpolate="monotone").encode(
        x=alt.X("coefficient:O", title="MFCC coefficient", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean:Q", title="mean value"),
        color=alt.Color("recording:N", scale=alt.Scale(domain=["A · reference", "B · suspected"], range=[CYAN, AMBER]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["recording", "coefficient", alt.Tooltip("mean:Q", format=".2f")]).properties(height=260)


def _verdict_card(slot: str, res: dict | None, det) -> str:
    title, _, tone, col = SLOTS[slot]
    if res is None:
        return (f"<div class='vg-model' style='height:100%'><div class='top'><div><div class='nm'>{slot} · "
                f"{esc(title)}</div><div class='sub'>Not analyzed yet</div></div>{chip('No verdict', 'mute')}</div>"
                f"<div style='font-size:.84rem;color:#8b98ad;margin-top:14px'>Run the detector to get a REAL / "
                f"SYNTHETIC verdict for this recording.</div></div>")
    c = VERDICT_COLOR[res["verdict"]]
    p = 100 * res["p_fake"]
    t = det.trust_of(res)
    bars = "".join(f"<div class='vg-mbar'><div class='l'><span>{lb}</span><b>{v:.1f}%</b></div><div class='t'>"
                   f"<div style='--w:{v:.1f}%;--c:{cc}'></div></div></div>"
                   for lb, v, cc in (("REAL", 100 - p, GREEN), ("SYNTHETIC", p, RED)))
    return (f"<div class='vg-model' style='height:100%;border-top:2px solid {col}'><div class='top'><div>"
            f"<div class='nm'>{slot} · {esc(title)}</div><div class='sub'>{esc(det.model_label(res))} · "
            f"{esc(t['level'])} held-out trust</div></div>{verdict_chip(res['verdict'])}</div>"
            f"<div class='mid'>{ring(p, c, 84, 'synthetic', 8, small=True)}<div class='bars'>{bars}</div></div>"
            f"<div class='foot'><span>Confidence <b style='color:{c}'>{det.confidence_pct(res):.1f}%</b></span>"
            f"<span>{VERDICT_WORD[res['verdict']]}</span></div></div>")


def _summary(xa, xb, fa, fb, ra, rb) -> list[str]:
    out = []
    from adfd.inference import SR

    if xa is not None and xb is not None:
        da, db = len(xa) / SR, len(xb) / SR
        out.append(f"B is {abs(db - da):.1f} s {'longer' if db > da else 'shorter'} than A ({fmt_dur(db)} vs "
                   f"{fmt_dur(da)})." if abs(db - da) >= .05 else f"Both recordings last {fmt_dur(da)}.")
    if fa and fb:
        d = fb["rms_dbfs"] - fa["rms_dbfs"]
        out.append(f"B is {abs(d):.1f} dB {'louder' if d > 0 else 'quieter'} on average than A.")
        for k, nm in (("centroid_hz", "spectral centroid"), ("rolloff_hz", "spectral roll-off")):
            r = 100 * (fb[k] - fa[k]) / max(abs(fa[k]), 1e-9)
            out.append(f"B's {nm} is {abs(r):.0f}% {'higher' if r > 0 else 'lower'} than A's "
                       f"({fb[k]:.0f} vs {fa[k]:.0f} Hz).")
    if ra and rb:
        va, vb = VERDICT_WORD[ra["verdict"]], VERDICT_WORD[rb["verdict"]]
        out.append(f"The detector calls A {va} ({100 * ra['p_fake']:.0f}% synthetic) and B "
                   f"{vb} ({100 * rb['p_fake']:.0f}% synthetic)"
                   + (", the same verdict." if va == vb else ", different verdicts."))
    out.append("Differences in level, brightness or bandwidth also come from microphones, rooms and codecs; they are "
               "not evidence of synthesis on their own.")
    return out


def page(det) -> None:
    from adfd import registry
    from adfd.inference import SR

    s = st.session_state
    n = s.get("cmp_n", 0)
    a, b = s.get("cmp_A"), s.get("cmp_B")
    t1, t2, _ = st.columns([1, 1, 4])
    t1.button("Swap A ↔ B", icon=":material/swap_horiz:", on_click=_swap, disabled=not (a or b), width="stretch")
    t2.button("Reset", icon=":material/restart_alt:", on_click=_reset, disabled=not (a or b), width="stretch")

    xs, metas = {}, {}
    cols = st.columns(2, gap="medium")
    for (slot, (title, sub, tone, col)), c in zip(SLOTS.items(), cols):
        with c:
            st.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0 10px'><span class='vg-chip "
                        f"solid' style='background:{col}'>{slot}</span><div><div style='font-weight:600'>{esc(title)}"
                        f"</div><div style='font-size:.8rem;color:#8b98ad'>{esc(sub)}</div></div></div>",
                        unsafe_allow_html=True)
            item = s.get(f"cmp_{slot}")
            if item is None:
                up = st.file_uploader(f"Drop recording {slot} here", type=sorted(set(det.AUDIO_TYPES + det.VIDEO_TYPES)),
                                      key=f"cmp_up_{slot}_{n}", help=f"Up to {det.MAX_MB} MB; video uses its audio track.")
                if up is not None:
                    if up.size > det.MAX_MB * 2**20:
                        st.error(f"The file is larger than {det.MAX_MB} MB.", icon="🛑")
                    else:
                        s[f"cmp_{slot}"] = {"name": up.name, "data": up.getvalue()}
                        st.rerun()
                continue
            x = det.prepare(item["name"], item["data"])
            if x is None:
                st.button("Remove", key=f"cmp_rm_{slot}", on_click=_clear, args=(slot,), icon=":material/close:")
                continue
            xs[slot] = x
            ext = Path(item["name"]).suffix.lstrip(".").upper() or "AUDIO"
            res = s.get(_key(item["data"], registry.MODEL))
            tag = verdict_chip(res["verdict"]) if res else chip("Loaded", "cyan")
            st.markdown(file_card(item["name"], [("Size", fmt_size(len(item["data"]))), ("Format", ext),
                                                 ("Duration", fmt_dur(len(x) / SR)), ("Analysed as", "16 kHz mono")],
                                  tone, tag=tag), unsafe_allow_html=True)
            st.button("Replace file", key=f"cmp_rm_{slot}", on_click=_clear, args=(slot,), icon=":material/upload:",
                      help="Remove this recording so you can drop another")

    if not xs:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        empty("Load two recordings to compare them",
              "Drop a reference recording into A and the recording you want to check into B. Both are drawn on one "
              "time axis with linked playback, measured the same way, and can be run through the detector.",
              "compare")
        return

    section("Side-by-side signal view", "Both recordings on one time axis. Linked: play, seek and select both together; "
            "Independent: each lane plays alone.", "audio-waveform")
    lanes = []
    for slot, x in xs.items():
        item = s[f"cmp_{slot}"]
        res = s.get(_key(item["data"], registry.MODEL))
        lanes.append(lab.lane(x, SR, item["name"], slot, res))
    lab.render(lanes, key="cmp_lab")

    section("Model verdicts", "WavLM + CNN-BiLSTM-Att (E7a) on each recording.", "cpu")
    ens_ok = registry.MODEL in registry.available()
    todo = [sl for sl in xs if _key(s[f"cmp_{sl}"]["data"], registry.MODEL) not in s]
    if ens_ok and todo:
        if st.button(f"Run the detector on {' and '.join(todo)}", type="primary", icon=":material/graphic_eq:"):
            an = det.get_analyzer(registry.MODEL)
            for sl in todo:
                item = s[f"cmp_{sl}"]
                with st.spinner(f"Analyzing {sl} · {item['name']}…"):
                    try:
                        res = det.run_analysis(registry.MODEL, an, xs[sl])
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Recording {sl} could not be analysed ({e.__class__.__name__}: {e}).", icon="🛑")
                        continue
                k = _key(item["data"], registry.MODEL)
                s[k] = res
                det.remember(item["name"], registry.MODEL, res, k[2])
            s["vg_toast"] = ("Comparison verdicts ready", ":material/task_alt:")
            st.rerun()
    elif not ens_ok:
        note("The detection model is not available on this machine, so no verdicts can be shown.")
    rs = {sl: s.get(_key(s[f"cmp_{sl}"]["data"], registry.MODEL)) for sl in xs}
    vc = st.columns(2, gap="medium")
    for c, sl in zip(vc, ("A", "B")):
        if sl in xs:
            c.markdown(_verdict_card(sl, rs.get(sl), det), unsafe_allow_html=True)
        else:
            c.markdown(f"<div class='vg-empty' style='padding:22px'><div class='ib'>{icon('upload', 20)}</div>"
                       f"<h4>No recording {sl}</h4><p>Load one above to compare.</p></div>", unsafe_allow_html=True)

    feats = {sl: signal.features(x, SR) for sl, x in xs.items()}
    fa, fb = feats.get("A"), feats.get("B")
    section("Feature comparison", "Plain acoustic measurements of each whole recording (frames with signal only).",
            "chart")
    cards = _feature_html(fa, fb)
    for i in range(0, len(cards), 3):
        for c, h in zip(st.columns(3, gap="medium"), cards[i:i + 3]):
            c.markdown(h, unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    ch = _mfcc_chart(fa, fb)
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        st.markdown("<div style='font-weight:600;font-size:.92rem;margin:14px 0 8px'>MFCC profile · mean of 13 "
                    "coefficients</div>", unsafe_allow_html=True)
        if ch is not None:
            st.altair_chart(ch, width="stretch")
        else:
            note("MFCCs are not available for these recordings.")
    with c2:
        st.markdown("<div style='font-weight:600;font-size:.92rem;margin:14px 0 8px'>Comparison summary</div>",
                    unsafe_allow_html=True)
        items = "".join(f"<li style='margin-bottom:8px'>{esc(t)}</li>" for t in _summary(
            xs.get("A"), xs.get("B"), fa, fb, rs.get("A"), rs.get("B")))
        st.markdown(f"<div class='vg-panel'><ul style='margin:0;padding-left:1.1rem;font-size:.88rem;color:#c4cdd9;"
                    f"line-height:1.5'>{items}</ul></div>", unsafe_allow_html=True)
    rows = []
    for k, nm, unit, fmt, _ in FEATURES:
        rows.append({"feature": nm, "unit": unit, "A": None if fa is None else round(fa[k], 4),
                     "B": None if fb is None else round(fb[k], 4)})
    with st.expander("Exact values"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
