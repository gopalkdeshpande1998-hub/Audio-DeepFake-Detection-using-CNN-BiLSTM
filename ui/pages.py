"""One render function per page. The detector pages import PyTorch lazily, the research pages never do."""

from __future__ import annotations

import html
import sys

import pandas as pd
import streamlit as st

from ui import data, nav
from ui import theme
from ui.theme import AMBER, BLUE, GREEN, RED, hero, row, section, stat


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _detector():
    """Import the detector module (PyTorch, model code) with a visible notice on the first load."""
    if "ui.detector" not in sys.modules:
        with st.spinner("Starting the detector: loading PyTorch and the model code (about 10–20 s the first time)…"):
            from ui import detector
        return detector
    from ui import detector
    return detector


def _model_picker(det) -> str | None:
    from adfd import registry

    models = registry.available()
    if not models:
        st.error("The detection model is missing: `runs/e7a_wavlm_crnn_s42/best.pt` was not found (see Help).",
                 icon="🛑")
        return None
    run = registry.MODEL
    with st.sidebar:
        st.markdown("**Detector**")
        st.markdown(f"<div class='vg-panel' style='padding:10px 12px;font-size:.85rem'><div style='font-weight:600'>"
                    f"{html.escape(models[run])}</div><div style='color:#8b98ad;font-size:.75rem;margin-top:2px'>"
                    "Sole detection model · WavLM-Base+ features + CNN-BiLSTM with attention</div></div>",
                    unsafe_allow_html=True)
        gpu = det.gpu_device()                       # "cuda" (NVIDIA) or "mps" (Apple silicon Mac)
        if gpu:
            use_gpu = st.toggle("Use the GPU", value=det.default_device() != "cpu", key="use_gpu",
                                help="Much faster than the CPU. Turn off if another GPU job is running.")
            st.session_state["device"] = gpu if use_gpu else "cpu"
        else:
            st.session_state["device"] = "cpu"
        st.caption(f"Running on: **{det.device_label()}**")
        st.caption(det.LIMITS)
    return run


def _pct(v, nd=1) -> str:
    return "–" if v is None else f"{v:.{nd}f}%"



# ---------------------------------------------------------------------------------------
# Use
# ---------------------------------------------------------------------------------------
def _toast() -> None:
    t = st.session_state.pop("vg_toast", None)
    if t:
        st.toast(t[0], icon=t[1])


def detect() -> None:
    det = _detector()
    run = _model_picker(det)
    hero("Forensic analysis", "Analyze a recording",
         "Upload, preview the waveform, analyze with WavLM + CNN-BiLSTM-Att, then explore the evidence behind the "
         "verdict.",
         "audio-waveform")
    _toast()
    step_ph = st.empty()
    if run is None:
        return
    n = st.session_state.get("input_n", 0)
    with st.expander("Tips for a reliable result", icon=":material/lightbulb:"):
        st.markdown("- Use **5 s or more of speech**; under 2 s the verdict is weak.\n"
                    "- Prefer the **original file** over a re-recording or a heavily compressed copy.\n"
                    "- One speaker at a time, little music or background noise.\n"
                    "- Read the **trust panel** under the verdict and listen to every flagged segment.")
    got = None
    up = st.file_uploader(f"Drag and drop an audio or video file, up to {det.MAX_MB} MB (for a video, its audio "
                          "track is used)", type=sorted(set(det.AUDIO_TYPES + det.VIDEO_TYPES)), key=f"up_{n}")
    if up is not None:
        if up.size > det.MAX_MB * 2**20:
            st.error(f"The file is larger than {det.MAX_MB} MB. Cut or compress it first.", icon="🛑")
        else:
            got = (up.name, up.getvalue())
    if got is None:
        step_ph.markdown(theme.stepper_html(1), unsafe_allow_html=True)
        theme.empty("No recording loaded", "Drag a file into the drop area above or browse for one. You can preview "
                    "it before anything is analysed.", "upload")
        return
    det.analyse_one(got[0], got[1], run, step_ph=step_ph)


def compare() -> None:
    det = _detector()
    hero("Side-by-side forensics", "Compare a reference with a suspected recording",
         "Load a known voice into A and the recording you doubt into B. Listen to both together, compare their "
         "spectra and measurements, and run the detector on each.", "compare")
    _toast()
    from ui import compare as cmp

    cmp.page(det)


def batch() -> None:
    det = _detector()
    run = _model_picker(det)
    hero("Batch scoring", "Score many files",
         "Upload audio files or ZIP archives; each file gets one row with its verdict, both probabilities, "
         "confidence and flagged segments, ready to download as CSV.", "table")
    if run is None:
        return
    ups = st.file_uploader("Audio files or ZIP archives", type=det.AUDIO_TYPES + ["zip"], accept_multiple_files=True)
    if ups and st.button("Score all files", type="primary", icon=":material/play_arrow:"):
        files, skipped = det.unpack(ups)
        for m in skipped:
            st.warning(f"Skipped {m}.")
        if not files:
            st.info("No audio files found.")
        else:
            st.session_state["batch_result"] = det.batch(files, run)
    df = st.session_state.get("batch_result")
    if df is not None:
        ok = df[df["verdict"] != "ERROR"]
        row([stat(str(len(df)), "files"), stat(str((ok["verdict"] == "SYNTHETIC").sum()), "called SYNTHETIC",
                                               color=RED),
             stat(str((ok["verdict"] == "REAL").sum()), "called REAL", color=GREEN),
             stat(str((df["verdict"] == "ERROR").sum()), "could not be analysed", color=AMBER)])
        st.markdown("")
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "fake_pct": st.column_config.ProgressColumn("synthetic %", min_value=0, max_value=100, format="%.1f%%")})
        st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name="deepfake_batch_results.csv",
                           mime="text/csv", icon=":material/download:", on_click="ignore")


def history() -> None:
    hero("Session history", "This session's analyses",
         "Every recording analysed in this browser tab, newest first.", "history")
    hist = st.session_state.get("history", [])
    if not hist:
        theme.empty("Nothing analysed yet", "Analyses you run in this browser session are listed here.", "history")
        st.page_link(nav.get("detect"), label="Analyse a recording", icon=":material/graphic_eq:")
        return
    df = pd.DataFrame(hist).drop(columns=["key"])
    dcol = df["verdict"].replace({"FAKE": "SYNTHETIC"})
    low = df["trust"].isin(["very low", "low"]).sum() if "trust" in df else 0
    row([stat(str(len(df)), "analyses"), stat(str((dcol == "SYNTHETIC").sum()), "called SYNTHETIC", color=RED),
         stat(str((dcol == "REAL").sum()), "called REAL", color=GREEN),
         stat(str(low), "near the 50% line", color=AMBER)])
    st.markdown("")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "fake_pct": st.column_config.ProgressColumn("synthetic %", min_value=0, max_value=100, format="%.1f%%")})
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Download CSV", df.to_csv(index=False).encode(), file_name="deepfake_session_history.csv",
                       mime="text/csv", icon=":material/download:", on_click="ignore")
    if c2.button("Clear history", icon=":material/delete:"):
        st.session_state["history"] = []
        st.rerun()
    st.caption("Kept in this browser tab's memory only; closing or reloading the tab clears it.")


# ---------------------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------------------
METHOD_COLOR = {"App verdict (E7a)": "#3987e5", "New model: whole clip": "#c98500",
                "New model: any FAKE segment": "#9085e9"}     # validated: CVD-safe and >= 3:1 on the dark panel
METHOD_NOTE = {"App verdict (E7a)": "the verdict the app shows today (whole-clip model, SYNTHETIC at 50%)",
               "New model: whole clip": "the segment model's own whole-clip output",
               "New model: any FAKE segment": "a clip is FAKE if the segment model marks any part of it FAKE"}
SHOWN_SETS = ("ASVspoof eval",)          # held-out test sets shown on the website (In-the-Wild removed)
AXIS = dict(labelColor="#8b98ad", titleColor="#8b98ad", gridColor="#1f2a3c", domainColor="#2b3a52",
            tickColor="#2b3a52", labelFont="Inter", titleFont="Inter")


def _grouped_bars(df: pd.DataFrame, x: str, y: str, title_y: str, y_max: float | None, tips: list, height: int = 300,
                  label: str | None = None, x_sort=None):
    """Grouped bars: one group per x category, one bar per method (fixed method colours), hover tooltip."""
    import altair as alt

    methods = list(METHOD_COLOR)
    enc_x = alt.X(f"{x}:N", title=None, sort=x_sort, axis=alt.Axis(labelAngle=0, labelLimit=220, **AXIS))
    enc_y = alt.Y(f"{y}:Q", title=title_y, scale=alt.Scale(domain=[0, y_max]) if y_max else alt.Undefined,
                  axis=alt.Axis(values=[0, 20, 40, 60, 80, 100] if y_max == 110 else alt.Undefined, **AXIS))
    base = alt.Chart(df).encode(x=enc_x, xOffset=alt.XOffset("method:N", sort=methods))
    colour = alt.Color("method:N", sort=methods, scale=alt.Scale(domain=methods, range=list(METHOD_COLOR.values())),
                       legend=alt.Legend(orient="bottom", title=None, labelColor="#c4cdd9", labelFont="Inter",
                                         labelFontSize=12, symbolType="square", symbolSize=110, labelLimit=0,
                                         columns=3, columnPadding=32, symbolOffset=0, offset=14))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke="#111827", strokeWidth=2).encode(
        y=enc_y, color=colour, tooltip=tips)
    layers = [bars]
    if label:
        layers.append(base.mark_text(dy=-7, fontSize=11, font="Inter", color="#c4cdd9").encode(y=enc_y, text=f"{label}:N"))
    return alt.layer(*layers).properties(height=height)


def _from_counts(fp: int, tn: int, tp: int, fn: int) -> dict:
    truth = [False] * (fp + tn) + [True] * (tp + fn)
    pred = [True] * fp + [False] * tn + [True] * tp + [False] * fn
    return data.confusion_metrics(truth, pred)


def results() -> None:
    import altair as alt

    hero("Research", "Results",
         "How well VoiceGuard tells real from fake speech on recordings it never saw during training: 100 real and "
         "100 fake clips from the ASVspoof 2019 evaluation set.", "chart")
    rows = data.sample_test()
    if not rows:
        st.info("The sample test results (experiments/sample_test_100/results.json) are not on this machine.")
        return
    truth = [r["truth"] == "fake" for r in rows]
    m = {name: data.confusion_metrics(truth, [bool(r[k]) for r in rows]) for k, name in data.SAMPLE_METHODS.items()}
    n_real, n_fake = sum(not t for t in truth), sum(truth)
    spk = len({r["speaker"] for r in rows if r["truth"] == "real"})
    systems = sorted({r["system"] for r in rows if r["truth"] == "fake"})
    app, new = m["App verdict (E7a)"], m["New model: whole clip"]

    row([stat(f"{app['accuracy']:.1f}%", "App verdict accuracy", "E7a, the verdict the app shows", ico="shield-check"),
         stat(f"{new['accuracy']:.1f}%", "New model accuracy", "segment model, whole-clip output", ico="gauge"),
         stat(f"{app['real_called_fake_per100']:.0f} / {new['real_called_fake_per100']:.0f}",
              "Real clips called FAKE, per 100", "app verdict / new model", GREEN, "check"),
         stat(f"{app['fake_called_real_per100']:.0f} / {new['fake_called_real_per100']:.0f}",
              "Fake clips called REAL, per 100", "app verdict / new model", RED, "alert")])
    st.markdown("")
    theme.note(f"<b>Test set:</b> {n_real} real recordings from {spk} speakers and {n_fake} fake recordings from "
               f"{len(systems)} fake-speech systems ({systems[0]} to {systems[-1]}), all from the ASVspoof 2019 "
               "evaluation set, which no model here was trained on. FAKE is the class being detected. "
               "With 100 clips per class one mistake moves a rate by 1 point.", "info")

    section("Accuracy, F1, precision and recall", "Higher is better. Precision: of the clips called FAKE, how many were "
            "fake. Recall: of the fake clips, how many were caught.", "chart")
    names = ["Accuracy", "F1 score", "Precision", "Recall"]
    df = pd.DataFrame([{"metric": k, "method": name, "value": round(v[key], 1), "label": f"{v[key]:.1f}"}
                       for name, v in m.items() for k, key in zip(names, ("accuracy", "f1", "precision", "recall"))])
    st.altair_chart(_grouped_bars(df, "metric", "value", "%", 110,
                                  ["method", "metric", alt.Tooltip("value:Q", title="%", format=".1f")], label="label",
                                  x_sort=names),
                    width="stretch")

    section("Mistakes per 100 recordings", "Lower is better. Left: real recordings wrongly called FAKE. Right: fake "
            "recordings missed and called REAL.", "alert")
    kinds = ["Real clips called FAKE", "Fake clips called REAL"]
    df = pd.DataFrame([{"kind": kind, "method": name, "per100": round(v[key], 1),
                        "count": f"{v[cnt]} of {v[tot]}", "label": str(v[cnt])}
                       for name, v in m.items()
                       for kind, key, cnt, tot in ((kinds[0], "real_called_fake_per100", "fp", "n_real"),
                                                   (kinds[1], "fake_called_real_per100", "fn", "n_fake"))])
    st.altair_chart(_grouped_bars(df, "kind", "per100", "per 100 recordings", None,
                                  ["method", "kind", alt.Tooltip("count:N", title="mistakes"),
                                   alt.Tooltip("per100:Q", title="per 100", format=".1f")], label="label", x_sort=kinds),
                    width="stretch")

    section("Which fakes are missed", "Fake clips called REAL, by the fake-speech system that made them (7 to 9 clips "
            "each). A10 is a neural text-to-speech system that every model here finds hard.", "search")
    per_sys = []
    for k, name in data.SAMPLE_METHODS.items():
        for sysid in systems:
            fk = [r for r in rows if r["truth"] == "fake" and r["system"] == sysid]
            miss = sum(not r[k] for r in fk)
            per_sys.append({"system": sysid, "method": name, "missed": miss, "of": f"{miss} of {len(fk)}",
                            "label": str(miss) if miss else ""})
    st.altair_chart(_grouped_bars(pd.DataFrame(per_sys), "system", "missed", "fake clips missed", 10,
                                  ["method", "system", alt.Tooltip("of:N", title="missed")], height=260,
                                  label="label", x_sort=systems),
                    width="stretch")

    section("All numbers", "Counts behind the charts (FAKE = positive class).", "table")
    st.dataframe(pd.DataFrame([{"method": name, "what it is": METHOD_NOTE[name],
                                "accuracy %": round(v["accuracy"], 1), "F1 %": round(v["f1"], 1),
                                "precision %": round(v["precision"], 1), "recall %": round(v["recall"], 1),
                                "fakes caught": v["tp"], "real called FAKE": v["fp"], "fakes missed": v["fn"],
                                "real correct": v["tn"]} for name, v in m.items()]),
                 hide_index=True, width="stretch",
                 column_config={"what it is": st.column_config.TextColumn(width="large")})

    section("On the large held-out tests", "The same measures on tens of thousands of clips; these are the reliable "
            "long-run figures.", "layers")
    big = []
    ea = data.error_analysis()
    if ea:
        for name, d in ea["sets"].items():
            if name not in SHOWN_SETS:
                continue
            o = d["overall"]
            fp = round(o["real_called_fake_pct"] / 100 * o["n_real"])
            fn = round(o["fake_called_real_pct"] / 100 * o["n_fake"])
            big.append({"model": "App verdict (E7a)", "test set": name, "clips": o["n"],
                        **_from_counts(fp, o["n_real"] - fp, o["n_fake"] - fn, fn)})
    tt = data.temporal_test()
    if tt:
        c = tt["t1"]["clip_confusion"]
        big.append({"model": "New model: whole clip", "test set": "PartialSpoof eval (sample)", "clips": tt["utterances"],
                    **_from_counts(c["fp_real_called_fake"], c["tn_real"], c["tp_fake"], c["fn_fake_called_real"])})
    if big:
        st.dataframe(pd.DataFrame([{"model": b["model"], "test set": b["test set"], "clips": b["clips"],
                                    "accuracy %": round(b["accuracy"], 1), "F1 %": round(b["f1"], 1),
                                    "real called FAKE per 100": round(b["real_called_fake_per100"], 1),
                                    "fakes called REAL per 100": round(b["fake_called_real_per100"], 1)} for b in big]),
                     hide_index=True, width="stretch")
        st.caption("ASVspoof eval: studio speech, fake-speech systems unseen in training. PartialSpoof eval: the "
                   "new model's own test set, built from the same studio recordings, so it is not directly "
                   "comparable.")


def errors() -> None:
    import altair as alt

    hero("Research", "Error analysis", "Where WavLM + CNN-BiLSTM-Att goes wrong on held-out data, and when to "
         "trust a verdict.", "search")
    ea = data.error_analysis()
    if not ea:
        st.info("Run `python scripts/evaluation/error_analysis.py` to build this page (it reads the stored scores; about 1 min).")
        return
    st.markdown("Where the app's model, WavLM + CNN-BiLSTM-Att (E7a), goes wrong on held-out data at its "
                "REAL/SYNTHETIC line of 50%. Built from its stored per-clip scores by `scripts/evaluation/error_analysis.py`.")
    names = [n for n in ea["sets"] if n in SHOWN_SETS]
    if not names:
        st.info("No held-out test set to show.")
        return
    name = (st.segmented_control("Test set", names, default=names[0], key="ea_set") or names[0]) if len(names) > 1         else names[0]
    if len(names) == 1:
        st.caption(f"Test set: {name} (ASVspoof 2019 LA evaluation: studio speech, fake-speech systems A07–A19 never "
                   "seen in training).")
    d = ea["sets"][name]
    o = d["overall"]
    row([stat(f"{o['n']:,}", "clips" if "windows" not in name else "4-s windows"),
         stat(_pct(o["accuracy_pct"]), "accuracy"),
         stat(_pct(o["real_called_fake_pct"]), "real called SYNTHETIC", color=RED),
         stat(_pct(o["fake_called_real_pct"]), "fakes called REAL", color=AMBER)])

    st.markdown("#### When to trust a verdict")
    conf = pd.DataFrame([{"confidence": k, **v} for k, v in d["reliability"]["by_confidence"].items()])
    with st.container():
        st.altair_chart(alt.Chart(conf).mark_bar(color=BLUE).encode(
            x=alt.X("confidence:N", sort=None, title="confidence of the model probability",
                    axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("accuracy_pct:Q", title="accuracy (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["confidence", alt.Tooltip("accuracy_pct:Q", format=".1f"), "n"]).properties(height=260),
            width="stretch")
    st.caption("The Detect page looks up this table for every verdict and shows it under the verdict.")

    if "by_attack" in d:
        st.markdown("#### Per attack (ASVspoof A07–A19)")
        att = pd.DataFrame([{"attack": a, "fakes called REAL %": v["fake_called_real_pct"], "n": v["n"]}
                            for a, v in d["by_attack"].items()])
        st.altair_chart(alt.Chart(att).mark_bar(color=RED).encode(
            x=alt.X("attack:N", title=None), y=alt.Y("fakes called REAL %:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=list(att.columns)).properties(height=240), width="stretch")
        st.dataframe(att, hide_index=True, width="stretch")
        st.caption("'Fakes called REAL' uses the app's 50% line; n is the number of fake recordings from that attack. "
                   "A10 is hard for every model in this project.")
    st.markdown("#### By clip duration")
    st.dataframe(pd.DataFrame([{"duration": k, **v} for k, v in d["by_duration"].items()]), hide_index=True,
                 width="stretch")
    if "by_speaker" in d:
        st.markdown("#### Hardest speakers")
        st.dataframe(pd.DataFrame([{"speaker": k, **v} for k, v in d["by_speaker"].items()]).head(15),
                     hide_index=True, width="stretch")
    if "by_recording" in d:
        st.markdown("#### Per recording")
        st.dataframe(pd.DataFrame(d["by_recording"]), hide_index=True, width="stretch", column_config={
            "p_fake_pct": st.column_config.ProgressColumn("P(fake) %", min_value=0, max_value=100, format="%.1f")})
    st.markdown("#### Worst mistakes")
    w1, w2 = st.tabs(["Fakes called REAL (false accepts)", "Real called FAKE (false rejects)"])
    w1.dataframe(pd.DataFrame(d["false_accepts"]), hide_index=True, width="stretch")
    w2.dataframe(pd.DataFrame(d["false_rejects"]), hide_index=True, width="stretch")
