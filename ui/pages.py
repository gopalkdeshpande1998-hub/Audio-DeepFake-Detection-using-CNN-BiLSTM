"""One render function per page. The detector pages import PyTorch lazily, the research pages never do."""

from __future__ import annotations

import html
import sys

import pandas as pd
import streamlit as st

from ui import content, data, nav
from ui import theme
from ui.icons import icon
from ui.theme import AMBER, BLUE, CYAN, GREEN, RED, card, hero, row, section, stat


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
        if det.gpu_available():
            use_gpu = st.toggle("Use the GPU", value=det.default_device().startswith("cuda"), key="use_gpu",
                                help="About 5–10× faster than the CPU. Turn off if another GPU job is running.")
            st.session_state["device"] = "cuda" if use_gpu else "cpu"
        else:
            st.session_state["device"] = "cpu"
        st.caption(f"Running on: **{det.device_label()}**")
        st.caption(det.LIMITS)
    return run


def _ea_set(name: str) -> dict | None:
    ea = data.error_analysis()
    return ea["sets"].get(name) if ea else None


def _pct(v, nd=1) -> str:
    return "–" if v is None else f"{v:.{nd}f}%"


def _md_file(rel: str) -> None:
    txt = data.read_text(rel)
    if txt is None:
        st.info(f"`{rel}` is not on this machine.")
    else:
        st.markdown(data.strip_title(txt))


# ---------------------------------------------------------------------------------------
# Use
# ---------------------------------------------------------------------------------------
def _toast() -> None:
    t = st.session_state.pop("vg_toast", None)
    if t:
        st.toast(t[0], icon=t[1])


WORKFLOW = [("upload", "Upload", "Drop audio or video, record with the microphone, or pick a demo clip."),
            ("audio-waveform", "Preview", "Listen and inspect the waveform and spectrogram before analysing."),
            ("cpu", "Analyze", "WavLM + CNN-BiLSTM-Att scores the file on this computer; the bar shows real "
                               "progress."),
            ("shield-check", "Results", "REAL or SYNTHETIC, both probabilities, confidence and held-out trust."),
            ("search", "Explore", "Spectrogram lab, explainability center, flagged segments, report.")]


def _recent_html(hist: list[dict]) -> str:
    rows = []
    for h in hist[:5]:
        word = {"FAKE": "SYNTHETIC"}.get(h["verdict"], h["verdict"])
        kind = "syn" if word == "SYNTHETIC" else "real"
        rows.append(f"<div style='display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:14px;align-items:center;"
                    f"padding:11px 0;border-bottom:1px solid #1f2a3c'><div style='min-width:0'><div style='font-weight:"
                    f"500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{html.escape(h['file'])}</div>"
                    f"<div style='font-size:.75rem;color:#5d6b82'>{html.escape(h['time'])} · {h['duration_s']} s · "
                    f"{h.get('confidence_pct', '–')}% confidence · {html.escape(str(h.get('trust', '–')))} "
                    f"held-out trust</div></div><span class='vg-num' style='font-size:.85rem;color:#8b98ad'>"
                    f"{h['fake_pct']:.1f}% syn.</span>{theme.chip(word, kind)}</div>")
    return "<div class='vg-panel' style='padding:6px 20px'>" + "".join(rows) + "</div>"


def overview() -> None:
    hero("VoiceGuard · audio forensics workstation",
         "Check whether a voice recording is real or synthetic",
         "One model, WavLM + CNN-BiLSTM-Att, scores your recording, points to the seconds that sound generated, "
         "and says how often calls like its own were right on held-out test data. Everything runs on this computer.",
         "shield-check")
    _toast()
    with st.container(horizontal=True, gap="small"):
        st.page_link(nav.get("detect"), label="Analyze a recording", icon=":material/graphic_eq:")
        st.page_link(nav.get("compare"), label="Compare two recordings", icon=":material/compare_arrows:")
        st.page_link(nav.get("results"), label="See the results", icon=":material/table_chart:")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    asv = _ea_set("ASVspoof eval")
    vh = asv["reliability"]["by_confidence"].get("very high (≥90% / ≤10%)") if asv else None
    row([stat(_pct(asv["overall"]["accuracy_pct"]) if asv else "81.4%", "Accuracy, unseen attacks",
              "71,237 clips · ASVspoof 2019 LA eval, E7a", ico="shield-check"),
         stat("10.7%", "EER on internet audio", "In-the-Wild, WavLM model alone (was 36–72%)", CYAN, "waves"),
         stat(_pct(vh["accuracy_pct"]) if vh else "99.1%", "Right at very high confidence",
              "held-out ASVspoof eval, P ≥ 90% or ≤ 10%", GREEN, "gauge"),
         stat(_pct(asv["overall"]["real_called_fake_pct"]) if asv else "11.2%", "Real recordings called SYNTHETIC",
              "ASVspoof 2019 LA eval, E7a", RED, "alert")])

    section("Analysis workflow", "Five steps from a file to an explained verdict.", "activity")
    cols = st.columns(5, gap="small")
    for i, (c, (ic, t, d)) in enumerate(zip(cols, WORKFLOW), 1):
        c.markdown(f"<div class='vg-card'><div style='display:flex;justify-content:space-between;align-items:center'>"
                   f"<div class='ib' style='margin:0'>{icon(ic, 17)}</div><span style='font-size:.72rem;"
                   f"color:#5d6b82;font-weight:600'>0{i}</span></div><h4 style='margin-top:12px'>{t}</h4>"
                   f"<p>{html.escape(d)}</p></div>", unsafe_allow_html=True)

    left, right = st.container(), st.container()
    with left:
        section("Workstation", "Every tool in the app.", "layers")
        items = [("detect", "AI model verdict", "REAL or SYNTHETIC from WavLM + CNN-BiLSTM-Att, with both "
                  "probabilities and held-out trust.", "cpu"),
                 ("detect", "Spectrogram lab", "Zoom, select, play regions and read time, frequency and level.",
                  "audio-waveform"),
                 ("detect", "Explainability center", "Why a file was flagged: model evidence beside model-free "
                  "measurements.", "search"),
                 ("compare", "A/B comparison", "A reference and a suspect on one time axis, with feature "
                  "charts.", "compare"),
                 ("batch", "Batch scoring", "Many files or a ZIP scored into one CSV.", "table"),
                 ("errors", "Error analysis", "Which attacks, speakers and clip lengths fail.", "chart")]
        for chunk in (items[:3], items[3:]):
            for c, (k, t, d, ic) in zip(st.columns(3, gap="small"), chunk):
                with c:
                    st.markdown(card(t, d, ic), unsafe_allow_html=True)
                    st.page_link(nav.get(k), label=f"Open {nav.get(k).title}", icon=":material/arrow_forward:")
    with right:
        hist = st.session_state.get("history", [])
        section("Recent analyses", "This browser session only.", "history", f"{len(hist)} total" if hist else "")
        if hist:
            st.markdown(_recent_html(hist), unsafe_allow_html=True)
            st.page_link(nav.get("history"), label="Open the full history", icon=":material/history:")
        else:
            theme.empty("No analyses yet", "Results you produce on the Analyze page appear here, newest first.",
                        "history")
            st.page_link(nav.get("detect"), label="Analyze a recording", icon=":material/graphic_eq:")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    theme.note("<b style='color:#fbbf24'>A screening aid, not proof.</b> Trained on English studio speech from 2019; "
               "modern voice conversion is mostly missed. Never use one verdict as the only evidence about a real "
               "person.", "alert")


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
    demos = det.demo_files()
    options = ["Upload audio or video", "Record with microphone"] + (["Demo clip"] if demos else [])
    src = st.segmented_control("Input", options, default=options[0], label_visibility="collapsed", key="src")
    with st.expander("Tips for a reliable result", icon=":material/lightbulb:"):
        st.markdown("- Use **5 s or more of speech**; under 2 s the verdict is weak.\n"
                    "- Prefer the **original file** over a re-recording or a heavily compressed copy.\n"
                    "- One speaker at a time, little music or background noise.\n"
                    "- Read the **trust panel** under the verdict and listen to every flagged segment.")
    got = None
    if src == "Upload audio or video":
        up = st.file_uploader(f"Drag and drop an audio or video file, up to {det.MAX_MB} MB (for a video, its audio "
                              "track is used)", type=sorted(set(det.AUDIO_TYPES + det.VIDEO_TYPES)), key=f"up_{n}")
        if up is not None:
            if up.size > det.MAX_MB * 2**20:
                st.error(f"The file is larger than {det.MAX_MB} MB. Cut or compress it first.", icon="🛑")
            else:
                got = (up.name, up.getvalue())
    elif src == "Record with microphone":
        rec = st.audio_input("Record a voice sample (a few seconds of speech works best)", key=f"mic_{n}")
        if rec is not None:
            got = ("microphone recording.wav", rec.getvalue())
    elif src == "Demo clip":
        demo = st.selectbox("DEEP-VOICE demonstration clip", ["(choose a clip)", *demos], key=f"demo_{n}")
        if demo != "(choose a clip)":
            got = (demo, demos[demo].read_bytes())
    if got is None:
        step_ph.markdown(theme.stepper_html(1), unsafe_allow_html=True)
        if src is None:
            theme.empty("Choose an input", "Pick upload, microphone or a demo clip above.", "upload")
        else:
            how = {"Upload audio or video": "Drag a file into the drop area above or browse for one.",
                   "Record with microphone": "Record a few seconds of speech with the recorder above.",
                   "Demo clip": "Pick one of the demonstration clips above."}.get(src, "")
            theme.empty("No recording loaded", f"{how} You can preview it before anything is analysed.",
                        "mic" if src == "Record with microphone" else "upload")
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
SET_COLS = {"eval": "ASVspoof eval", "itw": "In-the-Wild", "dv": "DEEP-VOICE (recording)",
            "deg_eval": "degraded eval"}


def _p3_frame() -> pd.DataFrame | None:
    p3 = data.phase3()
    if not p3:
        return None
    rows = []
    for k, v in p3.items():
        if k.startswith("_"):
            continue
        for col, name in SET_COLS.items():
            vals = v.get(col) or []
            if vals:
                rows.append({"model": v["label"], "test set": name, "EER %": sum(vals) / len(vals), "seeds": len(vals)})
    return pd.DataFrame(rows)


def results() -> None:
    import altair as alt

    hero("Research", "Results",
         "Equal error rate (EER, lower is better) of every model on the three held-out test sets. Models, "
         "thresholds and calibration were chosen on ASVspoof dev data only.", "chart")
    df = _p3_frame()
    t1, t2, t3, t4, t6, t5 = st.tabs(["Main comparison", "Retired ensemble", "Robustness", "Calibration",
                                      "Frequency bands", "Speed"])
    with t1:
        if df is not None and len(df):
            ch = alt.Chart(df).mark_bar().encode(
                y=alt.Y("model:N", title=None, sort=alt.EncodingSortField("EER %", op="min"),
                        axis=alt.Axis(labelLimit=340)),
                x=alt.X("EER %:Q", title="EER (%)"),
                color=alt.Color("test set:N", scale=alt.Scale(range=["#22d3ee", "#34d399", "#fbbf24", "#64748b"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                yOffset="test set:N",
                tooltip=["model", "test set", alt.Tooltip("EER %:Q", format=".2f"), "seeds"]).properties(height=520)
            st.altair_chart(ch, width="stretch")
        _md_file("experiments/tables/phase3.md")
        with st.expander("Phase 2 tables: features, shortcuts, augmentation (3 seeds each)"):
            _md_file("experiments/tables/phase2.md")
    with t2:
        theme.note("Research record only. The three-model ensemble (E7a + VoiceGuard improved + VoiceGuard v2) "
                   "was retired on 2026-11-23; the app and the API now run E7a alone.", "info")
        st.markdown("")
        _md_file("experiments/ensemble_binary/ensemble.md")
    with t3:
        st.markdown("EER on ASVspoof audio degraded by codecs, noise and the telephone band.")
        for f in ("experiments/tables/robustness_eval.png", "experiments/tables/robustness_dev.png"):
            if data.exists(f):
                st.image(str(data.path(f)), width="stretch")
    with t4:
        runs = sorted(p.stem for p in data.path("experiments/reliability").glob("*.png"))
        if runs:
            r = st.selectbox("Model", runs, index=runs.index("e7a_wavlm_crnn_s42") if "e7a_wavlm_crnn_s42" in runs
                             else 0)
            st.image(str(data.path(f"experiments/reliability/{r}.png")), width="stretch")
            _md_file(f"experiments/reliability/{r}.md")
    with t5:
        _md_file("experiments/efficiency/efficiency.md")
    with t6:
        if data.exists("experiments/band_analysis/band_analysis.png"):
            st.image(str(data.path("experiments/band_analysis/band_analysis.png")), width="stretch")
            _md_file("experiments/band_analysis/band_analysis.md")
        else:
            st.info("Run `python scripts/band_analysis.py` to add the per-attack frequency-band analysis.")


def errors() -> None:
    import altair as alt

    hero("Research", "Error analysis", "Where WavLM + CNN-BiLSTM-Att goes wrong on held-out data, and when to "
         "trust a verdict.", "search")
    ea = data.error_analysis()
    if not ea:
        st.info("Run `python scripts/error_analysis.py` to build this page (it reads the stored scores; about 1 min).")
        return
    st.markdown("Where the app's model, WavLM + CNN-BiLSTM-Att (E7a), goes wrong on held-out data at its "
                "REAL/SYNTHETIC line of 50%. Built from its stored per-clip scores by `scripts/error_analysis.py`.")
    names = list(ea["sets"])
    name = st.segmented_control("Test set", names, default=names[0], key="ea_set") or names[0]
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
        att = pd.DataFrame([{"attack": a, "fakes called REAL %": v["fake_called_real_pct"],
                             "EER %": v["merged_eer_pct"], "n": v["n"]} for a, v in d["by_attack"].items()])
        st.altair_chart(alt.Chart(att).mark_bar(color=RED).encode(
            x=alt.X("attack:N", title=None), y=alt.Y("fakes called REAL %:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=list(att.columns)).properties(height=240), width="stretch")
        st.dataframe(att, hide_index=True, width="stretch")
        st.caption("EER ranks the model's scores against all bona fide eval recordings; 'fakes called REAL' uses the "
                   "app's 50% line. A10 is hard for every model in this project.")
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


# ---------------------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------------------
def model_card() -> None:
    hero("About", "Model card", "Intended use, data, known failures and misuse.", "book")
    _md_file("docs/MODEL_CARD.md")


def help_page() -> None:
    hero("About", "Help", "Quick start, common questions and fixes.", "info")
    st.markdown("#### Quick start")
    for i, (t, d) in enumerate(content.STEPS, 1):
        st.markdown(f"**{i}. {t}.** {d}")
    st.markdown("#### Questions")
    for q, a in content.FAQ:
        with st.expander(q):
            st.markdown(a)
    st.markdown("#### Run it yourself")
    st.code("streamlit run app.py                     # this app, http://localhost:8501\n"
            "uvicorn api.main:app --port 8000         # REST API, docs at /docs\n"
            "python -m adfd.explain clip.wav          # verdict + figures on the command line\n"
            "python scripts/error_analysis.py         # rebuild the error analysis and trust table", language="bash")
    st.markdown("#### Something went wrong")
    st.markdown("- **No trained models found:** the `runs/` folders (`best.pt`, `reliability.json`) are not in git; "
                "copy them in or train (README).\n"
                "- **Could not read the file:** convert it to WAV or FLAC; for video, MP4 works best.\n"
                "- **Very slow:** the first analysis loads the model (~10 s); a hot laptop CPU can be 5× slower.\n"
                "- **Trust panel says to run a script:** run `python scripts/error_analysis.py` once.")
