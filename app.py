"""Audio deepfake detector: Streamlit app (v2, PyTorch).

    streamlit run app.py

Default model: the merged detector (adfd.ensemble: WavLM E7a + VoiceGuard improved +
VoiceGuard v2). Per file: a cautious Real / Uncertain / Fake verdict (Fake or Real only
when the models agree), a warning when the input does not look like speech, a forensic
view (waveform and mel spectrogram coloured by Real / Uncertain / Fake region, with the
saliency of the decisive window), plain-language reasons, each model's vote, acoustic
measurements against real reference speech, and an HTML report. Input: upload (audio
or video), microphone recording, or a demo clip. Batch mode scores many files (or a
ZIP) into a CSV. Models: see adfd.registry; inference runs on the CPU.
"""

from __future__ import annotations

import base64
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
from adfd.explain import plot_forensic, plot_heatmap, plot_timeline  # noqa: E402
from adfd.inference import SR, load_audio  # noqa: E402

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a", "aiff", "aif", "webm", "opus"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "avi", "webm"]
MAX_MB = 50
VERDICT_STYLE = {"Real": ("✅", "#1baf7a"), "Fake": ("⚠️", "#d1495b"), "Uncertain": ("❔", "#eda100")}
LIMITS = ("This is a research prototype. The merged detector says Fake or Real only when its models agree and "
          "answers Uncertain otherwise, which keeps false detections rare but cannot rule them out: on the "
          "held-out tests some real recordings are still called fake (see experiments/ensemble/ensemble.md). "
          "Never use a single verdict as proof that a recording is real or fake.")

st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
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


def verdict_card(res: dict) -> None:
    icon, color = VERDICT_STYLE[res["verdict"]]
    pct = 100 * res["p_fake"]
    st.markdown(
        f"<div style='border-left:6px solid {color};padding:0.6rem 1rem;border-radius:6px;"
        f"background:rgba(127,127,127,0.08)'><div style='font-size:1.6rem;font-weight:600'>{icon} {res['verdict']}</div>"
        f"<div>Probability of fake: <b>{pct:.1f}%</b>"
        f"{'' if res['calibrated'] else ' (uncalibrated model)'}</div></div>", unsafe_allow_html=True)
    if res["thresholds"]:
        t = res["thresholds"]
        st.caption(f"Real below {100 * t['t_real']:.0f}%, Fake above {100 * t['t_fake']:.0f}%, Uncertain in between "
                   "(thresholds fitted on ASVspoof dev data).")


def ood_warning(res: dict) -> None:
    if res["out_of_distribution"]:
        st.warning("**This doesn't look like normal speech**, so the verdict is not meaningful: "
                   + "; ".join(res["ood_reasons"]) + ".", icon="🚫")


def report_html(name: str, res: dict, timeline_png: bytes, heatmap_png: bytes | None, others: pd.DataFrame | None,
                model_label: str) -> str:
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    icon, color = VERDICT_STYLE[res["verdict"]]
    segs = "".join(f"<li>{w['start_s']:.1f}–{w['end_s']:.1f} s: P(fake) {w['p_fake']:.2f}</li>"
                   for w in res["suspicious_segments"]) or "<li>none above the Fake threshold</li>"
    ood = ("<p style='color:#b35c00'><b>Out-of-distribution warning:</b> " + "; ".join(res["ood_reasons"]) + "</p>"
           if res["out_of_distribution"] else "<p>Input looks like speech (signal and embedding checks passed).</p>")
    table = others.to_html(index=False, float_format=lambda v: f"{v:.3f}") if others is not None else ""
    heat = f"<h2>Spectrogram and saliency</h2><img src='data:image/png;base64,{b64(heatmap_png)}'>" if heatmap_png else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Deepfake report: {name}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f1f1e}}
img{{max-width:100%}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.v{{border-left:6px solid {color};padding:.6rem 1rem;background:#f4f4f2}}</style></head><body>
<h1>Audio deepfake report</h1>
<p>File: <b>{name}</b> ({res['duration_s']:.1f} s) · {datetime.now():%Y-%m-%d %H:%M} · model: {model_label} (<code>{res['model']}</code>)</p>
<div class='v'><h2 style='margin:0'>{icon} {res['verdict']}</h2>Probability of fake: <b>{100 * res['p_fake']:.1f}%</b></div>
{ood}<h2>Score over time</h2><img src='data:image/png;base64,{b64(timeline_png)}'>
<h3>Most suspicious segments</h3><ul>{segs}</ul>{heat}
{'<h2>All models</h2>' + table if table else ''}
<h2>Limitations</h2><p>{LIMITS}</p></body></html>"""


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
        if pct is None:
            flag = "n/a"
        elif pct <= 5:
            flag = "unusual (low)"
        elif pct >= 95:
            flag = "unusual (high)"
        else:
            flag = "normal"
        rows.append({"measurement": name, "value": round(float(v), 3) if ok else None, "unit": unit,
                     "percentile among real speech": None if pct is None else round(pct), "reading": flag})
    return pd.DataFrame(rows)


def ensemble_view(name: str, x: np.ndarray, res: dict) -> tuple[bytes, pd.DataFrame]:
    left, right = st.columns([1, 2])
    with left:
        verdict_card(res)
        if res.get("gate_reason"):
            st.info(res["gate_reason"], icon="🛡️")
        ood_warning(res)
        st.audio(to_wav_bytes(x), format="audio/wav")
        st.caption(f"{res['duration_s']:.1f} s · scored by {len(res['members'])} models")
        members = pd.DataFrame([{"model": m["label"], "vote": m["vote"], "P(fake)": round(m["p_fake"], 3)}
                                for m in res["members"]])
    with right:
        st.markdown("**Why: the evidence behind the verdict**")
        body = []
        for ln in res["explanation"]["lines"][1:]:
            body.append("    - " + ln[2:] if ln.startswith("• ") else "- " + ln)
        st.markdown("\n".join(body))
        st.markdown("**Model votes**")
        st.dataframe(members, hide_index=True, width="stretch")
    with st.spinner("Drawing the forensic view…"):
        fpng = fig_png(plot_forensic(x, res))
    st.image(fpng, width="stretch")
    st.caption("Top: the waveform, coloured by region. Middle: the 128-band log-mel spectrogram the VoiceGuard "
               "models read; the bar on top repeats the regions, and the cyan outline marks the time-frequency "
               "areas that moved the score most in the analysed window (gradient × input). Bottom: the merged "
               "P(fake) per second, with each model's own score. Saliency shows where a model looked, not proof "
               "of manipulation.")
    fake_regions = sorted((r for r in res["regions"] if r["label"] == "Fake"), key=lambda r: -r["p_fake_mean"])[:3]
    if fake_regions:
        st.markdown("**Regions scored as fake** (listen to them)")
        for c, r in zip(st.columns(len(fake_regions)), fake_regions):
            a, b = int(r["start_s"] * SR), int(r["end_s"] * SR)
            with c:
                st.write(f"{r['start_s']:.0f}–{r['end_s']:.0f} s · P(fake) {r['p_fake_mean']:.2f}")
                st.audio(to_wav_bytes(x[a:b]), format="audio/wav")
    with st.expander("Acoustic measurements compared with real speech"):
        st.dataframe(measurement_table(res), hide_index=True, width="stretch")
        st.caption("Percentile among 1,700 real reference recordings (ASVspoof train/dev bona fide, clean and "
                   "with codec, noise and telephone degradations). Values below the 5th or above the 95th "
                   "percentile are reported as observations; they never change the verdict.")
    if res.get("bands"):
        with st.expander("Which frequency bands the mel-spectrogram model relied on"):
            st.dataframe(pd.DataFrame([{"band": b["band"], "share of evidence %": round(b["share_pct"], 1),
                                        "share of mel bands %": round(b["bandwidth_share_pct"], 1)}
                                       for b in res["bands"]]), hide_index=True, width="stretch")
    return fpng, members


def ensemble_report(name: str, res: dict, fpng: bytes, members: pd.DataFrame) -> str:
    b64 = base64.b64encode(fpng).decode()
    icon, color = VERDICT_STYLE[res["verdict"]]
    lines = "".join(f"<li>{ln.lstrip('• ')}</li>" for ln in res["explanation"]["lines"][1:])
    regs = "".join(f"<tr><td>{r['label']}</td><td>{r['start_s']:.0f}–{r['end_s']:.0f} s</td>"
                   f"<td>{r['p_fake_mean']:.2f}</td></tr>" for r in res["regions"])
    gate = f"<br>{res['gate_reason']}" if res.get("gate_reason") else ""
    meas = measurement_table(res).to_html(index=False)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Deepfake report: {name}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1f1f1e}}
img{{max-width:100%}} table{{border-collapse:collapse;margin:.5rem 0}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.v{{border-left:6px solid {color};padding:.6rem 1rem;background:#f4f4f2}}</style></head><body>
<h1>Audio deepfake report</h1>
<p>File: <b>{name}</b> ({res['duration_s']:.1f} s) · {datetime.now():%Y-%m-%d %H:%M} · model: {registry.ENSEMBLE_LABEL}</p>
<div class='v'><h2 style='margin:0'>{icon} {res['verdict']}</h2>Merged probability of fake: <b>{100 * res['p_fake']:.1f}%</b>{gate}</div>
<h2>Why</h2><ul>{lines}</ul>
<h2>Forensic view</h2><img src='data:image/png;base64,{b64}'>
<h2>Regions</h2><table><tr><th>verdict</th><th>time</th><th>mean P(fake)</th></tr>{regs}</table>
<h2>Model votes</h2>{members.to_html(index=False)}
<h2>Acoustic measurements</h2>{meas}
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
    with st.spinner("Analysing with all models…"):
        res = get_analyzer(run).analyze(x)
    fpng, members = ensemble_view(name, x, res)
    st.download_button("Download report (HTML)", ensemble_report(name, res, fpng, members),
                       file_name=f"{Path(name).stem}_deepfake_report.html", mime="text/html")


def analyse_single(name: str, x: np.ndarray, run: str, compare: bool) -> None:
    an = get_analyzer(run)
    with st.spinner("Analysing…"):
        res = an.analyze(x)
    left, right = st.columns([1, 2])
    with left:
        verdict_card(res)
        ood_warning(res)
        st.audio(to_wav_bytes(x), format="audio/wav")
        st.caption(f"{res['duration_s']:.1f} s · analysed {res['analysed_region_s'][0]:.1f}–"
                   f"{res['analysed_region_s'][1]:.1f} s · {len(res['timeline'])} window(s) of 4 s")
    with right:
        tl_png = fig_png(plot_timeline(res))
        st.image(tl_png, width="stretch")
        if res["suspicious_segments"]:
            st.markdown("**Most suspicious segments**")
            for w in res["suspicious_segments"]:
                a, b = int(w["start_s"] * SR), int(w["end_s"] * SR)
                st.write(f"{w['start_s']:.1f}–{w['end_s']:.1f} s · P(fake) {w['p_fake']:.2f}")
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
            rows.append({"model": label, "verdict": o["verdict"], "P(fake)": o["p_fake"],
                         "out of distribution": o["out_of_distribution"]})
        others = pd.DataFrame(rows)
        st.markdown("**All models**")
        st.dataframe(others, hide_index=True, width="stretch")
        decided = others[others["verdict"] != "Uncertain"]["verdict"]
        if len(decided):
            agree = decided.value_counts().iloc[0]
            st.caption(f"{agree} of {len(decided)} decided models agree on **{decided.value_counts().index[0]}**.")
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
            rows.append({"file": name, "verdict": res["verdict"], "p_fake": round(res["p_fake"], 4),
                         "out_of_distribution": res["out_of_distribution"], "duration_s": round(res["duration_s"], 2),
                         "ood_reasons": "; ".join(res["ood_reasons"])})
        except Exception as e:  # noqa: BLE001
            rows.append({"file": name, "verdict": "error", "p_fake": None, "out_of_distribution": None,
                         "duration_s": None, "ood_reasons": f"could not read: {e.__class__.__name__}"})
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
        compare = st.toggle("Compare all models", value=False, help="Score the file with every available model.")
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
