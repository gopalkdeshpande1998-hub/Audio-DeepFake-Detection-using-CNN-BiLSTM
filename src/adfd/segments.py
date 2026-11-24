"""Where and why: evidence for each model-flagged segment of a recording.

For every segment the merged detector flags (local merged P(fake) >= 50%), two kinds of
evidence are kept apart, so a reader can weigh them:

Model evidence: the merged and per-model scores inside the segment (what flagged it).
Acoustic measurements (model-free; they never change the verdict):
  - contrast with the rest of the recording: a 10 ms frame feature (loudness, energy above
    4 kHz, spectral centroid, frame-to-frame spectral change, high-band noisiness, pitch)
    whose median inside the segment is at least 1 robust SD away from the rest;
  - boundary checks: spectral change or a loudness jump right at the segment edges that is
    exceptional for this recording;
  - comparison with real reference speech (segments >= 1 s): measurements outside the
    central 90% of bona fide speech (adfd.forensics.observations).
Only measured differences are reported, each with its possible reading, and a flagged
segment is always described as suspected, never as proven manipulation.
"""

from __future__ import annotations

import numpy as np

from adfd.forensics import HOP, SR, _stft_power, measure, observations

FRAME_S = HOP / SR                       # 10 ms frames
SEVERITY = [(0.85, "High"), (0.70, "Medium"), (0.0, "Low")]   # by the segment's mean merged P(fake)
SEVERITY_TEXT = "High: mean fake probability ≥ 85%, Medium: 70–85%, Low: 50–70%"

# frame feature -> (name, unit, reading when the segment is higher, reading when lower)
FRAME_CUES = {
    "level_db": ("loudness", "dB",
                 "a level change that the surrounding speech does not share, as when material from another "
                 "source is inserted (emphasis can also do this)",
                 "a level change that the surrounding speech does not share, as when material from another "
                 "source is inserted (quiet speech can also do this)"),
    "hf_pct": ("share of energy above 4 kHz", "%",
               "more high-frequency energy than the rest; some neural vocoders add hiss or metallic high-band noise",
               "less high-frequency energy than the rest; band-limited vocoders and re-encoding remove it"),
    "centroid_hz": ("spectral centroid (brightness)", "Hz",
                    "a brighter spectrum than the rest of the recording, i.e. a shift of the frequency distribution",
                    "a duller spectrum than the rest of the recording, i.e. a shift of the frequency distribution"),
    "flux": ("frame-to-frame spectral change", "",
             "the spectrum changes more abruptly between frames than in the rest; frame-level glitches can come "
             "from splicing or unstable synthesis",
             "the spectrum changes more smoothly than in the rest; over-smoothed spectra are typical of "
             "text-to-speech acoustic models"),
    "hf_flatness": ("noisiness of the 4-8 kHz band", "",
                    "a more noise-like high band than the rest of the recording",
                    "a more tonal (structured) high band than the rest; this can be a vocoder artifact"),
    "f0_st": ("pitch (relative to the recording's median)", "semitones",
              "a pitch shift relative to the rest; voice conversion can move or destabilise pitch",
              "a pitch shift relative to the rest; voice conversion can move or destabilise pitch"),
}


def fmt_ts(t: float) -> str:
    """Seconds -> mm:ss.s (e.g. 00:02.4)."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def severity(p: float) -> str:
    return next(name for lo, name in SEVERITY if p >= lo)


def frame_features(x: np.ndarray) -> dict:
    """10 ms frame features of a 16 kHz recording."""
    import librosa

    x = np.asarray(x, dtype=np.float64)
    S = _stft_power(x)
    freqs = np.linspace(0, SR / 2, S.shape[0])
    tot = S.sum(axis=0)
    level = 10 * np.log10(tot)
    hb = S[freqs >= 4000]
    flux = np.r_[0.0, np.sqrt(np.mean(np.diff(np.log(S), axis=1) ** 2, axis=0))]
    f0 = librosa.yin(x.astype(np.float32), fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    n = min(S.shape[1], len(f0))
    active = level[:n] > level.max() - 35
    voiced = active & (f0[:n] > 72) & (f0[:n] < 395)
    f0_st = np.full(n, np.nan)
    if voiced.sum() > 10:
        f0_st[voiced] = 12 * np.log2(f0[:n][voiced] / np.median(f0[:n][voiced]))
    return {"t": np.arange(n) * FRAME_S, "active": active, "level_db": level[:n],
            "hf_pct": 100 * (hb.sum(axis=0) / tot)[:n], "centroid_hz": ((freqs[:, None] * S).sum(axis=0) / tot)[:n],
            "flux": flux[:n], "hf_flatness": (np.exp(np.mean(np.log(hb), axis=0)) / np.mean(hb, axis=0))[:n],
            "f0_st": f0_st}


def robust_shift(a: np.ndarray, b: np.ndarray) -> float:
    """(median a - median b) in units of b's robust spread (1.4826 * MAD)."""
    mad = 1.4826 * np.median(np.abs(b - np.median(b))) + 1e-9
    return float((np.median(a) - np.median(b)) / mad)


def segment_evidence(x: np.ndarray, seg: dict, ff: dict, res: dict, ref: dict | None,
                     min_shift: float = 1.0) -> dict:
    """{'model': [str], 'acoustic': [{kind, cue, statement}], 'notes': [str]} for one segment."""
    a, b = seg["start_s"], seg["end_s"]
    dur = res["duration_s"]
    model, acoustic, notes = [], [], []
    # --- model evidence -----------------------------------------------------------------
    idx = [i for i, w in enumerate(res["per_second"])
           if w["start_s"] < b and w["end_s"] > a and w["p_fake"] is not None]
    if idx:
        peak = max(idx, key=lambda i: res["per_second"][i]["p_fake"])
        pk = res["per_second"][peak]
        model.append(f"Synthetic probability averages {100 * seg['p_fake']:.0f}% in this segment (peak "
                     f"{100 * pk['p_fake']:.0f}% at {fmt_ts(pk['start_s'])}), above the 50% decision line.")
        above = []
        for m in res.get("members", []):
            vals = [m["per_second"][i] for i in idx if m["per_second"][i] is not None]
            if not vals:
                continue
            q = float(np.mean(vals))
            above.append(q >= 0.5)
            model.append(f"{m['label']}: {100 * q:.0f}% fake on average here.")
        if above:
            model.append(f"{sum(above)} of {len(above)} models score this segment above 50% on their own.")
    # --- acoustic: contrast with the rest of the recording -------------------------------
    t, act = ff["t"], ff["active"]
    inside = (t >= a) & (t < b)
    n_in, n_out = int((inside & act).sum()), int((~inside & act).sum())
    if n_in >= 30 and n_out >= 30:
        found = False
        for k, (name, unit, hi_txt, lo_txt) in FRAME_CUES.items():
            v = ff[k]
            ok = np.isfinite(v) & act
            seg_v, rest_v = v[inside & ok], v[~inside & ok]
            if len(seg_v) < 20 or len(rest_v) < 20:
                continue
            d = robust_shift(seg_v, rest_v)
            if abs(d) < min_shift:
                continue
            found = True
            u = f" {unit}" if unit else ""
            acoustic.append({"kind": "contrast", "cue": k, "shift_sd": d, "statement":
                             f"{name[0].upper() + name[1:]} is {'higher' if d > 0 else 'lower'} here: median "
                             f"{np.median(seg_v):.2f}{u} vs {np.median(rest_v):.2f}{u} in the rest of the recording "
                             f"({abs(d):.1f} robust SD). Possible reading: {hi_txt if d > 0 else lo_txt}."})
        if not found:
            notes.append("No measured acoustic property differs clearly (≥ 1 robust SD) between this segment and the "
                         "rest of the recording; the flag rests on the models' learned features.")
    elif n_out < 30:
        notes.append("The segment covers (almost) the whole recording, so there is no unflagged speech to compare it "
                     "with; only the comparison with real reference speech is available.")
    else:
        notes.append("Too little speech inside the segment for a reliable acoustic contrast.")
    # --- acoustic: boundaries --------------------------------------------------------------
    flux_act = ff["flux"][act]
    for edge, which in ((a, "start"), (b, "end")):
        if edge < 0.3 or edge > dur - 0.3 or len(flux_act) < 50:
            continue
        near = act & (np.abs(t - edge) <= 0.15)
        if near.any():
            pk = float(ff["flux"][near].max())
            pct = 100 * float((flux_act < pk).mean())
            if pct >= 97:
                acoustic.append({"kind": "boundary", "cue": "flux", "statement":
                                 f"Abrupt spectral transition at the segment {which} ({fmt_ts(edge)}): the "
                                 f"frame-to-frame spectral change there exceeds {pct:.0f}% of this recording's speech "
                                 "frames. Splice points can look like this; so can ordinary word onsets."})
        before = (t >= edge - 0.3) & (t < edge) & act
        after = (t >= edge) & (t < edge + 0.3) & act
        if before.sum() >= 10 and after.sum() >= 10:
            jump = float(np.median(ff["level_db"][after]) - np.median(ff["level_db"][before]))
            if abs(jump) >= 8:
                acoustic.append({"kind": "boundary", "cue": "level_db", "statement":
                                 f"Loudness {'jumps up' if jump > 0 else 'drops'} by {abs(jump):.0f} dB at the segment "
                                 f"{which} ({fmt_ts(edge)}). Editing can cause abrupt level changes; pauses and "
                                 "emphasis can too."})
    # --- acoustic: against real reference speech -----------------------------------------
    if ref and b - a >= 1.0:
        for o in observations(measure(x[int(a * SR):int(b * SR)]), ref)[:3]:
            acoustic.append({"kind": "reference", "cue": o["cue"],
                             "statement": "Compared with real reference speech: " + o["statement"]})
    return {"model": model, "acoustic": acoustic, "notes": notes}
