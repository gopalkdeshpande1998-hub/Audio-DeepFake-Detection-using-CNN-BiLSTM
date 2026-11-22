"""Human-readable explanation of a verdict: where in the recording, which frequencies, and why.

Three kinds of evidence, kept apart so a reader can weigh them:

1. Model evidence (what decided the verdict): the fused score over time, split into
   Real / Uncertain / Fake regions, and how many models agree.
2. Where the model looked: gradient x input saliency on the 128-band log-mel input of
   the VoiceGuard models, summarised per frequency band (below 1 kHz, 1-4 kHz, 4-8 kHz).
3. Acoustic observations (model-free): measurements of the recording compared with
   real reference speech (percentiles of bona fide ASVspoof train + dev recordings,
   `experiments/forensics/reference.json`, fitted by scripts/fit_forensics.py).
   Only measurements outside the central 90% of real speech are reported, with the
   usual forensic reading of such a deviation. They are observations, not proof: real
   recordings can be unusual too, and they never change the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adfd.paths import REPO_ROOT

SR = 16000
N_FFT, HOP = 512, 160
REFERENCE = REPO_ROOT / "experiments" / "forensics" / "reference.json"
BANDS = [("low (below 1 kHz: pitch and first formant)", 0, 1000),
         ("mid (1-4 kHz: formants, consonants)", 1000, 4000),
         ("high (4-8 kHz: fricatives, breath, vocoder artifacts)", 4000, 8000)]


# ---------------------------------------------------------------------------------------
# acoustic measurements
# ---------------------------------------------------------------------------------------
def _stft_power(x: np.ndarray) -> np.ndarray:
    import librosa

    return np.abs(librosa.stft(x.astype(np.float32), n_fft=N_FFT, hop_length=HOP)) ** 2 + 1e-12


def _pitch(x: np.ndarray, active: np.ndarray):
    import librosa

    f0 = librosa.yin(x.astype(np.float32), fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    n = min(len(f0), len(active))
    f0, active = f0[:n], active[:n]
    # yin always returns a value; keep active frames whose f0 is inside the search range
    voiced = active & (f0 > 72) & (f0 < 395)
    return f0[voiced]


def measure(x: np.ndarray) -> dict:
    """Model-free measurements of a 16 kHz mono recording (speech frames only)."""
    x = np.asarray(x, dtype=np.float64)
    S = _stft_power(x)
    freqs = np.linspace(0, SR / 2, S.shape[0])
    frame_db = 10 * np.log10(S.sum(axis=0))
    active = frame_db > frame_db.max() - 35
    pauses = (frame_db < frame_db.max() - 45)
    Sa = S[:, active] if active.sum() > 5 else S
    total = Sa.sum()
    hf = Sa[freqs >= 4000].sum() / total
    # bandwidth: highest frequency whose long-term level is within 50 dB of the peak
    lt = 10 * np.log10(Sa.mean(axis=1))
    above = np.where(lt > lt.max() - 50)[0]
    bandwidth = float(freqs[above[-1]]) if len(above) else 0.0
    # spectral flux between consecutive speech frames (smoothness of the spectrum over time)
    logS = np.log(Sa)
    flux = float(np.mean(np.sqrt(np.mean(np.diff(logS, axis=1) ** 2, axis=0)))) if Sa.shape[1] > 2 else float("nan")
    # high-band spectral flatness (noise-like vs. structured high frequencies)
    hb = Sa[freqs >= 4000]
    hf_flat = float(np.median(np.exp(np.mean(np.log(hb), axis=0)) / np.mean(hb, axis=0)))
    f0 = _pitch(x, active)
    f0_st = 12 * np.log2(f0 / np.median(f0)) if len(f0) > 10 else np.array([])
    # pause floor: level of the quietest stretches between words relative to speech
    if pauses.sum() >= 5 and active.any():
        floor = float(np.median(frame_db[pauses]) - np.median(frame_db[active]))
    else:
        floor = float("nan")
    return {
        "hf_energy_pct": 100 * float(hf),
        "bandwidth_khz": bandwidth / 1000,
        "spectral_flux": flux,
        "hf_flatness": hf_flat,
        "pitch_variation_st": float(np.std(f0_st)) if len(f0_st) else float("nan"),
        "pitch_jitter_st": float(np.median(np.abs(np.diff(f0_st)))) if len(f0_st) > 2 else float("nan"),
        "pause_floor_db": floor,
    }


# What a value outside the real-speech range usually means. (low reading, high reading)
CUES = {
    "hf_energy_pct": ("share of energy above 4 kHz", "%",
                      "unusually little high-frequency energy; band-limited vocoders and heavy compression remove it",
                      "unusually much high-frequency energy; some neural vocoders add hiss or metallic high-band noise"),
    "bandwidth_khz": ("effective bandwidth", "kHz",
                      "the spectrum stops early, as after resampling, telephone channels or a low-rate vocoder",
                      None),
    "spectral_flux": ("frame-to-frame spectral change", "",
                      "the spectrum changes unusually smoothly over time; over-smoothed spectra are typical of "
                      "acoustic models in text-to-speech",
                      "the spectrum changes unusually abruptly between frames; frame-level glitches can come from "
                      "splicing or unstable synthesis"),
    "hf_flatness": ("noisiness of the 4-8 kHz band", "",
                    "the high band is unusually structured (tonal), which can be a vocoder artifact",
                    "the high band is unusually noise-like"),
    "pitch_variation_st": ("pitch variation", "semitones",
                           "unusually flat, monotone intonation, common in synthetic or converted speech",
                           "unusually wide pitch swings"),
    "pitch_jitter_st": ("frame-to-frame pitch jitter", "semitones",
                        "pitch is unusually steady from frame to frame; natural voices have small irregularities",
                        "pitch is unusually unstable from frame to frame; voice conversion can produce pitch glitches"),
    "pause_floor_db": ("noise floor in pauses, relative to speech", "dB",
                       "pauses are unusually silent (near digital silence), as in generated audio with no room "
                       "noise",
                       "pauses are unusually noisy relative to the speech"),
}


# Measured and shown in the table, but never turned into a statement: silent pauses are
# also produced by MP3 encoders, noise suppression and editing, and leading/trailing
# silence is a known ASVspoof shortcut, so this cue would mislead on real recordings.
TABLE_ONLY = {"pause_floor_db"}


def load_reference(path: Path = REFERENCE) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def observations(meas: dict, ref: dict | None) -> list[dict]:
    """Measurements outside the central 90% of real reference speech, most unusual first."""
    if not ref:
        return []
    out = []
    for k, (name, unit, low_txt, high_txt) in CUES.items():
        if k in TABLE_ONLY:
            continue
        v = meas.get(k)
        r = ref["percentiles"].get(k)
        if v is None or r is None or not np.isfinite(v):
            continue
        grid, vals = np.array(ref["grid"]), np.array(r)
        pct = float(np.interp(v, vals, grid))          # percentile of v among real speech
        side = "low" if pct <= 5 else "high" if pct >= 95 else None
        if side is None:
            continue
        txt = low_txt if side == "low" else high_txt
        if txt is None:
            continue
        rare = pct if side == "low" else 100 - pct
        out.append({"cue": k, "name": name, "value": v, "unit": unit, "percentile": pct, "side": side,
                    "statement": f"{name.capitalize()} is {v:.2f}{(' ' + unit) if unit else ''}: "
                                 f"{'lower' if side == 'low' else 'higher'} than "
                                 f"{'over 99' if rare < 1 else f'{100 - rare:.0f}'}% of real "
                                 f"reference recordings. Possible reading: {txt}.",
                    "rarity": rare})
    return sorted(out, key=lambda o: o["rarity"])


# ---------------------------------------------------------------------------------------
# where the model looked
# ---------------------------------------------------------------------------------------
def mel_band_edges(n_mels: int = 128, fmax: float = SR / 2) -> np.ndarray:
    import librosa

    return librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fmax, htk=False)


def band_shares(sal_map: np.ndarray) -> list[dict]:
    """Share of |saliency| per frequency band, for a (128 mel, time) map."""
    f = mel_band_edges(sal_map.shape[0])
    per_bin = np.abs(sal_map).sum(axis=1)
    tot = per_bin.sum() + 1e-12
    width = np.r_[np.diff(f), np.diff(f)[-1]]
    out = []
    for name, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out.append({"band": name, "share_pct": 100 * float(per_bin[m].sum() / tot),
                    "bandwidth_share_pct": 100 * float(m.sum() / len(f)), "hz": (lo, hi), "_w": width[m].sum()})
    for o in out:
        o.pop("_w")
    return out


# ---------------------------------------------------------------------------------------
# segments and statements
# ---------------------------------------------------------------------------------------
def regions(times: np.ndarray, p: np.ndarray, t_real: float, t_fake: float, step: float) -> list[dict]:
    """Merge consecutive per-second scores into Real / Uncertain / Fake regions."""
    lab = np.where(p >= t_fake, "Fake", np.where(p <= t_real, "Real", "Uncertain"))
    out = []
    for i, (t, l) in enumerate(zip(times, lab)):
        if out and out[-1]["label"] == l:
            out[-1]["end_s"] = float(t + step)
            out[-1]["p"].append(float(p[i]))
        else:
            out.append({"label": str(l), "start_s": float(t), "end_s": float(t + step), "p": [float(p[i])]})
    for r in out:
        r["p_fake_mean"] = float(np.mean(r.pop("p")))
    return out


def statements(res: dict) -> dict:
    """Plain-language explanation of an ensemble result (adfd.ensemble.EnsembleAnalyzer.analyze)."""
    v, p = res["verdict"], res["p_fake"]
    members = res["members"]
    n_fake = sum(m["vote"] == "Fake" for m in members)
    n_real = sum(m["vote"] == "Real" for m in members)
    lines = []
    if res["out_of_distribution"]:
        lines.append("The input does not look like normal speech (" + "; ".join(res["ood_reasons"]) +
                     "), so no verdict is given.")
    head = {"Fake": f"Verdict: likely FAKE (synthetic or converted speech), P(fake) = {p:.2f}.",
            "Real": f"Verdict: likely REAL (bona fide speech), P(fake) = {p:.2f}.",
            "Uncertain": f"Verdict: UNCERTAIN, P(fake) = {p:.2f}. The evidence is not strong or consistent "
                         "enough to call it either way."}[v]
    lines.append(head)
    votes = ", ".join(f"{m['label']}: {m['p_fake']:.2f}" for m in members)
    lines.append(f"{n_fake} of {len(members)} models lean fake, {n_real} lean real ({votes}).")
    if res.get("gate_reason"):
        lines.append(res["gate_reason"])
    regs = res.get("regions", [])
    fake_r = [r for r in regs if r["label"] == "Fake"]
    real_r = [r for r in regs if r["label"] == "Real"]
    dur = res["duration_s"]
    if fake_r:
        tot = sum(r["end_s"] - r["start_s"] for r in fake_r)
        spans = ", ".join(f"{r['start_s']:.0f}–{r['end_s']:.0f} s" for r in fake_r[:4])
        lines.append(f"Where: {tot:.0f} s of {dur:.0f} s score as fake ({spans}).")
        if real_r and v != "Real":
            lines.append("Other parts score as real, which fits partly manipulated (spliced) audio, or a model "
                         "that is unsure about this recording.")
    elif real_r and v == "Real":
        lines.append("Where: no part of the recording scores above the fake threshold.")
    bands = res.get("bands")
    if bands:
        top = max(bands, key=lambda b: b["share_pct"] / max(b["bandwidth_share_pct"], 1e-6))
        lines.append(f"Which frequencies: for the most suspicious window, the mel-spectrogram model relied most on the "
                     f"{top['band']} band ({top['share_pct']:.0f}% of the evidence on "
                     f"{top['bandwidth_share_pct']:.0f}% of the mel bands).")
    obs = res.get("observations", [])
    if obs:
        lines.append("Acoustic observations (model-free, compared with real reference speech; these support, but "
                     "never decide, the verdict):")
        lines += ["• " + o["statement"] for o in obs[:4]]
    else:
        lines.append("Acoustic observations: every measured property is inside the normal range of real speech.")
    return {"summary": lines[0] if not res["out_of_distribution"] else lines[0], "lines": lines}
