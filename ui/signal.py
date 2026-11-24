"""Display-side signal measurements: waveform peaks, the spectrogram, the average spectrum and plain acoustic
features (RMS, spectral centroid, roll-off, zero-crossing rate, flatness, crest factor).

Everything here is measured from the decoded audio the app already holds; nothing feeds the detector.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf
import streamlit as st

N_FFT = 512                     # 32 ms at 16 kHz, 31.25 Hz bins
DB_RANGE = 80.0                 # spectrogram display range below the loudest bins


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _frames(x: np.ndarray, hop: int, n_fft: int = N_FFT) -> np.ndarray:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _power(x: np.ndarray, hop: int) -> np.ndarray:
    """Power spectrogram, shape (frames, N_FFT // 2 + 1)."""
    fr = _frames(x.astype(np.float32), hop) * np.hanning(N_FFT).astype(np.float32)
    return np.abs(np.fft.rfft(fr, axis=1)) ** 2


@st.cache_data(show_spinner=False, max_entries=16)
def peaks(x: np.ndarray, n: int = 6000) -> dict:
    """Min / max envelope in n bins (float32, base64) for drawing the waveform at any zoom."""
    n = int(max(1, min(n, len(x))))
    edges = np.linspace(0, len(x), n + 1).astype(int)
    lo = np.minimum.reduceat(x, edges[:-1]).astype(np.float32)
    hi = np.maximum.reduceat(x, edges[:-1]).astype(np.float32)
    return {"n": n, "min": _b64(lo), "max": _b64(hi), "peak": float(max(np.abs(x).max(), 1e-4))}


@st.cache_data(show_spinner=False, max_entries=16)
def spectrogram(x: np.ndarray, sr: int, max_cols: int = 1800) -> dict:
    """STFT magnitude in dB relative to the loudest bins, as uint8 rows (row 0 = highest frequency)."""
    hop = int(max(64, np.ceil(max(len(x) - N_FFT, 1) / max_cols)))
    p = _power(x, hop)
    db = 10 * np.log10(p + 1e-12)
    hi = float(np.percentile(db, 99.7))
    lo = hi - DB_RANGE
    q = np.clip((db - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)    # (frames, bins)
    img = np.ascontiguousarray(q.T[::-1])                                  # (bins, frames), top = high freq
    return {"rows": int(img.shape[0]), "cols": int(img.shape[1]), "data": _b64(img), "db_lo": -DB_RANGE,
            "db_hi": 0.0, "fmax": sr / 2, "hop_s": hop / sr, "win_s": N_FFT / sr}


@st.cache_data(show_spinner=False, max_entries=16)
def spectrogram_png(x: np.ndarray, sr: int, width: int = 900, height: int = 220) -> bytes:
    """A small spectrogram thumbnail (PNG) in the same colour map as the lab."""
    from PIL import Image

    sp = spectrogram(x, sr, max_cols=width)
    a = np.frombuffer(base64.b64decode(sp["data"]), dtype=np.uint8).reshape(sp["rows"], sp["cols"])
    rgb = HEAT_LUT[a]
    img = Image.fromarray(rgb, "RGB").resize((width, height), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=16)
def ltas(x: np.ndarray, sr: int) -> dict:
    """Long-term average spectrum: mean power per frequency bin (dB relative to its maximum)."""
    hop = int(max(256, len(x) // 4000))
    p = _power(x, hop).mean(axis=0)
    db = 10 * np.log10(p + 1e-12)
    db -= db.max()
    f = np.linspace(0, sr / 2, len(db))
    return {"freq_hz": f.tolist(), "db": db.tolist()}


@st.cache_data(show_spinner=False, max_entries=16)
def features(x: np.ndarray, sr: int) -> dict:
    """Plain acoustic features of the whole recording (frame means over frames with signal)."""
    hop = 256
    fr = _frames(x.astype(np.float32), hop)
    rms_f = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    active = rms_f > rms_f.max() * 10 ** (-40 / 20)                   # within 40 dB of the loudest frame
    if active.sum() < 3:
        active = np.ones_like(active)
    p = _power(x, hop)[: len(rms_f)]
    f = np.linspace(0, sr / 2, p.shape[1])
    tot = p.sum(axis=1) + 1e-12
    centroid = (p * f).sum(axis=1) / tot
    cum = np.cumsum(p, axis=1) / tot[:, None]
    rolloff = f[np.argmax(cum >= 0.85, axis=1)]
    flat = np.exp(np.log(p + 1e-12).mean(axis=1)) / (p.mean(axis=1) + 1e-12)
    zc = (np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)).sum(axis=1)) / fr.shape[1]
    out = {
        "rms_dbfs": float(20 * np.log10(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12)),
        "peak_dbfs": float(20 * np.log10(np.abs(x).max() + 1e-12)),
        "centroid_hz": float(centroid[active].mean()),
        "rolloff_hz": float(rolloff[active].mean()),
        "zcr": float(zc[active].mean()),
        "flatness": float(flat[active].mean()),
        "active_pct": float(100 * active.mean()),
    }
    out["crest_db"] = out["peak_dbfs"] - out["rms_dbfs"]
    return out


def audio_uri(x: np.ndarray, sr: int) -> str:
    """Audio for the in-page player: 16-bit WAV up to 2 minutes, Ogg Vorbis beyond (keeps the page light)."""
    buf = io.BytesIO()
    if len(x) / sr <= 120:
        sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
        mime = "audio/wav"
    else:
        sf.write(buf, x, sr, format="OGG", subtype="VORBIS")
        mime = "audio/ogg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


HEAT_STOPS = [[0.0, "#04070d"], [0.18, "#0b2540"], [0.36, "#0f5a73"], [0.52, "#128f9b"], [0.66, "#3cc6a8"],
              [0.8, "#b9e06a"], [0.91, "#f8d34a"], [1.0, "#fff6d5"]]


def _heat() -> np.ndarray:
    """Navy-teal-amber heat map: dark for quiet bins, bright for loud ones (monotonic in lightness)."""
    t = np.linspace(0, 1, 256)
    xs = [s[0] for s in HEAT_STOPS]
    rgb = [[int(c[1][i:i + 2], 16) for i in (1, 3, 5)] for c in HEAT_STOPS]
    return np.stack([np.interp(t, xs, [c[k] for c in rgb]) for k in range(3)], axis=1).astype(np.uint8)


HEAT_LUT = _heat()
