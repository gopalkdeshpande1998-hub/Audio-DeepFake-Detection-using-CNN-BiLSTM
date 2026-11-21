"""Training-time waveform augmentation (P2.3). Operates on float32 numpy crops in [-1, 1].

  RawBoost  (Tak et al., ICASSP 2022): linear/non-linear convolutive noise (LnL),
            impulsive signal-dependent noise (ISD), stationary signal-independent
            noise (SSI). Needs no extra data. Follows the reference implementation.
  Codec     MP3 (about 19-51 kbps), Opus (about 10-135 kbps) or a telephone channel
            (300-3400 Hz band-pass, 8 kHz, mu-law), encoded and decoded in memory.
  Resample  down to a random lower rate (8, 11.025 or 12 kHz) and back: removes the high band.
  Speed     speed perturbation by a factor of 0.9-1.1 (changes tempo and pitch, as in Kaldi).
  Gain      random gain in dB, clipped to [-1, 1].

Used only on the ASVspoof train split. Test sets are never augmented.
Configure with an `augment:` block, e.g.

    augment:
      rawboost: {p: 0.7, algos: [lnl, isd]}
      codec:    {p: 0.5, types: [mp3, opus, telephone]}
      resample: {p: 0.3, rates: [8000, 11025, 12000]}
      speed:    {p: 0.3, min: 0.9, max: 1.1}
      gain:     {p: 0.5, min_db: -6, max_db: 6}

Output length changes (speed, codec delay) are fixed by repeating the clip, never by
zero-padding, so augmentation does not add silence (a known shortcut, see P2.2).
"""

from __future__ import annotations

import io
import random

import numpy as np
import soundfile as sf
import soxr
from scipy import signal

# ----------------------------------------------------------------------------
# RawBoost
# ----------------------------------------------------------------------------

def _rand(lo, hi, integer=False):
    return random.randint(int(lo), int(hi)) if integer else random.uniform(lo, hi)


def _norm(x, always: bool):
    peak = np.max(np.abs(x)) + 1e-12
    return x / peak if (always or peak > 1) else x


def _notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs):
    b = np.array([1.0])
    for _ in range(n_bands):
        fc = _rand(min_f, max_f)
        bw = _rand(min_bw, max_bw)
        c = _rand(min_coeff, max_coeff, integer=True)
        if c % 2 == 0:
            c += 1
        f1, f2 = max(fc - bw / 2, 1e-3), min(fc + bw / 2, fs / 2 - 1e-3)
        b = np.convolve(signal.firwin(c, [f1, f2], window="hamming", fs=fs), b)
    gain = _rand(min_g, max_g)
    _, h = signal.freqz(b, 1, fs=fs)
    return 10 ** (gain / 20) * b / np.max(np.abs(h))


def _fir(x, b):
    n = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, n)))
    return y[n // 2: y.shape[0] - n // 2][: len(x)]


def rawboost_lnl(x, fs=16000, n_f=5, n_bands=5, min_f=20, max_f=8000, min_bw=100, max_bw=1000,
                 min_coeff=10, max_coeff=100, min_g=0, max_g=0, min_bias=5, max_bias=20):
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(n_f):
        if i == 1:
            min_g, max_g = min_g - min_bias, max_g - max_bias
        b = _notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs)
        y = y + _fir(np.power(x, i + 1), b)
    y = y - np.mean(y)
    return _norm(y, always=False)


def rawboost_isd(x, p=10, g_sd=2):
    y = x.astype(np.float64).copy()
    n = int(len(x) * _rand(0, p) / 100)
    idx = np.random.permutation(len(x))[:n]
    f_r = (2 * np.random.rand(n) - 1) * (2 * np.random.rand(n) - 1)
    y[idx] = x[idx] + g_sd * x[idx] * f_r
    return _norm(y, always=False)


def rawboost_ssi(x, fs=16000, snr_min=10, snr_max=40, n_bands=5, min_f=20, max_f=8000, min_bw=100, max_bw=1000,
                 min_coeff=10, max_coeff=100, min_g=0, max_g=0):
    noise = np.random.normal(0, 1, len(x))
    b = _notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs)
    noise = _norm(_fir(noise, b), always=True)
    snr = _rand(snr_min, snr_max)
    noise = noise / (np.linalg.norm(noise) + 1e-12) * np.linalg.norm(x) / 10 ** (0.05 * snr)
    return x + noise


def _isd(x, fs=16000):
    return rawboost_isd(x)


RAWBOOST = {"lnl": rawboost_lnl, "isd": _isd, "ssi": rawboost_ssi}


# ----------------------------------------------------------------------------
# Codecs and channels (also used to build the degraded test sets in P2.4)
# ----------------------------------------------------------------------------

def _roundtrip(x, sr, fmt, subtype, level):
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), sr, format=fmt, subtype=subtype, compression_level=level)
    buf.seek(0)
    y, _ = sf.read(buf, dtype="float32")
    return y


def mp3(x, sr=16000, level=0.5):
    """MP3 round trip. compression_level 0 -> about 51 kbps, 0.99 -> about 19 kbps (16 kHz mono)."""
    return _roundtrip(x, sr, "MP3", "MPEG_LAYER_III", level)


def opus(x, sr=16000, level=0.7):
    """Opus round trip. compression_level 0.5 -> about 135 kbps, 0.99 -> about 10 kbps."""
    return _roundtrip(x, sr, "OGG", "OPUS", level)


def telephone(x, sr=16000):
    """300-3400 Hz band-pass, resample to 8 kHz, mu-law, back to sr."""
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x).astype(np.float32)
    y8 = soxr.resample(y, sr, 8000)
    buf = io.BytesIO()
    sf.write(buf, np.clip(y8, -1, 1), 8000, format="WAV", subtype="ULAW")
    buf.seek(0)
    y8, _ = sf.read(buf, dtype="float32")
    return soxr.resample(y8, 8000, sr)


def add_noise(x, snr_db, color="white", rng=None):
    rng = rng or np.random.default_rng()
    n = rng.standard_normal(len(x))
    if color == "pink":
        spec = np.fft.rfft(n)
        spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
        n = np.fft.irfft(spec, len(x))
    n = n / (np.linalg.norm(n) + 1e-12) * np.linalg.norm(x) / 10 ** (snr_db / 20)
    return (x + n).astype(np.float32)


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------

class Augmenter:
    """Applies each configured step in order, each with its own probability.

    Steps are stored as plain data (no closures) so the object pickles into
    Windows DataLoader worker processes.
    """

    def __init__(self, cfg: dict, sample_rate: int = 16000):
        self.cfg, self.sr = cfg, sample_rate
        self.steps: list[tuple[str, float, dict]] = []
        if "rawboost" in cfg:
            c = cfg["rawboost"]
            algos = list(c.get("algos", ["lnl", "isd"]))
            unknown = set(algos) - set(RAWBOOST)
            if unknown:
                raise ValueError(f"unknown RawBoost algos {unknown}")
            self.steps.append(("rawboost", c.get("p", 0.7), {"algos": algos}))
        if "codec" in cfg:
            c = cfg["codec"]
            types = list(c.get("types", ["mp3", "opus", "telephone"]))
            unknown = set(types) - {"mp3", "opus", "telephone"}
            if unknown:
                raise ValueError(f"unknown codecs {unknown}")
            self.steps.append(("codec", c.get("p", 0.5), {"types": types}))
        if "resample" in cfg:
            c = cfg["resample"]
            self.steps.append(("resample", c.get("p", 0.3), {"rates": list(c.get("rates", [8000, 11025, 12000]))}))
        if "speed" in cfg:
            c = cfg["speed"]
            self.steps.append(("speed", c.get("p", 0.3), {"lo": c.get("min", 0.9), "hi": c.get("max", 1.1)}))
        if "gain" in cfg:
            c = cfg["gain"]
            self.steps.append(("gain", c.get("p", 0.5), {"lo": c.get("min_db", -6.0), "hi": c.get("max_db", 6.0)}))
        if not self.steps:
            raise ValueError("augment block has no known steps (rawboost, codec, resample, speed, gain)")

    def _apply(self, name: str, args: dict, x: np.ndarray) -> np.ndarray:
        if name == "rawboost":
            for a in args["algos"]:
                x = RAWBOOST[a](x, self.sr)
            return x
        if name == "codec":
            t = random.choice(args["types"])
            if t == "mp3":
                return mp3(x, self.sr, random.uniform(0.0, 0.99))
            if t == "opus":
                return opus(x, self.sr, random.uniform(0.5, 0.99))
            return telephone(x, self.sr)
        if name == "resample":
            low = random.choice(args["rates"])
            return soxr.resample(soxr.resample(x, self.sr, low), low, self.sr)
        if name == "speed":
            # Play back at factor f: resample to sr/f and treat the result as sr.
            f = random.uniform(args["lo"], args["hi"])
            return soxr.resample(x, self.sr, self.sr / f)
        if name == "gain":
            return np.clip(x * 10 ** (random.uniform(args["lo"], args["hi"]) / 20), -1, 1)
        raise ValueError(f"unknown step {name!r}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        for name, p, args in self.steps:
            if random.random() < p:
                x = self._apply(name, args, x)
        x = np.asarray(x, dtype=np.float32)
        if len(x) > n:
            x = x[:n]
        elif len(x) < n:  # speed-up or codec delay: repeat the clip (never zero-pad, see P2.2)
            x = np.tile(x, int(np.ceil(n / max(len(x), 1))))[:n] if len(x) else np.zeros(n, np.float32)
        return x

    def __repr__(self):
        return " -> ".join(f"{name}(p={p})" for name, p, _ in self.steps)


def build_augmentation(cfg: dict | None, sample_rate: int = 16000, seed: int = 0):
    return Augmenter(cfg, sample_rate) if cfg else None
