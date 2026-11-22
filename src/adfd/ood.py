"""P4.3 out-of-distribution (OOD) check: does the input look like speech at all?

A deepfake detector trained on speech gives confident but meaningless answers on
tones, noise, music or silence. Two layers of checks:

1. Signal checks (model-free, cheap), on 16 kHz mono audio:
     duration_s        total length
     level_dbfs        RMS level of the loudest 90% of frames
     speech_ratio      fraction of frames within 40 dB of the loudest frame
     flatness          median spectral flatness (1 = white noise, ~0 = tonal)
     tonality          median share of frame energy in the 3 strongest FFT bins (pure tones ~1)
     voiced_ratio      fraction of active frames with a clear pitch in 70-400 Hz (autocorrelation)
     syllabic_mod      share of envelope modulation energy at 2-8 Hz (speech rhythm)
   Each check is one-sided (too quiet, mostly silence, noise-like, tone-like, unvoiced,
   no speech rhythm): its limit is the 0.5th or 99.5th percentile of bona fide ASVspoof
   **dev** speech, widened by a margin and never stricter than a fixed floor, so that
   noisy real-world recordings (In-the-Wild) are not flagged.
2. Embedding check: Mahalanobis distance of the model's penultimate-layer
   embedding from the dev embeddings (`EmbeddingOOD`); flagged above the 99th
   percentile of dev distances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 16000
FRAME, HOP = 400, 160          # 25 ms / 10 ms
FEATURES = ["level_dbfs", "speech_ratio", "flatness", "tonality", "voiced_ratio", "syllabic_mod"]
MIN_DURATION_S = 1.0


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def signal_features(x: np.ndarray) -> dict:
    """Model-free features of a float waveform in [-1, 1] at 16 kHz."""
    x = np.asarray(x, dtype=np.float64)
    f = _frames(x)
    rms = np.sqrt(np.mean(f ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    active = db > db.max() - 40
    loud = np.sort(rms)[int(0.1 * len(rms)):] if len(rms) > 10 else rms
    spec = np.abs(np.fft.rfft(f * np.hanning(FRAME), axis=1)) ** 2 + 1e-12
    act_spec = spec[active] if active.any() else spec
    flatness = np.exp(np.mean(np.log(act_spec), axis=1)) / np.mean(act_spec, axis=1)
    top3 = np.sort(act_spec, axis=1)[:, -3:].sum(axis=1) / act_spec.sum(axis=1)
    # pitch: normalised autocorrelation peak between 70 and 400 Hz on active frames
    af = f[active] - f[active].mean(axis=1, keepdims=True) if active.any() else f
    ac = np.fft.irfft(np.abs(np.fft.rfft(af, n=2 * FRAME, axis=1)) ** 2, axis=1)[:, :FRAME]
    lo, hi = SR // 400, SR // 70
    voiced = (ac[:, lo:hi].max(axis=1) / (ac[:, 0] + 1e-12)) > 0.5
    # syllabic modulation: spectrum of the frame-energy envelope (100 Hz frame rate)
    env = rms - rms.mean()
    if len(env) >= 64:
        m = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
        fm = np.fft.rfftfreq(len(env), d=HOP / SR)
        band = m[(fm >= 2) & (fm <= 8)].sum()
        total = m[(fm >= 0.5) & (fm <= 20)].sum() + 1e-12
        syl = float(band / total)
    else:
        syl = float("nan")
    return {"duration_s": len(x) / SR,
            "level_dbfs": float(20 * np.log10(np.sqrt(np.mean(loud ** 2)) + 1e-12)),
            "speech_ratio": float(active.mean()),
            "flatness": float(np.median(flatness)),
            "tonality": float(np.median(top3)),
            "voiced_ratio": float(voiced.mean()),
            "syllabic_mod": syl}


# Which side of each feature means "not speech", and the loosest allowed limit.
DIRECTION = {"level_dbfs": "low", "speech_ratio": "low", "flatness": "high", "tonality": "high",
             "voiced_ratio": "low", "syllabic_mod": "low"}
FLOOR = {"level_dbfs": -50.0, "speech_ratio": 0.15, "flatness": 0.25, "tonality": 0.9, "voiced_ratio": 0.03,
         "syllabic_mod": 0.08}
# (low, high) limits; None = no limit on that side. Used when no fitted ranges are available
# (scripts/fit_reliability.py writes fitted ones next to each model).
DEFAULT_RANGES = {k: ((FLOOR[k], None) if d == "low" else (None, FLOOR[k])) for k, d in DIRECTION.items()}
REASONS = {"level_dbfs": "the audio is almost silent", "speech_ratio": "most of the audio is silence",
           "flatness": "it sounds like noise rather than speech", "tonality": "it sounds like a steady tone or hum",
           "voiced_ratio": "there is almost no voiced (pitched) sound", "syllabic_mod": "it lacks the rhythm of speech"}


def fit_signal_ranges(feats: list[dict], lo_q: float = 0.5, hi_q: float = 99.5) -> dict:
    """One-sided limit per feature from bona fide dev speech: percentile, 10% margin, floor."""
    out = {}
    for k in FEATURES:
        v = np.array([f[k] for f in feats if np.isfinite(f[k])])
        lo, hi = np.percentile(v, lo_q), np.percentile(v, hi_q)
        pad = 0.1 * (hi - lo)
        if DIRECTION[k] == "low":
            out[k] = (float(min(lo - pad, FLOOR[k])), None)
        else:
            out[k] = (None, float(max(hi + pad, FLOOR[k])))
    return out


def check_signal(feats: dict, ranges: dict | None = None) -> list[str]:
    """Reasons the input does not look like speech (empty list = looks like speech)."""
    ranges = ranges or DEFAULT_RANGES
    reasons = []
    if feats["duration_s"] < MIN_DURATION_S:
        reasons.append(f"the clip is shorter than {MIN_DURATION_S:.0f} s")
    for k in FEATURES:
        v = feats[k]
        if k not in ranges or not np.isfinite(v):
            continue
        lo, hi = ranges[k]
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            reasons.append(REASONS[k])
    return reasons


@dataclass
class EmbeddingOOD:
    """Mahalanobis distance of an embedding from the dev embedding distribution."""
    mean: np.ndarray
    precision: np.ndarray
    threshold: float
    extra: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, emb: np.ndarray, q: float = 99.0, shrink: float = 0.1) -> "EmbeddingOOD":
        mu = emb.mean(axis=0)
        cov = np.cov(emb - mu, rowvar=False)
        cov = (1 - shrink) * cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        prec = np.linalg.inv(cov)
        obj = cls(mu, prec, 0.0)
        obj.threshold = float(np.percentile(obj.distance(emb), q))
        return obj

    def distance(self, emb: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(emb) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision, d))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "precision": self.precision.tolist(), "threshold": self.threshold}

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingOOD":
        return cls(np.array(d["mean"]), np.array(d["precision"]), float(d["threshold"]))
