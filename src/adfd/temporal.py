"""Temporal (segment-level) inference with the frame-head model (configs/t1_wavlm_crnn_temporal.yaml).

    ta = TemporalAnalyzer("runs/t1_wavlm_crnn_temporal_s42", device="cuda")
    out = ta.analyze(x)        # x: float32 mono 16 kHz, any length

How a recording is scored
  1. The audio is padded to the 0.16-s grid. Windows of WIN_FRAMES frames (4 s, the training crop) start every
     HOP_FRAMES frames (1.92 s), so every frame is seen by one or two windows; the last window is aligned to the
     end. Each window gives one logit per 0.16-s frame (frame head) and one clip logit (clip head).
  2. Overlapping frame logits are averaged with weights that favour the centre of each window (edge frames have
     less context). There is no further smoothing, so a fake run of a single 0.16-s frame can still show.
  3. Frame logits -> calibrated P(fake) (Platt scaling fitted on validation frames, temporal_calibration.json);
     P(real) = 1 - P(fake). Every frame gets the class with the higher probability: FAKE if P(fake) > P(real),
     otherwise REAL (a 50/50 tie is REAL). No threshold band: confidence never withholds a label.
  4. Silent frames (speech energy within 40 dB of the loudest frame and above -70 dBFS for less than 30% of the
     frame) are NO SPEECH, drawn grey: silence is neither called real nor fake. Every speech frame is REAL or FAKE.
  5. Consecutive frames with the same label form regions; each region's confidence is the mean calibrated
     probability of its class (informational, it never changes the label).
The whole-clip verdict stays separate (adfd.single, E7a); the clip head here is reported as clip_p_fake only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from adfd.data.partialspoof import EXTRA, FRAME_S, HOP, samples_for
from adfd.device import resolve as resolve_device
from adfd.models import build_model

WIN_FRAMES = 25
HOP_FRAMES = 12
LABELS = ("REAL", "FAKE")
SILENCE = "NO SPEECH"            # silent frames: shown grey, never REAL / FAKE
DEFAULT_CAL = {"a": 1.0, "b": 0.0, "fitted": False}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def window_starts(n_frames: int, win: int = WIN_FRAMES, hop: int = HOP_FRAMES) -> list[int]:
    if n_frames <= win:
        return [0]
    s = list(range(0, n_frames - win + 1, hop))
    if s[-1] != n_frames - win:
        s.append(n_frames - win)
    return s


def centre_weights(win: int = WIN_FRAMES) -> np.ndarray:
    """Triangular weights, 0.25 at the window edges and 1 in the middle."""
    t = np.abs(np.arange(win) - (win - 1) / 2) / ((win - 1) / 2)
    return 1.0 - 0.75 * t


def speech_frames(x: np.ndarray, n_frames: int, min_frac: float = 0.3) -> np.ndarray:
    """True for 0.16-s frames whose 20-ms sub-frames carry speech-level energy at least min_frac of the time."""
    f = 320
    m = max(1, len(x) // f)
    xx = np.zeros(m * f) if len(x) < f else x[: m * f].astype(np.float64)
    db = 10 * np.log10(np.mean(xx.reshape(m, f) ** 2, axis=1) + 1e-12)
    active = (db > db.max() - 40) & (db > -70)
    per = HOP // f                                      # 8 sub-frames per 0.16-s frame
    out = np.zeros(n_frames, bool)
    for k in range(n_frames):
        a = active[k * per:(k + 1) * per]
        out[k] = a.size > 0 and a.mean() >= min_frac
    return out


def binary_label(p_fake: float) -> str:
    """Argmax of (P(real), P(fake)) with P(real) = 1 - P(fake): FAKE only when P(fake) > P(real); ties are REAL."""
    return "FAKE" if p_fake > 1.0 - p_fake else "REAL"


def frame_labels(p: np.ndarray, speech: np.ndarray | None = None) -> list[str]:
    """REAL / FAKE by argmax for speech frames; NO SPEECH for silent ones (speech=None: every frame is speech)."""
    speech = np.ones(len(p), bool) if speech is None else np.asarray(speech, bool)
    return [binary_label(float(q)) if s else SILENCE for q, s in zip(p, speech)]


def regions(p: np.ndarray, labels: list[str], dur: float) -> list[dict]:
    """Consecutive frames with one label -> regions with start/end (s), label, confidence and mean P(fake)."""
    out = []
    for k, lab in enumerate(labels):
        if out and out[-1]["label"] == lab:
            out[-1]["_k"].append(k)
        else:
            out.append({"label": lab, "_k": [k]})
    for r in out:
        ks = r.pop("_k")
        r["start_s"] = round(ks[0] * FRAME_S, 3)
        r["end_s"] = round(min((ks[-1] + 1) * FRAME_S, dur), 3)
        q = p[ks]
        silent = r["label"] == SILENCE
        r["p_fake"] = None if silent else float(q.mean())
        r["p_real"] = None if silent else float(1 - q.mean())
        r["confidence"] = None if silent else r["p_fake"] if r["label"] == "FAKE" else r["p_real"]
        r["n_frames"] = len(ks)
    return [r for r in out if r["end_s"] > r["start_s"]]


class TemporalAnalyzer:
    def __init__(self, run_dir, device: str = "cpu", batch: int = 8):
        self.run_dir = Path(run_dir)
        ck = torch.load(self.run_dir / "best.pt", map_location="cpu", weights_only=False)
        cfg = ck["config"]
        cfg["model"]["frontend"]["grad_checkpointing"] = False
        self.model = build_model(cfg)
        self.model.load_state_dict(ck["model"])
        self.device = torch.device(resolve_device(device))      # cuda / mps (macOS) / cpu
        self.model.to(self.device).eval()
        self.batch = batch
        self.name = self.run_dir.name
        cal = self.run_dir / "temporal_calibration.json"
        self.cal = {**DEFAULT_CAL, **json.loads(cal.read_text(encoding="utf-8"))} if cal.exists() else dict(DEFAULT_CAL)

    @torch.no_grad()
    def logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """(frame logits (n_frames,), window clip logits, n_frames) for float32 16-kHz audio."""
        n_frames = max(1, int(np.ceil(len(x) / HOP)))
        span = max(n_frames, WIN_FRAMES)
        xx = np.zeros(samples_for(span), np.float32)
        xx[: len(x)] = x[: len(xx)]
        if n_frames < WIN_FRAMES and len(x) > 0:        # short clip: repeat it (as in training), score frames once
            core = xx[: n_frames * HOP]
            rep = np.tile(core, int(np.ceil(WIN_FRAMES / n_frames)) + 1)[: WIN_FRAMES * HOP]
            xx = np.concatenate([rep, np.zeros(EXTRA, np.float32)])
        starts = window_starts(span)
        acc = np.zeros(span)
        wsum = np.zeros(span)
        clips = []
        w = centre_weights()
        amp = self.device.type == "cuda"
        for i in range(0, len(starts), self.batch):
            chunk = starts[i:i + self.batch]
            wav = torch.from_numpy(np.stack([xx[s * HOP: s * HOP + samples_for(WIN_FRAMES)] for s in chunk]))
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                c, f = self.model.forward_frames(wav.to(self.device))
            f = f.float().cpu().numpy()
            clips.extend(c.float().cpu().numpy().tolist())
            for s, row in zip(chunk, f):
                k = min(len(row), WIN_FRAMES)
                acc[s:s + k] += row[:k] * w[:k]
                wsum[s:s + k] += w[:k]
        z = acc / np.maximum(wsum, 1e-9)
        if n_frames < WIN_FRAMES:                       # average the repeats of each original frame
            reps = z[:WIN_FRAMES]
            z = np.array([reps[k::n_frames].mean() for k in range(n_frames)])
        return z[:n_frames], np.asarray(clips, float), n_frames

    def prob(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(self.cal["a"] * np.asarray(z, float) + self.cal["b"])

    def analyze(self, x: np.ndarray, progress=None) -> dict:
        dur = len(x) / 16000
        z, clips, n = self.logits(x)
        p = self.prob(z)
        speech = speech_frames(x, n)
        labs = frame_labels(p, speech)
        regs = regions(p, labs, dur)
        secs = {k: sum(r["end_s"] - r["start_s"] for r in regs if r["label"] == k) for k in LABELS + (SILENCE,)}
        scored = secs["REAL"] + secs["FAKE"]
        return {
            "model": self.name, "frame_s": FRAME_S, "duration_s": dur, "calibrated": bool(self.cal.get("fitted")),
            "decision": "argmax: FAKE if P(fake) > P(real), else REAL; silent frames NO SPEECH",
            "frames": [{"start_s": round(k * FRAME_S, 3), "end_s": round(min((k + 1) * FRAME_S, dur), 3),
                        "p_fake": float(q), "p_real": float(1 - q), "speech": bool(s), "label": lab}
                       for k, (q, s, lab) in enumerate(zip(p, speech, labs))],
            "regions": regs,
            "seconds": secs, "analysed_s": scored,
            "pct_of_speech": {k: (100 * secs[k] / scored if scored else 0.0) for k in LABELS},
            "counts": {k: sum(r["label"] == k for r in regs) for k in LABELS + (SILENCE,)},
            "clip_p_fake": float(sigmoid(np.mean(clips))) if len(clips) else None,
        }
