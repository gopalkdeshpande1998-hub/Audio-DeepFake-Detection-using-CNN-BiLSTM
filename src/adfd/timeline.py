"""Time-local analysis helpers shared by the single-model analyzer (adfd.single) and the retired ensemble.

local_grid      analysis windows over the whole recording (hop grows for long files, at most MAX_LOCAL)
speech_windows  which windows carry speech-level energy
step_logodds    timeline steps with the mean log-odds of the valid windows covering each step centre
timeline_regions consecutive steps merged into Fake / Real / No speech regions
"""

from __future__ import annotations

import numpy as np

from adfd.inference import SR

DECISION_P = 0.5
LOCAL_S = 2.0            # default local window (s)
MIN_HOP_S = 0.5          # timeline resolution for clips up to ~1 min
MAX_LOCAL = 120          # local windows at most (CPU time); the hop grows for long files


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def local_grid(n: int, local_s: float = LOCAL_S) -> tuple[list[tuple[int, int]], float, float]:
    """Local windows (sample spans) over the whole recording, the timeline step and window length (s)."""
    dur = n / SR
    hop = MIN_HOP_S if dur <= local_s else max(MIN_HOP_S, (dur - local_s) / (MAX_LOCAL - 1))
    win = max(local_s, hop)
    if dur <= win:
        return [(0, n)], dur, dur
    starts = list(np.arange(0.0, dur - win + 1e-9, hop))
    if starts[-1] + win < dur - 1e-6:
        starts.append(dur - win)
    return [(int(round(s * SR)), min(n, int(round((s + win) * SR)))) for s in starts], hop, win


def speech_windows(x: np.ndarray, spans: list[tuple[int, int]], min_frac: float = 0.2) -> np.ndarray:
    """True for windows where at least `min_frac` of the 20 ms frames carry speech-level energy
    (within 40 dB of the loudest frame of the recording and above -70 dBFS)."""
    f = int(0.02 * SR)
    n = max(1, len(x) // f)
    xx = np.zeros(n * f) if len(x) < f else x[: n * f].astype(np.float64)
    db = 10 * np.log10(np.mean(xx.reshape(n, f) ** 2, axis=1) + 1e-12)
    active = (db > db.max() - 40) & (db > -70)
    return np.array([active[a // f: max(a // f + 1, b // f)].mean() >= min_frac for a, b in spans])


def step_logodds(spans: list[tuple[int, int]], z: np.ndarray, n: int, step: float,
                 valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Timeline steps of `step` s over the recording; each gets the mean log-odds of the valid
    local windows that contain its centre, or NaN when no valid window covers it."""
    dur = n / SR
    k = max(1, int(np.ceil(dur / step - 1e-9)))
    t0 = np.arange(k) * step
    centre = np.minimum(t0 + step / 2, dur - 1e-6)
    a = np.array([s for s, _ in spans]) / SR
    b = np.array([e for _, e in spans]) / SR
    z = np.asarray(z, float)
    ok = np.isfinite(z) if valid is None else (np.asarray(valid, bool) & np.isfinite(z))
    out = np.full(k, np.nan)
    for i, c in enumerate(centre):
        m = (a <= c) & (b > c) & ok
        if m.any():
            out[i] = z[m].mean()
    return t0, out


def timeline_regions(t0: np.ndarray, p: np.ndarray, step: float, dur: float) -> list[dict]:
    """Consecutive steps merged into Fake (p > 0.5), Real (p <= 0.5) and No speech (no score) regions."""
    out = []
    for t, q in zip(t0, p):
        lab = "No speech" if not np.isfinite(q) else ("Fake" if q > DECISION_P else "Real")   # 50/50 -> Real
        if out and out[-1]["label"] == lab:
            out[-1]["end_s"] = float(min(t + step, dur))
            out[-1]["p"].append(q)
        else:
            out.append({"label": lab, "start_s": float(t), "end_s": float(min(t + step, dur)), "p": [q]})
    for r in out:
        v = [q for q in r.pop("p") if np.isfinite(q)]
        r["p_fake_mean"] = float(np.mean(v)) if v else None
    return out


# ---------------------------------------------------------------------------------------
# segment-level authenticity blocks (display only)
# ---------------------------------------------------------------------------------------
BLOCK_SIZES = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0)
MAX_BLOCKS = 60
LABELS = ("REAL", "FAKE", "NO SPEECH")


def block_label(p: float | None) -> str:
    """Argmax of (P(real), P(fake)): FAKE only when P(fake) > P(real); a 50/50 tie is REAL. A block the model
    never scored (no speech in any of its windows) has no probability and stays NO SPEECH."""
    if p is None or not np.isfinite(p):
        return "NO SPEECH"
    return "FAKE" if p > 1.0 - p else "REAL"


def segment_blocks(per_step: list[dict], dur: float, step_s: float, window_s: float, time_resolved: bool,
                   max_blocks: int = MAX_BLOCKS) -> dict:
    """Group the model's timeline steps into equal display blocks (1 s for clips up to a minute).

    Every block gets the mean log-odds of the scored timeline steps whose centre falls inside it (the same
    model scores the timeline chart draws: 4-s windows of E7a, calibrated), labelled by argmax (FAKE if
    P(fake) > P(real), else REAL); blocks without any scored step are NO SPEECH. Confidence is the model's
    probability for the block's class (informational, never an accuracy and never a reason to withhold a label). When the recording fits in one
    analysis window (no time resolution) nothing is invented: 'available' is False with the reason.
    """
    scored = [w for w in per_step if w.get("p_fake") is not None]
    base = {"block_s": None, "window_s": float(window_s), "step_s": float(step_s), "duration_s": float(dur),
            "blocks": []}
    if not time_resolved or len(scored) < 2:
        why = (f"The recording fits in one {window_s:.0f}-s analysis window, so the model gives one score for the "
               "whole clip and cannot say which part is real or fake." if not scored or not time_resolved else
               "Fewer than two parts of the recording contain speech the model could score.")
        return {**base, "available": False, "reason": why}
    size = next((c for c in BLOCK_SIZES if c >= step_s - 1e-9 and np.ceil(dur / c - 1e-9) <= max_blocks),
                max(step_s, dur / max_blocks))
    edges = list(np.arange(0.0, dur - 1e-9, size)) + [dur]
    if len(edges) > 2 and edges[-1] - edges[-2] < size / 2:          # fold a short tail into the block before
        edges.pop(-2)
    centres = np.array([(w["start_s"] + w["end_s"]) / 2 for w in per_step])
    zs = np.array([np.nan if w.get("p_fake") is None else logit(w["p_fake"]) for w in per_step], float)
    blocks = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (centres >= a) & (centres < b) & np.isfinite(zs)
        p = float(1 / (1 + np.exp(-zs[m].mean()))) if m.any() else None
        lab = block_label(p)
        blocks.append({"start_s": float(a), "end_s": float(b), "label": lab, "p_fake": p, "n_steps": int(m.sum()),
                       "confidence": None if p is None else float(max(p, 1 - p))})
    secs = {k: sum(x["end_s"] - x["start_s"] for x in blocks if x["label"] == k) for k in LABELS}
    speech = secs["REAL"] + secs["FAKE"]
    pct = {k: (100 * secs[k] / speech if speech else 0.0) for k in ("REAL", "FAKE")}
    return {**base, "available": True, "reason": None, "block_s": float(size), "blocks": blocks,
            "seconds": secs, "pct_of_speech": pct, "analysed_s": float(speech),
            "counts": {k: sum(x["label"] == k for x in blocks) for k in LABELS}}
