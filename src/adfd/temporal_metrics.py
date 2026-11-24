"""Metrics for segment-level fake detection on 0.16-s frames (1 = fake, 0 = real).

frame_metrics     confusion matrix, precision / recall / F1 for FAKE, FPR, FNR, accuracy at one threshold
runs              fake regions (start, end) in frames from a 0/1 vector
segment_metrics   region precision / recall / F1 with IoU >= 0.5 one-to-one matching, and the boundary error
                  (|start| and |end| difference in seconds) of matched regions
short_recall      share of ground-truth fake regions, by length, whose frames are at least half predicted FAKE
"""

from __future__ import annotations

import numpy as np

FRAME_S = 0.16
LENGTH_BINS = [(0.0, 0.5, "< 0.5 s"), (0.5, 1.0, "0.5-1 s"), (1.0, 2.0, "1-2 s"), (2.0, 1e9, ">= 2 s")]


def frame_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(y), "confusion": {"tp_fake": tp, "fp_real_called_fake": fp, "fn_fake_called_real": fn,
                                       "tn_real": tn},
            "accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0}


def runs(v: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, np.asarray(v, int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _iou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def segment_counts(y: np.ndarray, pred: np.ndarray, iou: float = 0.5) -> dict:
    """One utterance: matched / ground-truth / predicted fake regions and boundary errors (frames)."""
    gt, pr = runs(y), runs(pred)
    pairs = sorted(((_iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pr)), reverse=True)
    used_g, used_p, starts, ends = set(), set(), [], []
    for v, i, j in pairs:
        if v < iou:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        starts.append(abs(gt[i][0] - pr[j][0]))
        ends.append(abs(gt[i][1] - pr[j][1]))
    return {"tp": len(used_g), "n_gt": len(gt), "n_pred": len(pr), "start_err": starts, "end_err": ends}


def segment_summary(counts: list[dict]) -> dict:
    tp = sum(c["tp"] for c in counts)
    ng = sum(c["n_gt"] for c in counts)
    npr = sum(c["n_pred"] for c in counts)
    p = tp / npr if npr else 0.0
    r = tp / ng if ng else 0.0
    se = [e for c in counts for e in c["start_err"]]
    ee = [e for c in counts for e in c["end_err"]]
    return {"gt_regions": ng, "pred_regions": npr, "matched": tp, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "boundary_error_s": {"start_mean": float(np.mean(se)) * FRAME_S if se else None,
                                 "end_mean": float(np.mean(ee)) * FRAME_S if ee else None,
                                 "start_median": float(np.median(se)) * FRAME_S if se else None,
                                 "end_median": float(np.median(ee)) * FRAME_S if ee else None}}


def short_recall(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Recall of ground-truth fake regions by length: detected if >= half of its frames are predicted FAKE."""
    hit = {b[2]: [0, 0] for b in LENGTH_BINS}
    for y, pred in pairs:
        for a, b in runs(y):
            L = (b - a) * FRAME_S
            name = next(n for lo, hi, n in LENGTH_BINS if lo <= L < hi)
            hit[name][1] += 1
            hit[name][0] += int(np.mean(np.asarray(pred[a:b], bool)) >= 0.5)
    return {k: {"regions": n, "detected": d, "recall": d / n if n else None} for k, (d, n) in hit.items()}
