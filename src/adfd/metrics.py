"""Evaluation metrics: EER, min t-DCF (ASVspoof 2019), AUC and threshold metrics.

Score convention (as in ASVspoof): a HIGHER score means MORE LIKELY BONA FIDE.
The models output a spoof logit, so the CM score is -logit.
Labels: 0 = bona fide, 1 = spoof.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
# EER (same algorithm as the official ASVspoof evaluation scripts)
# ----------------------------------------------------------------------------

def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """False-rejection and false-acceptance rates over all thresholds.

    target = bona fide trials, nontarget = spoof trials.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]
    tar_sums = np.cumsum(labels)
    nontar_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_sums)
    frr = np.concatenate(([0.0], tar_sums / target_scores.size))
    far = np.concatenate(([1.0], nontar_sums / nontarget_scores.size))
    thresholds = np.concatenate(([all_scores[order[0]] - 0.001], all_scores[order]))
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores) -> tuple[float, float]:
    """Return (EER, threshold). Accept as bona fide when score >= threshold."""
    target_scores = np.asarray(target_scores)
    nontarget_scores = np.asarray(nontarget_scores)
    if target_scores.size == 0 or nontarget_scores.size == 0:
        return float("nan"), float("nan")
    frr, far, thr = compute_det_curve(target_scores, nontarget_scores)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(thr[i])


def eer_from_labels(scores, labels) -> tuple[float, float]:
    scores, labels = np.asarray(scores), np.asarray(labels)
    return compute_eer(scores[labels == 0], scores[labels == 1])


# ----------------------------------------------------------------------------
# min t-DCF (ASVspoof 2019 formulation)
# ----------------------------------------------------------------------------

# ASVspoof 2019 cost model.
TDCF_COST_MODEL = {
    "Pspoof": 0.05,            # prior probability of a spoofing attack
    "Ptar": (1 - 0.05) * 0.99,  # prior of a target speaker
    "Pnon": (1 - 0.05) * 0.01,  # prior of a non-target speaker
    "Cmiss_asv": 1, "Cfa_asv": 10,
    "Cmiss_cm": 1, "Cfa_cm": 10,
}


@dataclass(frozen=True)
class AsvScores:
    target: np.ndarray
    nontarget: np.ndarray
    spoof: np.ndarray


def load_asv_scores(path: Path) -> AsvScores:
    """Official ASV score file: lines of 'SOURCE KEY SCORE' (KEY = target | nontarget | spoof)."""
    keys, scores = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                keys.append(parts[1])
                scores.append(float(parts[2]))
    keys, scores = np.array(keys), np.array(scores)
    return AsvScores(scores[keys == "target"], scores[keys == "nontarget"], scores[keys == "spoof"])


def min_tdcf(cm_bonafide, cm_spoof, asv: AsvScores, cost=TDCF_COST_MODEL) -> float:
    """Normalised minimum tandem detection cost function (ASVspoof 2019)."""
    # Fixed ASV system: operate at its EER threshold (target vs nontarget).
    _, asv_thr = compute_eer(asv.target, asv.nontarget)
    p_miss_asv = np.mean(asv.target < asv_thr)
    p_fa_asv = np.mean(asv.nontarget >= asv_thr)
    p_miss_spoof_asv = np.mean(asv.spoof < asv_thr)

    p_miss_cm, p_fa_cm, _ = compute_det_curve(np.asarray(cm_bonafide), np.asarray(cm_spoof))

    c1 = cost["Ptar"] * (cost["Cmiss_cm"] - cost["Cmiss_asv"] * p_miss_asv) - cost["Pnon"] * cost["Cfa_asv"] * p_fa_asv
    c2 = cost["Cfa_cm"] * cost["Pspoof"] * (1 - p_miss_spoof_asv)
    if c1 < 0 or c2 < 0:
        raise ValueError("t-DCF cost model gives negative weights; check the ASV scores")
    tdcf = c1 * p_miss_cm + c2 * p_fa_cm
    return float(np.min(tdcf / min(c1, c2)))


# ----------------------------------------------------------------------------
# Threshold-based metrics (spoof = positive class, as in the original paper)
# ----------------------------------------------------------------------------

def threshold_metrics(scores, labels, threshold: float) -> dict:
    """Accuracy / precision / recall / F1 for 'spoof' when score < threshold."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    pred_spoof = scores < threshold
    is_spoof = labels == 1
    tp = int(np.sum(pred_spoof & is_spoof))
    fp = int(np.sum(pred_spoof & ~is_spoof))
    fn = int(np.sum(~pred_spoof & is_spoof))
    tn = int(np.sum(~pred_spoof & ~is_spoof))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def bootstrap_eer(scores, labels, groups=None, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap confidence interval for the EER.

    groups: optional cluster ids (e.g. DEEP-VOICE recordings). Whole clusters are
    resampled, because windows of one recording are not independent.
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    rng = np.random.default_rng(seed)
    if groups is None:
        n = len(scores)
    else:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    eers = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
        e, _ = eer_from_labels(scores[idx], labels[idx])
        if e == e:
            eers.append(e)
    lo, hi = np.percentile(eers, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo_pct": round(100 * float(lo), 3), "hi_pct": round(100 * float(hi), 3), "n_boot": len(eers),
            "resampled": "utterances" if groups is None else "groups"}


def auc(scores, labels) -> float:
    """ROC AUC for detecting bona fide (higher score = bona fide)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels == 0, np.asarray(scores)))
