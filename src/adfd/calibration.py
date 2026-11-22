"""P4.1 calibration and P4.2 three-way decision, fitted on ASVspoof **dev** data only.

Calibration (Platt scaling): p_fake = sigmoid(a * z + b), z = spoof logit (= -score).
It is fitted with balanced class weights, so p_fake assumes equal prior odds of real
and fake (ASVspoof dev is 1 real : 9 fake, which should not leak into the app).
Fitting data: clean dev scores plus, when the model has them, its scores on the 10
degraded dev conditions (still dev clips). Clean dev alone is almost perfectly
separable for most models, which makes Platt scaling extremely over-confident.

Three-way decision: Fake if p_fake >= t_fake, Real if p_fake <= t_real, else Uncertain.
t_fake is set so at most `target`% of real dev clips are called Fake, t_real so at
most `target`% of fake dev clips are called Real. If the two overlap (a dev set that
is too easy), both collapse to the equal-error threshold and there is no band.

ECE (expected calibration error, 15 equal-width bins) is reported before (raw
sigmoid of the logit) and after calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from adfd.metrics import eer_from_labels


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


@dataclass
class Calibration:
    a: float
    b: float
    t_real: float
    t_fake: float
    eer_threshold_p: float
    fitted_on: str
    target_error_pct: float

    def prob_fake(self, z) -> np.ndarray:
        return sigmoid(self.a * np.asarray(z, dtype=float) + self.b)

    def decide(self, p: float) -> str:
        if p >= self.t_fake:
            return "Fake"
        if p <= self.t_real:
            return "Real"
        return "Uncertain"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def fit_calibration(z: np.ndarray, label: np.ndarray, fitted_on: str, target_pct: float = 5.0) -> Calibration:
    """z = spoof logits, label 1 = fake."""
    z, label = np.asarray(z, float), np.asarray(label, int)
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000).fit(z.reshape(-1, 1), label)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    p = sigmoid(a * z + b)
    t_fake = float(np.percentile(p[label == 0], 100 - target_pct))   # <= target% of real called Fake
    t_real = float(np.percentile(p[label == 1], target_pct))         # <= target% of fake called Real
    # equal-error threshold, in probability (scores: higher = real, so negate)
    _, thr = eer_from_labels(-p, label)
    p_eer = float(-thr)
    if t_real >= t_fake:
        t_real = t_fake = p_eer
    return Calibration(a, b, t_real, t_fake, p_eer, fitted_on, target_pct)


def ece(p_fake: np.ndarray, label: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the predicted class probability, in %."""
    p_fake, label = np.asarray(p_fake, float), np.asarray(label, int)
    pred = (p_fake >= 0.5).astype(int)
    conf = np.where(pred == 1, p_fake, 1 - p_fake)
    correct = (pred == label).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    total = 0.0
    for i in range(bins):
        m = idx == i
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100 * total


def reliability(p_fake: np.ndarray, label: np.ndarray, bins: int = 10):
    """(bin centre, observed fraction fake, count) for a reliability diagram over p_fake."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_fake, edges) - 1, 0, bins - 1)
    rows = []
    for i in range(bins):
        m = idx == i
        rows.append(((edges[i] + edges[i + 1]) / 2, float(label[m].mean()) if m.any() else np.nan, int(m.sum())))
    return rows


def three_way_stats(p_fake: np.ndarray, label: np.ndarray, cal: Calibration) -> dict:
    """Share of clips per verdict, and the error rate among the decided ones."""
    fake = p_fake >= cal.t_fake
    real = p_fake <= cal.t_real
    unc = ~(fake | real)
    decided = ~unc
    err = ((fake & (label == 0)) | (real & (label == 1)))
    return {"pct_real": 100 * real.mean(), "pct_uncertain": 100 * unc.mean(), "pct_fake": 100 * fake.mean(),
            "error_pct_decided": 100 * err[decided].mean() if decided.any() else float("nan"),
            "real_called_fake_pct": 100 * fake[label == 0].mean() if (label == 0).any() else float("nan"),
            "fake_called_real_pct": 100 * real[label == 1].mean() if (label == 1).any() else float("nan")}
