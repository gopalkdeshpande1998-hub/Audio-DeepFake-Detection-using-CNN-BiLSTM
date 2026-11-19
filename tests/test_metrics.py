"""Metric correctness on synthetic scores."""

import numpy as np
import pytest

from adfd.metrics import AsvScores, auc, compute_eer, eer_from_labels, min_tdcf, threshold_metrics


def test_eer_perfect_and_inverted():
    assert compute_eer([2.0, 3.0], [0.0, 1.0])[0] == 0.0
    assert compute_eer([0.0, 1.0], [2.0, 3.0])[0] == 1.0


def test_eer_matches_gaussian_theory():
    # Two unit-variance Gaussians 2 apart: EER = Phi(-1) = 0.1587.
    rng = np.random.default_rng(0)
    eer, thr = compute_eer(rng.normal(1, 1, 200_000), rng.normal(-1, 1, 200_000))
    assert eer == pytest.approx(0.1587, abs=0.003)
    assert thr == pytest.approx(0.0, abs=0.02)


def test_eer_from_labels_uses_label_zero_as_bonafide():
    scores = np.array([3.0, 2.0, -1.0, -2.0])
    labels = np.array([0, 0, 1, 1])
    assert eer_from_labels(scores, labels)[0] == 0.0


def test_min_tdcf_zero_for_perfect_cm_and_bounded():
    rng = np.random.default_rng(1)
    asv = AsvScores(target=rng.normal(3, 1, 5000), nontarget=rng.normal(-3, 1, 5000),
                    spoof=rng.normal(2, 1, 5000))
    assert min_tdcf(np.full(100, 5.0), np.full(100, -5.0), asv) == pytest.approx(0.0)
    # A useless CM (same score for everything) costs at most 1 when normalised.
    useless = min_tdcf(np.zeros(100), np.zeros(100), asv)
    assert 0.0 < useless <= 1.0 + 1e-9


def test_threshold_metrics_treat_spoof_as_positive():
    scores = np.array([2.0, 1.0, -1.0, -2.0, 0.5])
    labels = np.array([0, 0, 1, 1, 1])
    m = threshold_metrics(scores, labels, threshold=0.0)
    assert m["confusion"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 2}
    assert m["precision"] == 1.0
    assert m["recall"] == pytest.approx(2 / 3)


def test_auc_bonafide_high():
    assert auc([3.0, 2.0, -1.0], [0, 0, 1]) == 1.0
