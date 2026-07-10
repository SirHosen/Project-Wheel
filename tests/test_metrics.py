# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Log-loss and Brier score: known-answer checks (pure numpy, no torch)."""
import numpy as np

from ai import metrics
from ai.dataset import NUM_CLASSES


def test_perfect_predictions_score_zero():
    y = [0, 3, 8]
    probs = np.zeros((3, NUM_CLASSES))
    for r, c in enumerate(y):
        probs[r, c] = 1.0
    assert metrics.log_loss(y, probs) < 1e-6
    assert metrics.brier_score(y, probs) < 1e-6
    print("OK perfect predictions -> ~0 log-loss and Brier")


def test_uniform_reference_values():
    K = NUM_CLASSES
    y = [0, 1, 2, 3]
    probs = np.full((len(y), K), 1.0 / K)
    assert abs(metrics.log_loss(y, probs) - np.log(K)) < 1e-9
    assert abs(metrics.brier_score(y, probs) - (1.0 - 1.0 / K)) < 1e-9
    print("OK uniform guess -> ln(K) log-loss and 1-1/K Brier")


def test_confident_mistake_is_punished():
    y = [0]
    wrong = np.zeros((1, NUM_CLASSES))
    wrong[0, NUM_CLASSES - 1] = 1.0 - 1e-9
    wrong[0, 0] = 1e-9
    uniform = np.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES)
    assert metrics.log_loss(y, wrong) > metrics.log_loss(y, uniform)
    assert metrics.brier_score(y, wrong) > metrics.brier_score(y, uniform)
    print("OK confident mistakes are punished harder than uniform")


if __name__ == "__main__":
    test_perfect_predictions_score_zero()
    test_uniform_reference_values()
    test_confident_mistake_is_punished()
    print("ALL CHECKS PASSED")
