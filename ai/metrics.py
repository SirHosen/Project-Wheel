# -*- coding: utf-8 -*-
"""Probabilistic calibration metrics: log-loss and Brier score.

Top-1 accuracy only asks 'was the single guess right?'. For a 9-way wheel that
is a blunt instrument. Log-loss and the (multiclass) Brier score instead grade
the FULL predicted probability vector -- they reward a model for being both
right AND well-calibrated (honest about its confidence), and punish confident
mistakes hard. They are the right lens for 'are these probabilities trustworthy
enough to bet on?'. Pure numpy; no torch needed.
"""
import numpy as np


def _as_probs(probs):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    return p


def log_loss(y_idx, probs, eps=1e-12):
    """Mean negative log-likelihood of the true class. Lower is better; a
    uniform guess over K classes scores ln(K)."""
    p = _as_probs(probs)
    y = np.asarray(y_idx, dtype=int)
    rows = np.arange(len(y))
    picked = np.clip(p[rows, y], eps, 1.0)
    return float(-np.mean(np.log(picked)))


def brier_score(y_idx, probs):
    """Mean multiclass Brier score = mean sum_k (p_k - 1[k==truth])^2. Lower is
    better; ranges [0, 2]. A uniform guess scores 1 - 1/K."""
    p = _as_probs(probs)
    y = np.asarray(y_idx, dtype=int)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss_single(prob_vec, true_idx, eps=1e-12):
    """Negative log-likelihood of a single prediction."""
    return float(-np.log(max(eps, float(prob_vec[int(true_idx)]))))


def brier_single(prob_vec, true_idx):
    """Multiclass Brier score of a single prediction."""
    p = np.asarray(prob_vec, dtype=np.float64)
    oh = np.zeros_like(p)
    oh[int(true_idx)] = 1.0
    return float(np.sum((p - oh) ** 2))
