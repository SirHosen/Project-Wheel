# -*- coding: utf-8 -*-
"""Turn a stream of results into supervised training windows for the LSTM.

This is a clean, self-contained data layer for the AI-training playground:
  * to_indices / from_indices : map wheel numbers <-> class indices (0..8)
  * make_windows              : sliding windows X=[last N results], y=next result
  * synthetic_fair / synthetic_biased / synthetic_markov : generate practice
    data so you can SEE when an LSTM has something to learn vs nothing to learn.
    This is the whole pedagogical point.

Needs numpy only.
"""
import random

import numpy as np

from config import SEQUENCE_LENGTH, VALID_NUMBERS
from core.wheel import design_distribution

INDEX = {n: i for i, n in enumerate(VALID_NUMBERS)}
NUMBER = {i: n for n, i in INDEX.items()}
NUM_CLASSES = len(VALID_NUMBERS)


def to_indices(numbers):
    return [INDEX[n] for n in numbers if n in INDEX]


def from_indices(indices):
    return [NUMBER[i] for i in indices]


def make_windows(numbers, seq_len=SEQUENCE_LENGTH):
    """Return (X, y) int arrays: X shape (m, seq_len), y shape (m,)."""
    idx = to_indices(numbers)
    X, y = [], []
    for i in range(len(idx) - seq_len):
        X.append(idx[i:i + seq_len])
        y.append(idx[i + seq_len])
    if not X:
        return np.empty((0, seq_len), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.asarray(X, dtype=np.int32), np.asarray(y, dtype=np.int32)


# --- richer, CAUSAL features ------------------------------------------------
# Each feature at timestep t is computed using ONLY spins up to and including t,
# so nothing leaks from the future into training (which would inflate accuracy).
FEATURE_NAMES = ("running_freq", "recency_gap", "base_rate", "is_repeat")
FEATURE_DIM = len(FEATURE_NAMES)


def compute_features(numbers, seq_len=SEQUENCE_LENGTH):
    """Return a (L, FEATURE_DIM) float32 array of per-timestep context features:

      running_freq : how often this number has come up so far (count / total)
      recency_gap  : spins since it last appeared, normalized by seq_len (1.0
                     if never seen before) -- captures 'it's been a while'
      base_rate    : the number's FAIR design probability (static prior)
      is_repeat    : 1.0 if it equals the immediately previous spin

    These give the model cheap, meaningful signal beyond the raw index, which is
    all an Embedding alone provides.
    """
    idx = to_indices(numbers)
    length = len(idx)
    design = design_distribution()
    base_rate = np.array([design[NUMBER[i]] for i in range(NUM_CLASSES)],
                         dtype=np.float64)
    feats = np.zeros((length, FEATURE_DIM), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    last_seen = [-1] * NUM_CLASSES
    denom = float(max(1, seq_len))
    for t, c in enumerate(idx):
        prev_last = last_seen[c]
        counts[c] += 1.0
        running_freq = counts[c] / float(t + 1)
        if prev_last < 0:
            recency = 1.0
        else:
            recency = min(t - prev_last, seq_len) / denom
        is_repeat = 1.0 if (t > 0 and idx[t] == idx[t - 1]) else 0.0
        feats[t, 0] = running_freq
        feats[t, 1] = recency
        feats[t, 2] = base_rate[c]
        feats[t, 3] = is_repeat
        last_seen[c] = t
    return feats


def make_feature_windows(numbers, seq_len=SEQUENCE_LENGTH):
    """Like make_windows, but also returns per-timestep features.

    Returns (X_idx, X_feat, y):
      X_idx  int32   (m, seq_len)
      X_feat float32 (m, seq_len, FEATURE_DIM)
      y      int32   (m,)
    """
    idx = to_indices(numbers)
    feats = compute_features(numbers, seq_len)
    X, Xf, y = [], [], []
    for i in range(len(idx) - seq_len):
        X.append(idx[i:i + seq_len])
        Xf.append(feats[i:i + seq_len])
        y.append(idx[i + seq_len])
    if not X:
        return (np.empty((0, seq_len), dtype=np.int32),
                np.empty((0, seq_len, FEATURE_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    return (np.asarray(X, dtype=np.int32),
            np.asarray(Xf, dtype=np.float32),
            np.asarray(y, dtype=np.int32))


def _weighted_draw(probs, rng):
    r = rng.random()
    acc = 0.0
    for n, p in probs.items():
        acc += p
        if r <= acc:
            return n
    return list(probs)[-1]


def synthetic_fair(n_spins, seed=0):
    """Draw i.i.d. results from the FAIR design distribution. An LSTM can learn
    the base rates here but NOT the order -- there is no sequence pattern."""
    rng = random.Random(seed)
    probs = design_distribution()
    return [_weighted_draw(probs, rng) for _ in range(n_spins)]


def synthetic_biased(n_spins, bias_number=8, strength=3.0, seed=0):
    """A deliberately BIASED wheel: `bias_number` is over-weighted. Useful to
    prove the pipeline + bias tracker really detect an edge when one exists."""
    rng = random.Random(seed)
    base = design_distribution()
    w = dict(base)
    w[bias_number] = w.get(bias_number, 0.0) * strength + 0.05
    tot = sum(w.values())
    probs = {k: v / tot for k, v in w.items()}
    return [_weighted_draw(probs, rng) for _ in range(n_spins)]


def synthetic_markov(n_spins, repeat_prob=0.5, seed=0):
    """A wheel with a fake SEQUENTIAL pattern: each spin repeats the previous one
    with probability `repeat_prob`, else draws fair. Real wheels are NOT like
    this -- it exists so you can watch the LSTM actually beat the baseline when
    order genuinely matters, then contrast with fair data where it can't."""
    rng = random.Random(seed)
    probs = design_distribution()
    out = [_weighted_draw(probs, rng)]
    for _ in range(1, n_spins):
        out.append(out[-1] if rng.random() < repeat_prob
                   else _weighted_draw(probs, rng))
    return out
