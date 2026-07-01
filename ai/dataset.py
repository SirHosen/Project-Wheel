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
