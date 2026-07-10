# -*- coding: utf-8 -*-
"""Statistical POWER analysis: 'how many spins until I could actually detect a
bias of a given size?'

The bias tracker is deliberately conservative -- it says SKIP until the evidence
is strong. This module answers the natural follow-up: if the wheel really were
biased by some amount, how many spins would I need before my tests would notice
(with decent probability)? It does this by Monte-Carlo: simulate many biased
sessions of size n and measure how often (a) the chi-square bias test fires and
(b) the tracker actually recommends a bet -- and how often that bet is the truly
biased number. Pure-numpy; no torch/scipy.
"""
import numpy as np

from config import VALID_NUMBERS
from core.wheel import design_distribution
from core.bias_tracker import OnlineBiasTracker


def biased_distribution(boost_number, boost_to, valid_numbers=None):
    """Return a probability dict where `boost_number` is lifted to probability
    `boost_to` and the remaining mass is shared among the others in proportion
    to the FAIR design distribution.
    """
    valid = list(valid_numbers) if valid_numbers else list(VALID_NUMBERS)
    if not (0.0 < boost_to < 1.0):
        raise ValueError("boost_to must be in (0, 1)")
    if boost_number not in valid:
        raise ValueError(f"{boost_number} not in valid numbers")
    design = design_distribution(valid_numbers=valid)
    rest = [n for n in valid if n != boost_number]
    rest_total = sum(design[n] for n in rest)
    probs = {boost_number: boost_to}
    scale = (1.0 - boost_to) / rest_total if rest_total > 0 else 0.0
    for n in rest:
        probs[n] = design[n] * scale
    return probs


def _sample(probs, n, rng):
    nums = list(probs.keys())
    p = np.array([probs[k] for k in nums], dtype=float)
    p = p / p.sum()
    draws = rng.choice(len(nums), size=int(n), p=p)
    return [nums[i] for i in draws]


def simulate_power(boost_number, boost_to, n, trials=300, seed=0,
                   valid_numbers=None):
    """Monte-Carlo the detection power at sample size `n` for a wheel biased so
    `boost_number` occurs with probability `boost_to`.

    Returns {n, trials, bias_power, bet_power, bet_on_target_power}:
      bias_power          = P(chi-square flags the wheel as biased)
      bet_power           = P(tracker recommends ANY bet)
      bet_on_target_power = P(tracker recommends betting the truly biased number)
    """
    rng = np.random.default_rng(seed)
    probs = biased_distribution(boost_number, boost_to, valid_numbers)
    biased_hits = bet_hits = target_hits = 0
    for _ in range(int(trials)):
        sample = _sample(probs, n, rng)
        t = OnlineBiasTracker(valid_numbers=valid_numbers)
        t.observe_many(sample)
        if t.bias_test()["biased"]:
            biased_hits += 1
        bb = t.best_bet()
        if bb is not None:
            bet_hits += 1
            if bb["number"] == boost_number:
                target_hits += 1
    trials = int(trials)
    return {"n": int(n), "trials": trials,
            "bias_power": biased_hits / trials,
            "bet_power": bet_hits / trials,
            "bet_on_target_power": target_hits / trials}


def sample_size_for_power(boost_number, boost_to, target_power=0.8,
                          metric="bias_power", ns=None, trials=200, seed=0,
                          valid_numbers=None):
    """Smallest simulated sample size whose `metric` reaches `target_power`.
    Returns (n, power) or (None, best_power) if no tested n reached it.
    """
    ns = ns or [25, 50, 75, 100, 150, 200, 300, 400, 600, 800]
    best = 0.0
    for n in ns:
        r = simulate_power(boost_number, boost_to, n, trials=trials, seed=seed,
                           valid_numbers=valid_numbers)
        best = max(best, r[metric])
        if r[metric] >= target_power:
            return n, r[metric]
    return None, best
