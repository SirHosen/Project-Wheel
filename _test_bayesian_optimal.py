# -*- coding: utf-8 -*-
"""Tests for the Bayesian-Optimal predictor + EV/edge engine."""
import random

from predictors.bayesian_optimal import BayesianOptimalEngine
from config import settings


def _approx(a, b, tol=0.03):
    return abs(a - b) <= tol


def test_posterior_converges_to_true_frequency():
    """Feed many i.i.d. draws from a known distribution; posterior mean must
    converge to the empirical frequency."""
    eng = BayesianOptimalEngine()
    rng = random.Random(42)
    true = {1: 0.5, 2: 0.25, 5: 0.15, 10: 0.10}
    nums, probs = zip(*true.items())
    hist = rng.choices(nums, weights=probs, k=4000)
    preds = {p["number"]: p["confidence"] for p in eng.predict_next(hist)}
    for n, p in true.items():
        assert _approx(preds[n], p, 0.04), (n, preds[n], p)
    # top pick must be the true mode (1)
    assert eng.predict_next(hist)[0]["number"] == 1
    print("OK converge:", {k: round(preds[k], 3) for k in true})


def test_credible_interval_ordering():
    eng = BayesianOptimalEngine()
    hist = [1, 2, 5, 1, 2, 1, 10, 1, 2, 5] * 5
    for p in eng.predict_next(hist):
        assert p["ci_low"] <= p["confidence"] <= p["ci_high"] + 1e-9
    print("OK ci ordering")


def test_fair_wheel_recommends_skip():
    """On data drawn from the fair wheel layout, NO bet should clear the robust
    +EV bar -> recommend SKIP (all token_bet == 0)."""
    eng = BayesianOptimalEngine()
    rng = random.Random(7)
    seq = settings.SPINWHEEL_SEQUENCE
    hist = rng.choices(seq, k=600)  # fair draws from the real layout
    allocs = eng.recommend(hist, capital=1000, risk_pct=0.30)
    assert all(a["token_bet"] == 0 for a in allocs), allocs
    assert all(not a["is_positive_ev"] for a in allocs)
    print("OK fair wheel -> SKIP")


def test_real_bias_is_detected_and_bet():
    """Inject a strong, real bias toward a high-payout number (20) and confirm
    the engine detects a robust +EV edge and stakes on it."""
    eng = BayesianOptimalEngine()
    rng = random.Random(99)
    # 20 pays 20:1; break-even prob = 1/21 ~= 4.76%. Make it appear ~15%.
    base = [1, 1, 2, 2, 5, 10]
    hist = []
    for _ in range(500):
        hist.append(20 if rng.random() < 0.15 else rng.choice(base))
    report = {r["number"]: r for r in eng.edge_report(hist)}
    assert report[20]["robust_positive"], report[20]
    allocs = eng.recommend(hist, capital=1000, risk_pct=0.30)
    bet_numbers = [a["number"] for a in allocs if a["token_bet"] > 0]
    assert 20 in bet_numbers, allocs
    print("OK real bias detected & bet:", [(a["number"], a["token_bet"]) for a in allocs])


def test_no_bet_before_min_obs():
    """Even a wild early spike must NOT trigger a bet before min_obs spins."""
    eng = BayesianOptimalEngine()
    hist = [20, 20, 20]  # 3 spins, looks amazing, but far too little evidence
    allocs = eng.recommend(hist, capital=1000, risk_pct=0.30)
    assert all(a["token_bet"] == 0 for a in allocs), allocs
    print("OK no bet before min_obs")


if __name__ == "__main__":
    test_posterior_converges_to_true_frequency()
    test_credible_interval_ordering()
    test_fair_wheel_recommends_skip()
    test_real_bias_is_detected_and_bet()
    test_no_bet_before_min_obs()
    print("\nALL BAYESIAN-OPTIMAL TESTS PASSED")
