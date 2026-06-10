# -*- coding: utf-8 -*-
"""Test PROMPT 8: bootstrap CI + support badge rendering logic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.bootstrap_ci import (
    _percentile,
    bootstrap_ci_for_numbers,
    attach_confidence_intervals,
    support_label,
)
from predictors.heuristic_engine import HeuristicEngine
from predictors.markov_engine import MarkovEngine
from predictors.bayesian_optimal import BayesianOptimalEngine
from config import settings

SEQ = settings.SPINWHEEL_SEQUENCE


def _history(n, seed=7):
    import random
    rng = random.Random(seed)
    return [rng.choice(SEQ) for _ in range(n)]


def test_percentile_basics():
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert abs(_percentile(vals, 0.0) - 0.0) < 1e-9
    assert abs(_percentile(vals, 1.0) - 1.0) < 1e-9
    assert abs(_percentile(vals, 0.5) - 0.5) < 1e-9
    assert _percentile([], 0.5) == 0.0
    assert _percentile([0.3], 0.5) == 0.3
    print("OK percentile helper")


def test_bootstrap_ci_bounds_valid():
    eng = HeuristicEngine()
    hist = _history(120)
    ci = bootstrap_ci_for_numbers(eng, hist, settings.VALID_NUMBERS, n_boot=120, seed=1)
    for num, (lo, hi) in ci.items():
        assert 0.0 <= lo <= hi <= 1.0, (num, lo, hi)
    print("OK bootstrap CI bounds valid & ordered (low <= high, within [0,1])")


def test_bootstrap_is_deterministic_with_seed():
    eng = HeuristicEngine()
    hist = _history(80)
    a = bootstrap_ci_for_numbers(eng, hist, [1, 2], n_boot=100, seed=99)
    b = bootstrap_ci_for_numbers(eng, hist, [1, 2], n_boot=100, seed=99)
    assert a == b
    print("OK bootstrap reproducible under fixed seed")


def test_attach_fills_missing_ci_and_support():
    eng = HeuristicEngine()
    hist = _history(60)
    preds = eng.predict_next(hist)
    # Heuristic has no native CI/support.
    assert all(p.get("ci_low") is None for p in preds)
    attach_confidence_intervals(eng, hist, preds, n_boot=60)
    top = preds[:3]
    for p in top:
        assert p["ci_low"] is not None and p["ci_high"] is not None
        assert p["ci_low"] <= p["confidence"] + 1e-6 or p["ci_high"] >= p["confidence"] - 1e-6
        assert p["support"] == len(hist)  # fallback to observation count
    print("OK attach fills CI for top picks + support fallback = len(history)")


def test_bayesian_native_ci_untouched():
    eng = BayesianOptimalEngine()
    hist = _history(150)
    preds = eng.recommend(hist, 1000, 0.30)
    before = [(p.get("ci_low"), p.get("ci_high")) for p in preds]
    attach_confidence_intervals(eng, hist, preds, n_boot=30)
    after = [(p.get("ci_low"), p.get("ci_high")) for p in preds]
    assert before == after, "native Bayesian CI must not be overwritten"
    print("OK Bayesian native CI preserved (not overwritten by bootstrap)")


def test_support_label_tiers():
    assert support_label(None) == ("RAW (cold start)", "error")
    assert support_label(0) == ("RAW (cold start)", "error")
    assert support_label(24) == ("RAW (cold start)", "error")
    assert support_label(25) == ("WARMING UP", "secondary")
    assert support_label(100) == ("WARMING UP", "secondary")
    assert support_label(101) == ("STABLE", "primary")
    assert support_label(500) == ("STABLE", "primary")
    print("OK support-label tiers: <25 RAW/red, 25-100 WARMING/gold, >100 STABLE/green")


def test_markov_support_preserved():
    # Markov exposes its own `support`; attach must keep it, not overwrite.
    eng = MarkovEngine()
    hist = _history(140)
    preds = eng.predict_next(hist)
    native = {int(p["number"]): p.get("support") for p in preds if p.get("support") is not None}
    attach_confidence_intervals(eng, hist, preds, n_boot=40)
    for p in preds:
        if int(p["number"]) in native and native[int(p["number"])] is not None:
            assert p["support"] == native[int(p["number"])]
    print("OK Markov native support preserved")


if __name__ == "__main__":
    test_percentile_basics()
    test_bootstrap_ci_bounds_valid()
    test_bootstrap_is_deterministic_with_seed()
    test_attach_fills_missing_ci_and_support()
    test_bayesian_native_ci_untouched()
    test_support_label_tiers()
    test_markov_support_preserved()
    print("\nALL PROMPT 8 TESTS PASSED")
