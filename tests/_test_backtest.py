# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Test PROMPT 2: walk-forward backtest framework (core/backtest.py)."""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from core.backtest import WalkForwardBacktester
from predictors.markov_engine import MarkovEngine


def test_seeded_autocorrelation_detected_as_edge():
    cycle = [1, 2, 5]
    actuals = [cycle[i % 3] for i in range(240)]
    bt = WalkForwardBacktester(actuals, settings.VALID_NUMBERS, settings.SPINWHEEL_SEQUENCE)
    r = bt.backtest_engine(MarkovEngine(), warmup=40)
    assert r["verdict"] == "edge", r
    assert r["top1_acc"] > r["baseline_top1_acc"]
    assert r["lift"] > 0.3, r["lift"]
    assert r["brier_score"] is not None and r["log_loss"] is not None
    print(f"OK seeded pattern -> edge (top1={r['top1_acc']*100:.1f}% vs base {r['baseline_top1_acc']*100:.1f}%, z={r['z_score']:.2f})")


def test_pure_random_no_edge():
    rng = random.Random(42)
    actuals = [rng.choice(settings.VALID_NUMBERS) for _ in range(320)]
    bt = WalkForwardBacktester(actuals, settings.VALID_NUMBERS, settings.SPINWHEEL_SEQUENCE)
    r = bt.backtest_engine(MarkovEngine(), warmup=40)
    assert r["verdict"] != "edge", r
    print(f"OK pure random -> {r['verdict']} (top1={r['top1_acc']*100:.1f}% vs base {r['baseline_top1_acc']*100:.1f}%, z={r['z_score']:.2f})")


def test_insufficient_data():
    bt = WalkForwardBacktester([1, 2, 5] * 5, settings.VALID_NUMBERS, settings.SPINWHEEL_SEQUENCE)
    r = bt.backtest_engine(MarkovEngine(), warmup=40)
    assert r["verdict"] == "insufficient", r
    print("OK insufficient data handled")


def test_calibration_curve_shape():
    cycle = [1, 2, 5]
    actuals = [cycle[i % 3] for i in range(160)]
    bt = WalkForwardBacktester(actuals, settings.VALID_NUMBERS, settings.SPINWHEEL_SEQUENCE)
    curve = bt.calibration_curve(MarkovEngine(), warmup=40, bins=10)
    assert isinstance(curve, list) and curve
    for pt in curve:
        assert 0.0 <= pt["expected_acc"] <= 1.0
        assert 0.0 <= pt["actual_acc"] <= 1.0
    print(f"OK calibration curve ({len(curve)} bins)")


if __name__ == "__main__":
    test_seeded_autocorrelation_detected_as_edge()
    test_pure_random_no_edge()
    test_insufficient_data()
    test_calibration_curve_shape()
    print("\nALL PROMPT 2 TESTS PASSED")
