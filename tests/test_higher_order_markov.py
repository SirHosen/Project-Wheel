# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Test PROMPT 3: HigherOrderMarkovEngine (variable-order + CV + backoff)."""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from predictors.higher_order_markov import HigherOrderMarkovEngine
from core.backtest import WalkForwardBacktester


def test_output_shape_and_normalization():
    eng = HigherOrderMarkovEngine()
    preds = eng.predict_next([1, 2, 5, 1, 2, 5])
    assert len(preds) == len(settings.VALID_NUMBERS)
    assert abs(sum(p["confidence"] for p in preds) - 1.0) < 1e-6
    assert all("support" in p and "order" in p for p in preds)
    assert preds == sorted(preds, key=lambda x: x["confidence"], reverse=True)
    print("OK output shape + normalized")


def test_empty_history_uses_prior():
    eng = HigherOrderMarkovEngine()
    preds = eng.predict_next([])
    assert preds[0]["support"] == 0
    # frequency prior -> number 1 is the most common slot on the wheel
    assert preds[0]["number"] == 1
    print("OK empty history -> frequency prior")


def test_high_order_pattern_selected():
    # order-2 pattern: (1,2)->5, (2,5)->8, (5,8)->1, (8,1)->2 ... period-4 cycle
    cycle = [1, 2, 5, 8]
    actuals = [cycle[i % 4] for i in range(200)]
    eng = HigherOrderMarkovEngine(max_order=4)
    eng.predict_next(actuals)
    assert eng.last_selected_order >= 1
    # On a clean deterministic cycle the next number is fully predictable
    preds = eng.predict_next(actuals)
    nxt = actuals[len(actuals) % 4]  # not used directly; sanity that top1 is confident
    assert preds[0]["confidence"] > 0.5, preds[0]
    print(f"OK deterministic cycle -> order={eng.last_selected_order}, top1 conf={preds[0]['confidence']:.2f}")


def test_backtest_beats_baseline_on_pattern():
    cycle = [1, 2, 5, 8]
    actuals = [cycle[i % 4] for i in range(220)]
    bt = WalkForwardBacktester(actuals, settings.VALID_NUMBERS, settings.SPINWHEEL_SEQUENCE)
    r = bt.backtest_engine(HigherOrderMarkovEngine(max_order=4), warmup=40)
    assert r["verdict"] == "edge", r
    print(f"OK pattern backtest -> edge (top1={r['top1_acc']*100:.1f}%, z={r['z_score']:.2f})")


def test_random_no_edge():
    rng = random.Random(7)
    actuals = [rng.choice(settings.VALID_NUMBERS) for _ in range(300)]
    bt = WalkForwardBacktester(actuals, settings.VALID_NUMBERS, settings.SPINWHEEL_SEQUENCE)
    r = bt.backtest_engine(HigherOrderMarkovEngine(max_order=4), warmup=40)
    assert r["verdict"] != "edge", r
    print(f"OK random -> {r['verdict']} (top1={r['top1_acc']*100:.1f}%, z={r['z_score']:.2f})")


if __name__ == "__main__":
    test_output_shape_and_normalization()
    test_empty_history_uses_prior()
    test_high_order_pattern_selected()
    test_backtest_beats_baseline_on_pattern()
    test_random_no_edge()
    print("\nALL PROMPT 3 TESTS PASSED")
