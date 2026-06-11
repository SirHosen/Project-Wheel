# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Test PROMPT 4: ReliabilityTracker + isotonic calibration."""
import os
import sys
import random
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.calibration import ReliabilityTracker, _pava


def test_pava_monotone():
    blocks = _pava([0.0, 1.0, 0.0, 1.0], [1, 1, 1, 1])
    vals = [b[2] for b in blocks]
    assert vals == sorted(vals), vals
    print("OK PAVA produces non-decreasing fit")


def test_brier_logloss_basic():
    t = ReliabilityTracker(path=os.path.join(tempfile.gettempdir(), "_cal_test.json"))
    t.pairs = []
    # perfectly calibrated-ish: p=1 correct, p=0 wrong
    t.record(1.0, True)
    t.record(0.0, False)
    assert t.brier() < 1e-6
    assert t.log_loss() < 0.01
    print("OK brier/log-loss compute")


def test_isotonic_fixes_overconfidence():
    path = os.path.join(tempfile.gettempdir(), "_cal_test2.json")
    if os.path.exists(path):
        os.remove(path)
    t = ReliabilityTracker(path=path)
    rng = random.Random(1)
    # Engine says 0.8 but is only right ~40% of the time (overconfident).
    for _ in range(400):
        t.record(0.8, rng.random() < 0.40)
        t.record(0.2, rng.random() < 0.10)
    assert t.fit_isotonic()
    cal_hi = t.calibrate(0.8)
    cal_lo = t.calibrate(0.2)
    assert cal_hi < 0.8, cal_hi  # deflated toward reality
    assert cal_lo <= cal_hi
    assert 0.30 < cal_hi < 0.55, cal_hi
    print(f"OK isotonic deflates 0.8 -> {cal_hi:.2f}, 0.2 -> {cal_lo:.2f}")


def test_persistence_roundtrip():
    path = os.path.join(tempfile.gettempdir(), "_cal_test3.json")
    if os.path.exists(path):
        os.remove(path)
    t = ReliabilityTracker(path=path)
    for i in range(50):
        t.record(0.5, i % 2 == 0)
    t.fit_isotonic()
    assert t.save()
    t2 = ReliabilityTracker(path=path)
    assert len(t2.pairs) == 50
    print("OK persistence round-trip")
    os.remove(path)


def test_calibrate_predictions_renormalizes():
    t = ReliabilityTracker(path=os.path.join(tempfile.gettempdir(), "_cal_test4.json"))
    preds = [{"number": 1, "confidence": 0.6}, {"number": 2, "confidence": 0.4}]
    out = t.calibrate_predictions(preds)
    assert abs(sum(p["confidence"] for p in out) - 1.0) < 1e-6
    assert all("confidence_raw" in p for p in out)
    print("OK calibrate_predictions renormalizes")


if __name__ == "__main__":
    test_pava_monotone()
    test_brier_logloss_basic()
    test_isotonic_fixes_overconfidence()
    test_persistence_roundtrip()
    test_calibrate_predictions_renormalizes()
    print("\nALL PROMPT 4 TESTS PASSED")
