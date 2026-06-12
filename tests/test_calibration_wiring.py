# -*- coding: utf-8 -*-
"""Audit V5 #1 & #2 regression tests.

Headless, no external deps: sklearn/scipy are ABSENT in CI, so isotonic fitting
falls back to the pure-Python PAVA path inside core/calibration.py. We do NOT
construct the full MainViewModel here (it needs TensorFlow); we test the two
pieces the wiring depends on directly:

  1. ReliabilityTracker: record -> brier/log_loss/ece -> fit_isotonic ->
     save/load round-trip, and that calibrate() is a safe no-op before a fit.
  2. Tracker top-1 HONEST accounting: only rounds graded LIVE
     (top1_graded_live) count toward top-1 accuracy; legacy/backfilled
     top1_hit must be ignored (no fake ~100%).
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import ReliabilityTracker
from data.tracker import Tracker


def test_reliability_record_and_metrics():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "calibration_state.json")
        rt = ReliabilityTracker(path=path)
        # Mix of correct/incorrect across confidences for engine "E".
        for i in range(40):
            conf = 0.2 + (i % 5) * 0.15  # spread 0.2..0.8
            was_correct = (i % 3 == 0)
            rt.record(conf, was_correct, engine="E")
        s = rt.summary("E")
        assert s["n"] == 40, s
        # Brier in [0,1]; ECE in [0,1]; log_loss finite & positive.
        assert 0.0 <= s["brier"] <= 1.0, s["brier"]
        assert 0.0 <= s["ece"] <= 1.0, s["ece"]
        assert s["log_loss"] is not None and s["log_loss"] > 0.0, s["log_loss"]
        assert isinstance(s["bins"], list) and s["bins"], s["bins"]
        for b in s["bins"]:
            assert 0.0 <= b["mean_predicted"] <= 1.0
            assert 0.0 <= b["observed_acc"] <= 1.0
            assert b["n"] >= 1
        print("  reliability metrics OK (brier=%.4f ece=%.4f)" % (s["brier"], s["ece"]))


def test_isotonic_fit_and_persist():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "calibration_state.json")
        rt = ReliabilityTracker(path=path)
        # Monotone-ish signal: higher confidence -> more likely correct.
        import random
        rng = random.Random(7)
        for _ in range(200):
            conf = rng.random()
            was_correct = rng.random() < conf
            rt.record(conf, was_correct, engine="E")
        ok = rt.fit_isotonic("E")
        assert ok is True, "fit_isotonic should succeed with >=10 pairs (PAVA fallback)"
        # Calibrated value stays a valid probability.
        c = rt.calibrate(0.5, "E")
        assert 0.0 <= c <= 1.0, c
        rt.save()
        assert os.path.exists(path), "calibration_state.json must be written"
        with open(path) as f:
            blob = json.load(f)
        assert "pairs" in blob or "engine_pairs" in blob, blob.keys()
        # Round-trip load into a fresh tracker.
        rt2 = ReliabilityTracker(path=path)
        rt2.load()
        assert rt2.summary("E")["n"] == 200, rt2.summary("E")
        print("  isotonic fit + save/load round-trip OK")


def test_calibrate_noop_before_fit():
    with tempfile.TemporaryDirectory() as d:
        rt = ReliabilityTracker(path=os.path.join(d, "calibration_state.json"))
        # No isotonic fit yet -> calibrate returns input unchanged (safe no-op).
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert rt.calibrate(p, "E") == p, (p, rt.calibrate(p, "E"))
        print("  calibrate() no-op before fit OK")


def _isolated_tracker(d):
    # Point the tracker at an ISOLATED temp history file so we never touch the
    # real data/history.json (or its sibling SQLite db). A fresh temp dir gives
    # an empty data set (capital 1000, no history) - no reset needed.
    return Tracker(history_file=os.path.join(d, "history.json"))


def test_top1_only_counts_live_grades():
    with tempfile.TemporaryDirectory() as d:
        t = _isolated_tracker(d)
        # Simulate the LIVE path: record_result called every round with top1_hit
        # (win OR loss). Two correct top-1, one wrong -> 2/3.
        snap = [{"number": 5, "confidence": 0.4, "token_bet": 1}]
        t.record_result(5, 5, 4, bet_snapshot=snap, engine_used="E", top1_hit=True)
        t.record_result(2, 5, -1, bet_snapshot=snap, engine_used="E", top1_hit=False)
        t.record_result(5, 5, 4, bet_snapshot=snap, engine_used="E", top1_hit=True)
        stats = t.get_stats()
        assert stats["top1_graded"] == 3, stats
        assert abs(stats["top1_accuracy"] - (2 / 3 * 100)) < 1e-6, stats
        print("  live top-1 grading counts every round OK (%.1f%%)" % stats["top1_accuracy"])


def test_legacy_backfilled_top1_is_ignored():
    with tempfile.TemporaryDirectory() as d:
        t = _isolated_tracker(d)
        # Inject a LEGACY-style record with a biased backfilled top1_hit but
        # WITHOUT top1_graded_live (how old migrate_stats left them). It must
        # NOT count toward top-1 accuracy.
        t.data["history"].append({
            "actual_number": 5,
            "predicted_number": 5,
            "top1_hit": True,  # biased backfill: stored only on wins
            # note: no top1_graded_live
        })
        stats = t.get_stats()
        assert stats["top1_graded"] == 0, stats
        assert stats["top1_accuracy"] is None, stats
        print("  legacy biased top1_hit correctly ignored OK")


def main():
    test_reliability_record_and_metrics()
    test_isotonic_fit_and_persist()
    test_calibrate_noop_before_fit()
    test_top1_only_counts_live_grades()
    test_legacy_backfilled_top1_is_ignored()
    print("ALL CALIBRATION WIRING TESTS PASSED")


if __name__ == "__main__":
    main()
