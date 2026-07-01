# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Bayesian bias tracker: prior sanity, honest gating, real-bias detection."""
from core.bias_tracker import OnlineBiasTracker


def test_prior_and_gating():
    t = OnlineBiasTracker(min_obs=25)
    post = t.posterior()
    assert abs(sum(p["mean"] for p in post.values()) - 1.0) < 1e-6
    # The 13 real observed results from the two videos.
    t.observe_many([30, 5, 20, 30, 1, 30, 8, 40, 15, 1, 10, 40, 30])
    assert t.n == 13
    # Below min_obs we must NOT claim bias or any +EV edge.
    assert t.bias_test()["biased"] is False
    assert t.best_bet() is None
    assert all(not e["is_edge"] for e in t.edges())
    print("OK prior sums to 1; no edge/bias claimed below min_obs")


def test_detects_real_bias():
    t = OnlineBiasTracker(min_obs=25)
    t.observe_many([40] * 60)  # absurd bias toward 40
    assert t.bias_test()["biased"] is True
    bb = t.best_bet()
    assert bb is not None and bb["number"] == 40
    print("OK detects a genuine bias and flags 40 as +EV")


if __name__ == "__main__":
    test_prior_and_gating()
    test_detects_real_bias()
    print("ALL CHECKS PASSED")
