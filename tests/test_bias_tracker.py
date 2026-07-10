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


def test_ev_samples_uncertainty():
    # Fully Bayesian EV via Dirichlet sampling reports P(EV>0), not just a bound.
    t = OnlineBiasTracker(min_obs=25)
    t.observe_many([40] * 80)  # wheel hammered toward 40
    ev = t.ev_samples(n_samples=3000, seed=1)
    assert set(ev) == set(t.valid)
    for row in ev.values():
        assert row["ev_lo"] <= row["ev_mean"] <= row["ev_hi"]
        assert 0.0 <= row["prob_positive"] <= 1.0
    # Betting 40 is now almost surely +EV; a rarely-seen number is not.
    assert ev[40]["prob_positive"] > 0.95, ev[40]
    assert ev[2]["prob_positive"] < 0.5, ev[2]
    print("OK ev_samples: P(EV>0) high for the biased number, low otherwise")


def test_multiple_testing_correction():
    # Correcting for testing all 9 numbers at once makes the per-number tail
    # STRICTER than the raw family alpha.
    none = OnlineBiasTracker(min_obs=25, mt_correction="none")
    sidak = OnlineBiasTracker(min_obs=25, mt_correction="sidak")
    bonf = OnlineBiasTracker(min_obs=25, mt_correction="bonferroni")
    assert sidak._edge_tail() < none._edge_tail()
    assert bonf._edge_tail() < none._edge_tail()
    # On mildly-suggestive real data, no edge survives the correction.
    sidak.observe_many([30, 5, 20, 30, 1, 30, 8, 40, 15, 1, 10, 40, 30])
    assert sidak.best_bet() is None
    print("OK multiple-testing correction tightens the per-number edge tail")


def test_detects_real_bias():
    t = OnlineBiasTracker(min_obs=25)
    t.observe_many([40] * 60)  # absurd bias toward 40
    assert t.bias_test()["biased"] is True
    bb = t.best_bet()
    assert bb is not None and bb["number"] == 40
    print("OK detects a genuine bias and flags 40 as +EV")


if __name__ == "__main__":
    test_prior_and_gating()
    test_ev_samples_uncertainty()
    test_multiple_testing_correction()
    test_detects_real_bias()
    print("ALL CHECKS PASSED")
