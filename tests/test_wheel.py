# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Wheel math: design distribution, payouts, EV signs, chi-square sanity."""
from config import VALID_NUMBERS, payout_multiplier
from core import wheel


def test_design_distribution():
    dist = wheel.design_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    # 1 occupies the most segments, so it must be the most likely number.
    assert max(dist, key=dist.get) == 1
    print("OK design_distribution sums to 1, peak at 1")


def test_payout_and_breakeven():
    assert payout_multiplier(1) == 1 and payout_multiplier(40) == 40
    assert abs(wheel.breakeven_prob(1) - 0.5) < 1e-9
    assert abs(wheel.breakeven_prob(40) - 1.0 / 41.0) < 1e-9
    print("OK payout + break-even math")


def test_house_edge():
    # On the fair design wheel every bet is -EV (the house edge, by construction).
    evs = wheel.design_evs()
    assert all(ev < 0 for ev in evs.values()), evs
    print("OK every number is -EV on the fair wheel:", {n: round(evs[n], 3) for n in VALID_NUMBERS})


def test_incomplete_beta_and_quantile():
    # betai is the Beta(a, b) CDF: monotone in x, anchored at 0 and 1.
    assert abs(wheel.betai(2, 3, 0.0) - 0.0) < 1e-12
    assert abs(wheel.betai(2, 3, 1.0) - 1.0) < 1e-12
    # Beta(1, 1) is Uniform(0, 1): CDF(x) == x.
    for x in (0.1, 0.37, 0.9):
        assert abs(wheel.betai(1, 1, x) - x) < 1e-6
    # Symmetry: Beta(a, a) has median 0.5.
    assert abs(wheel.beta_quantile(4, 4, 0.5) - 0.5) < 1e-4
    # Quantile is a true inverse of the CDF (round-trip).
    for p in (0.025, 0.5, 0.975):
        q = wheel.beta_quantile(3, 7, p)
        assert abs(wheel.betai(3, 7, q) - p) < 1e-4
    # normal_cdf turns the 95% z-score into ~0.975 upper mass.
    assert abs(wheel.normal_cdf(1.96) - 0.975) < 1e-3
    print("OK incomplete beta CDF + Beta quantile + normal_cdf")


def test_chi_square():
    # Observed == expected -> chi2 ~ 0, p ~ 1.
    exp = {n: 10.0 for n in VALID_NUMBERS}
    obs = {n: 10 for n in VALID_NUMBERS}
    chi2, dof, p = wheel.chi_square_gof(obs, exp)
    assert chi2 < 1e-9 and p > 0.99
    # A wild deviation -> tiny p.
    obs2 = {n: 0 for n in VALID_NUMBERS}
    obs2[1] = 90
    chi2b, _, pb = wheel.chi_square_gof(obs2, {n: 10.0 for n in VALID_NUMBERS})
    assert chi2b > 50 and pb < 0.001
    print("OK chi-square: flat=high-p, skewed=low-p")


if __name__ == "__main__":
    test_design_distribution()
    test_payout_and_breakeven()
    test_house_edge()
    test_incomplete_beta_and_quantile()
    test_chi_square()
    print("ALL CHECKS PASSED")
