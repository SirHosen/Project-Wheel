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
    test_chi_square()
    print("ALL CHECKS PASSED")
