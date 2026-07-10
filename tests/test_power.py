# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Power/sample-size simulation: valid pmf + power grows with more data."""
from core import power


def test_biased_distribution_sums_to_one():
    probs = power.biased_distribution(40, 0.30)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert abs(probs[40] - 0.30) < 1e-9
    assert all(p >= 0 for p in probs.values())
    print("OK biased distribution is a valid pmf with the target boost")


def test_more_data_gives_more_power():
    low = power.simulate_power(40, 0.30, n=25, trials=120, seed=1)
    high = power.simulate_power(40, 0.30, n=300, trials=120, seed=1)
    assert high["bias_power"] >= low["bias_power"]
    assert 0.0 <= low["bias_power"] <= 1.0
    # A big bias over 300 spins should be very detectable.
    assert high["bias_power"] > 0.8, high
    print(f"OK power grows with n: {low['bias_power']:.2f} -> {high['bias_power']:.2f}")


def test_sample_size_for_power():
    n, p = power.sample_size_for_power(40, 0.30, target_power=0.8,
                                       ns=[50, 150, 400], trials=120, seed=2)
    assert n is not None and p >= 0.8
    print(f"OK reach 0.8 bias power at n={n} (power={p:.2f})")


if __name__ == "__main__":
    test_biased_distribution_sums_to_one()
    test_more_data_gives_more_power()
    test_sample_size_for_power()
    print("ALL CHECKS PASSED")
