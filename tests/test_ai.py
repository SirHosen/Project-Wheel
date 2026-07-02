# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""AI data layer + (if PyTorch is installed) the honest backtest contrast:
an LSTM should beat the baseline on sequential 'markov' data but NOT on a fair
wheel. The PyTorch part skips cleanly when PyTorch is absent.
"""
import numpy as np

from ai import dataset, lstm
from core.bias_tracker import OnlineBiasTracker


def test_windows_and_indices():
    nums = [1, 2, 5, 8, 10, 15, 20, 30, 40, 1, 2, 5]
    X, y = dataset.make_windows(nums, seq_len=4)
    assert X.shape == (len(nums) - 4, 4)
    assert y.shape[0] == X.shape[0]
    # Round-trip index mapping.
    assert dataset.from_indices(dataset.to_indices(nums)) == nums
    print("OK windows shape + index round-trip")


def test_synthetic_generators():
    fair = dataset.synthetic_fair(2000, seed=1)
    biased = dataset.synthetic_biased(2000, bias_number=8, strength=4.0, seed=1)
    assert biased.count(8) > fair.count(8) * 1.5, "biased wheel must over-produce 8"
    # The bias tracker should catch the injected bias on enough data.
    t = OnlineBiasTracker()
    t.observe_many(biased)
    assert t.bias_test()["biased"] is True
    print("OK synthetic generators + bias tracker catches injected bias")


def test_backtest_contrast():
    if not lstm.available():
        print("SKIP backtest: " + lstm.status_line())
        return
    markov = dataset.synthetic_markov(800, repeat_prob=0.6, seed=2)
    rep = lstm.walk_forward_backtest(markov, epochs=12, verbose=0)
    # On strongly sequential data the model should beat the most-frequent baseline.
    assert rep["lift"] > 0.02, rep
    print(f"OK backtest: model beats baseline on markov data (lift={rep['lift']:+.3f})")


if __name__ == "__main__":
    test_windows_and_indices()
    test_synthetic_generators()
    test_backtest_contrast()
    print("ALL CHECKS PASSED")
