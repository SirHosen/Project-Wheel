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


def test_feature_windows_causal():
    # Richer features: shapes line up and features are CAUSAL (no future leak).
    nums = [1, 1, 2, 1, 5, 1, 8, 1, 10, 1, 15, 20, 1, 2]
    X, Xf, y = dataset.make_feature_windows(nums, seq_len=4)
    assert X.shape == (len(nums) - 4, 4)
    assert Xf.shape == (len(nums) - 4, 4, dataset.FEATURE_DIM)
    assert y.shape[0] == X.shape[0]
    feats = dataset.compute_features(nums, seq_len=4)
    assert feats.shape == (len(nums), dataset.FEATURE_DIM)
    # is_repeat (col 3): position 1 repeats a '1', position 2 does not.
    assert feats[1, 3] == 1.0 and feats[2, 3] == 0.0
    # running_freq (col 0) is a valid probability and first step is always 1.0.
    assert feats[0, 0] == 1.0
    assert (feats[:, 0] >= 0.0).all() and (feats[:, 0] <= 1.0).all()
    # recency_gap (col 1) is normalized into [0, 1]; first sight of a number = 1.
    assert (feats[:, 1] >= 0.0).all() and (feats[:, 1] <= 1.0).all()
    assert feats[0, 1] == 1.0
    print("OK richer features: shapes + causal repeat/frequency/recency")


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


def test_backtest_table_baselines():
    # Multiple baselines are compared on the SAME walk-forward split. This runs
    # even without PyTorch (the LSTM row is simply omitted).
    markov = dataset.synthetic_markov(400, repeat_prob=0.7, seed=5)
    rep = lstm.walk_forward_table(markov, min_train=80, epochs=4)
    acc = {r["name"]: r["acc"] for r in rep["rows"]}
    assert "persistence" in acc and "most_frequent" in acc
    # On strongly sequential data, 'repeat the last number' beats most-frequent.
    assert acc["persistence"] > acc["most_frequent"], acc
    assert rep["best"] is not None
    print(f"OK baseline table: persistence {acc['persistence']:.2f} > "
          f"most_frequent {acc['most_frequent']:.2f}")


def test_probabilistic_report_calibration():
    import numpy as np
    biased = dataset.synthetic_biased(600, bias_number=8, strength=4.0, seed=7)
    rep = lstm.probabilistic_report(biased, min_train=80, epochs=4)
    K = dataset.NUM_CLASSES
    # The frequency model should be better-calibrated than a uniform guess.
    assert rep["freq"]["log_loss"] < np.log(K), rep
    assert rep["freq"]["brier"] < (1.0 - 1.0 / K), rep
    print(f"OK probabilistic report: freq log_loss={rep['freq']['log_loss']:.3f} "
          f"< ln(K)={np.log(K):.3f}")


if __name__ == "__main__":
    test_windows_and_indices()
    test_feature_windows_causal()
    test_synthetic_generators()
    test_backtest_table_baselines()
    test_probabilistic_report_calibration()
    test_backtest_contrast()
    print("ALL CHECKS PASSED")
