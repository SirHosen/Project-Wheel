# -*- coding: utf-8 -*-
"""PROMPT 11: LSTM longer-context + attention + feature engineering.

The pure-numpy feature/windowing/augmentation logic (predictors/lstm_features.py)
is tested unconditionally. The TensorFlow architecture (build / train 1 epoch /
predict shape) is tested only if TF is importable -- otherwise it is SKIPPED
with a clear message, so this suite still passes on a TF-less box and the heavy
check runs on the GPU machine.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from predictors import lstm_features as lf

_failures = []
_skipped = []


def check(name, cond):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")
        _failures.append(name)


# ---------------------------------------------------------------- numpy layer
def test_compute_features_shape_and_onehot():
    C = 4
    enc = [0, 1, 1, 2, 0, 3]
    feats = lf.compute_features(enc, C)
    check("compute_features shape = (n, 2C+1)", feats.shape == (len(enc), 2 * C + 1))
    # onehot block: row i, column = class at i, equals 1
    ok = all(feats[i, enc[i]] == 1.0 for i in range(len(enc)))
    check("onehot block marks the current class", ok)
    # trailing-5 freq block sums to ~1 each row (it is a normalized distribution)
    freq = feats[:, C:2 * C]
    sums = freq.sum(axis=1)
    check("trailing-5 freq rows normalize to ~1", np.allclose(sums, 1.0))


def test_time_since_last_same():
    C = 3
    enc = [0, 1, 0]  # class 0 reappears after gap of 2
    feats = lf.compute_features(enc, C)
    tsl_col = 2 * C
    # first occurrence of 0 -> distance i+1 = 1 -> 1/50
    check("first-seen time_since_last_same = 1/50", abs(feats[0, tsl_col] - 1 / 50) < 1e-6)
    # third element (idx2) is 0 again, last seen at idx0 -> gap 2 -> 2/50
    check("repeat time_since_last_same = 2/50", abs(feats[2, tsl_col] - 2 / 50) < 1e-6)


def test_make_windows_shapes_and_shift():
    C = 4
    seq_len = 3
    enc = [0, 1, 2, 3, 0, 1, 2]
    feats = lf.compute_features(enc, C)
    Xn, Xf, y = lf.make_windows(enc, feats, seq_len)
    n_expected = len(enc) - seq_len
    check("X_num shape (N, seq_len)", Xn.shape == (n_expected, seq_len))
    check("X_feat shape (N, seq_len, 2C+2)", Xf.shape == (n_expected, seq_len, 2 * C + 2))
    check("y shape (N,)", y.shape == (n_expected,))
    check("feature_width matches X_feat", lf.feature_width(C) == Xf.shape[2])
    # +1 shift: embedding ids in [1, C], never 0 (0 is reserved PAD)
    check("embedding ids shifted to [1, C]", Xn.min() >= 1 and Xn.max() <= C)
    # y stays in class space [0, C)
    check("labels stay in class space [0, C)", y.min() >= 0 and y.max() < C)
    # session_position last column goes 0..1 across each window
    pos = Xf[0, :, -1]
    check("session_position ramps 0->1", abs(pos[0]) < 1e-6 and abs(pos[-1] - 1.0) < 1e-6)


def test_make_windows_too_short():
    C = 4
    enc = [0, 1]
    feats = lf.compute_features(enc, C)
    Xn, Xf, y = lf.make_windows(enc, feats, 5)
    check("too-short sequence -> empty windows", len(Xn) == 0 and len(y) == 0)


def test_predict_window():
    C = 4
    seq_len = 3
    enc = [0, 1, 2, 3, 0]
    feats = lf.compute_features(enc, C)
    win = lf.make_predict_window(enc, feats, seq_len)
    check("predict window not None", win is not None)
    Xn, Xf = win
    check("predict X_num shape (1, seq_len)", Xn.shape == (1, seq_len))
    check("predict X_feat shape (1, seq_len, 2C+2)", Xf.shape == (1, seq_len, 2 * C + 2))
    check("predict uses the most recent window", list(Xn[0]) == [e + 1 for e in enc[-seq_len:]])


def test_augmentation():
    C = 4
    seq_len = 4
    enc = list(range(C)) * 6  # 24 elements
    feats = lf.compute_features(enc, C)
    Xn, Xf, y = lf.make_windows(enc, feats, seq_len)
    n0 = len(Xn)
    Xn2, Xf2, y2 = lf.augment_drop_random_prefix(Xn, Xf, y, drop_prob=1.0, seed=1)
    check("augmentation adds rows (originals preserved)", len(Xn2) > n0)
    check("augmented arrays stay aligned", len(Xn2) == len(Xf2) == len(y2))
    # at least one augmented row has a PAD (0) prefix
    aug_part = Xn2[n0:]
    check("augmented rows contain PAD (0) prefix", (aug_part == 0).any())
    # empty input is a no-op
    e_n, e_f, e_y = lf.augment_drop_random_prefix(
        np.zeros((0, seq_len), dtype="int32"),
        np.zeros((0, seq_len, 2 * C + 2), dtype="float32"),
        np.zeros((0,), dtype="int32"))
    check("augmentation no-op on empty input", len(e_n) == 0)


def test_settings_present():
    check("LSTM_SEQUENCE_LENGTH default >= 10", settings.LSTM_SEQUENCE_LENGTH >= 10)
    for nm in ("LSTM_USE_ATTENTION", "LSTM_ATTENTION_HEADS", "LSTM_ATTENTION_DIM",
               "LSTM_USE_FEATURES", "LSTM_AUGMENT"):
        check(f"setting {nm} present", hasattr(settings, nm))
    check("attention heads == 4", settings.LSTM_ATTENTION_HEADS == 4)
    check("attention dim == 32", settings.LSTM_ATTENTION_DIM == 32)


# ------------------------------------------------------------------- TF layer
def test_tf_model_if_available():
    try:
        import tensorflow  # noqa: F401
    except Exception as e:
        _skipped.append(f"TF build/train/predict (no tensorflow: {type(e).__name__})")
        return
    from predictors.tf_lstm_engine import TfLstmEngine

    # enough data for both attention and vanilla
    nums = [settings.VALID_NUMBERS[i % len(settings.VALID_NUMBERS)] for i in range(80)]

    for use_attn in (True, False):
        eng = TfLstmEngine(use_attention=use_attn)
        check(f"model builds (attention={use_attn})", eng.model is not None)
        ok = eng.train(nums, epochs=1, bulk=False)
        check(f"train 1 epoch returns True (attention={use_attn})", ok is True)
        preds = eng.predict_next(nums)
        check(f"predict returns one row per class (attention={use_attn})",
              len(preds) == eng.num_classes)
        s = sum(p["confidence"] for p in preds)
        check(f"predict confidences ~sum to 1 (attention={use_attn})", abs(s - 1.0) < 1e-3)


if __name__ == "__main__":
    print("== PROMPT 11: LSTM attention + features ==")
    test_compute_features_shape_and_onehot()
    test_time_since_last_same()
    test_make_windows_shapes_and_shift()
    test_make_windows_too_short()
    test_predict_window()
    test_augmentation()
    test_settings_present()
    test_tf_model_if_available()
    if _skipped:
        print("\nSKIPPED:")
        for s in _skipped:
            print(f"  [SKIP] {s}")
    if _failures:
        print(f"\nFAILED ({len(_failures)}): {_failures}")
        sys.exit(1)
    print("\nALL PASSED" + (" (TF layer skipped)" if _skipped else ""))
