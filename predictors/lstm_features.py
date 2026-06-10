# -*- coding: utf-8 -*-
"""predictors/lstm_features.py - PURE-NUMPY feature engineering for the LSTM.

Deliberately free of any TensorFlow import so the windowing / feature logic can
be unit-tested anywhere (the GPU model in tf_lstm_engine.py is only importable
where TF is installed). All features are CAUSAL: timestep i uses only encoded
class indices up to and including i.

Class-index convention:
  * `encoded` holds class indices in [0, num_classes).
  * For the embedding input we shift by +1 so 0 can be reserved as PAD
    (used by augment_drop_random_prefix). So embedding ids live in
    [1, num_classes]; 0 == padding.

Per-timestep feature vector (width = 2*num_classes + 1), produced by
compute_features():
  [ last_number_onehot (num_classes),
    recent_5_window_freq (num_classes),
    time_since_last_same (1, normalized to /50 cap) ]

make_windows() appends ONE more column per timestep: the window-relative
session_position in [0, 1]. So the model's feature width is 2*num_classes + 2.
"""
import numpy as np

TSL_CAP = 50.0


def feature_width(num_classes: int) -> int:
    """Width of the per-timestep feature vector AS FED TO THE MODEL.

    = 2*num_classes (onehot + freq) + 1 (time_since_last_same)
      + 1 (session_position appended by make_windows).
    """
    return 2 * num_classes + 2


def compute_features(encoded, num_classes: int) -> np.ndarray:
    """Causal per-timestep features. Shape (len(encoded), 2*num_classes + 1)."""
    n = len(encoded)
    base = 2 * num_classes + 1
    feats = np.zeros((n, base), dtype="float32")
    last_pos = {}
    for i, c in enumerate(encoded):
        if 0 <= c < num_classes:
            feats[i, c] = 1.0  # last_number_onehot
        lo = max(0, i - 4)
        window = encoded[lo:i + 1]
        for v in window:
            if 0 <= v < num_classes:
                feats[i, num_classes + v] += 1.0
        wlen = len(window)
        if wlen > 0:
            feats[i, num_classes:2 * num_classes] /= float(wlen)  # recent_5 freq
        tsl = (i - last_pos[c]) if c in last_pos else (i + 1)  # time_since_last_same
        feats[i, 2 * num_classes] = min(float(tsl), TSL_CAP) / TSL_CAP
        last_pos[c] = i
    return feats


def _session_position_column(seq_len: int) -> np.ndarray:
    denom = max(1, seq_len - 1)
    return (np.arange(seq_len, dtype="float32") / denom).reshape(seq_len, 1)


def make_windows(encoded, features, seq_len: int):
    """Sliding windows for supervised training.

    Returns:
      X_num  (N, seq_len)         int32, EMBEDDING ids in [1, num_classes] (0=pad)
      X_feat (N, seq_len, F)      float32, F = features.shape[1] + 1
      y      (N,)                 int32, next-step CLASS index in [0, num_classes)
    """
    n = len(encoded)
    fcols = features.shape[1] + 1 if features.size or features.ndim == 2 else 1
    if n <= seq_len:
        return (np.zeros((0, seq_len), dtype="int32"),
                np.zeros((0, seq_len, fcols), dtype="float32"),
                np.zeros((0,), dtype="int32"))
    pos_col = _session_position_column(seq_len)
    Xn, Xf, y = [], [], []
    for i in range(n - seq_len):
        Xn.append([e + 1 for e in encoded[i:i + seq_len]])  # +1 shift, 0=pad
        fw = np.concatenate([features[i:i + seq_len], pos_col], axis=1)
        Xf.append(fw)
        y.append(encoded[i + seq_len])
    return (np.array(Xn, dtype="int32"),
            np.array(Xf, dtype="float32"),
            np.array(y, dtype="int32"))


def make_predict_window(encoded, features, seq_len: int):
    """Single most-recent window. Returns (X_num, X_feat) or None if too short."""
    if len(encoded) < seq_len:
        return None
    pos_col = _session_position_column(seq_len)
    xn = np.array([[e + 1 for e in encoded[-seq_len:]]], dtype="int32")
    fw = np.concatenate([features[-seq_len:], pos_col], axis=1)
    xf = fw.reshape(1, seq_len, features.shape[1] + 1).astype("float32")
    return xn, xf


def augment_drop_random_prefix(Xn, Xf, y, drop_prob=0.5, max_drop_frac=0.5,
                               seed=42, pad_idx=0):
    """Train-time regularization: for a random subset of windows, drop a random
    PREFIX (replace leading k steps with padding) so the model learns to cope
    with shorter effective context. Returns ORIGINAL + augmented rows.
    """
    if len(Xn) == 0:
        return Xn, Xf, y
    rng = np.random.default_rng(seed)
    seq_len = Xn.shape[1]
    max_drop = max(1, int(seq_len * max_drop_frac))
    aug_n, aug_f, aug_y = [], [], []
    for i in range(len(Xn)):
        if rng.random() < drop_prob:
            k = int(rng.integers(1, max_drop + 1))
            xn = Xn[i].copy()
            xf = Xf[i].copy()
            xn[:k] = pad_idx
            xf[:k] = 0.0
            aug_n.append(xn)
            aug_f.append(xf)
            aug_y.append(y[i])
    if not aug_n:
        return Xn, Xf, y
    Xn2 = np.concatenate([Xn, np.array(aug_n, dtype="int32")], axis=0)
    Xf2 = np.concatenate([Xf, np.array(aug_f, dtype="float32")], axis=0)
    y2 = np.concatenate([y, np.array(aug_y, dtype="int32")], axis=0)
    return Xn2, Xf2, y2
