# -*- coding: utf-8 -*-
"""A clean, well-documented LSTM next-result model + an HONEST walk-forward
backtest. This is the AI-training playground.

Architecture (small and readable):
    Embedding(9 -> EMBEDDING_DIM) -> LSTM(LSTM_UNITS) -> Dropout -> Dense(softmax)

The backtest is the most important part: it compares the model's top-1 accuracy
against a trivial "always guess the most frequent number so far" baseline, in a
walk-forward (train-on-past, test-on-next) fashion. On a FAIR wheel the model
will NOT beat the baseline -- that is the real lesson. On data with a genuine
pattern (see ai/dataset.synthetic_markov) it can, and you will see it.

TensorFlow is optional: if it is not installed, available() says so and the
trainer raises a clear, friendly error instead of crashing the whole app.
"""
import numpy as np

from config import (BATCH_SIZE, DROPOUT, EARLY_STOP_PATIENCE, EMBEDDING_DIM,
                    EPOCHS, LSTM_UNITS, MODEL_PATH, SEQUENCE_LENGTH,
                    VALIDATION_SPLIT)
from ai.dataset import NUM_CLASSES, from_indices, make_windows, to_indices

try:
    import tensorflow as tf
    _TF_ERR = None
except Exception as e:  # pragma: no cover - env dependent
    tf = None
    _TF_ERR = e


def available():
    return tf is not None


def status_line():
    if available():
        return f"tensorflow {tf.__version__} available"
    return (f"tensorflow NOT available ({type(_TF_ERR).__name__ if _TF_ERR else 'missing'})"
            "  -> pip install -r requirements-ai.txt")


def build_model(seq_len=SEQUENCE_LENGTH):
    """Build and compile the small LSTM classifier."""
    if not available():
        raise RuntimeError(status_line())
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(NUM_CLASSES, EMBEDDING_DIM),
        layers.LSTM(LSTM_UNITS),
        layers.Dropout(DROPOUT),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train(numbers, seq_len=SEQUENCE_LENGTH, epochs=EPOCHS, verbose=1,
          save_path=MODEL_PATH):
    """Train on a result history. Returns (model, history_dict)."""
    if not available():
        raise RuntimeError(status_line())
    X, y = make_windows(numbers, seq_len)
    if len(X) < 20:
        raise ValueError(f"need >=20 windows to train, got {len(X)} "
                         f"(feed more results; you have {len(numbers)})")
    from tensorflow.keras.callbacks import EarlyStopping
    model = build_model(seq_len)
    es = EarlyStopping(monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
                       restore_best_weights=True)
    hist = model.fit(X, y, validation_split=VALIDATION_SPLIT, epochs=epochs,
                     batch_size=BATCH_SIZE, callbacks=[es], verbose=verbose)
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
    return model, hist.history


def _most_frequent_so_far(idx_history):
    """Baseline prediction: the most common class seen so far."""
    counts = np.bincount(idx_history, minlength=NUM_CLASSES)
    return int(np.argmax(counts))


def walk_forward_backtest(numbers, seq_len=SEQUENCE_LENGTH, min_train=80,
                          step_retrain=40, epochs=8, verbose=0):
    """Honest walk-forward evaluation: train on the past, predict the next.

    Returns a dict with the model's top-1 accuracy, the most-frequent baseline
    accuracy, the lift, and the sample size. If the model does not beat the
    baseline by a clear margin, there is no real edge -- exactly what you should
    expect on a fair wheel.
    """
    if not available():
        raise RuntimeError(status_line())
    idx = to_indices(numbers)
    clean = from_indices(idx)  # numbers with invalid entries dropped, aligned to idx
    if len(idx) < min_train + seq_len + 20:
        raise ValueError("not enough data for a trustworthy backtest "
                         f"(have {len(idx)}, need >= {min_train + seq_len + 20})")

    model = None
    model_hits = base_hits = total = 0
    last_train_at = -10**9
    for t in range(min_train, len(idx) - 1):
        # Periodically retrain on everything seen so far (walk-forward).
        if model is None or (t - last_train_at) >= step_retrain:
            X, y = make_windows(clean[:t], seq_len)
            if len(X) >= 20:
                model = build_model(seq_len)
                from tensorflow.keras.callbacks import EarlyStopping
                model.fit(X, y, epochs=epochs, batch_size=BATCH_SIZE,
                          verbose=verbose,
                          callbacks=[EarlyStopping(monitor="loss", patience=3,
                                                   restore_best_weights=True)])
                last_train_at = t
        window = np.asarray([idx[t - seq_len:t]], dtype=np.int32)
        truth = idx[t]
        if model is not None:
            pred = int(np.argmax(model.predict(window, verbose=0)[0]))
            model_hits += int(pred == truth)
        base_hits += int(_most_frequent_so_far(np.asarray(idx[:t])) == truth)
        total += 1

    model_acc = model_hits / total if total else 0.0
    base_acc = base_hits / total if total else 0.0
    return {"n_eval": total, "model_acc": model_acc, "baseline_acc": base_acc,
            "lift": model_acc - base_acc,
            "verdict": ("model beats baseline (real sequential signal)"
                        if model_acc - base_acc > 0.02 else
                        "NO edge: model ~= baseline (expected on a fair wheel)")}
