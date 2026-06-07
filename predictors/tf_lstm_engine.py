# -*- coding: utf-8 -*-
"""
predictors/tf_lstm_engine.py - Deep LSTM sequence predictor (TensorFlow/Keras).

Upgraded to make full use of a real GPU (e.g. RTX 3080 / Ampere):
  * Categorical EMBEDDING input instead of crude /N normalization.
  * Deeper, configurable stacked-LSTM architecture.
  * Optional mixed-precision (float16) -> big speedup on Tensor Cores.
  * Proper bulk training with validation split + early stopping.
  * Model persistence (train once on the GPU, reuse across launches).
  * Honest walk-forward backtest vs the 'always most frequent' baseline.

The public interface is unchanged and backward compatible:
  - train(history_list, epochs=1, bulk=False) -> bool
  - predict_next(history: list) -> [{"number": int, "confidence": float}, ...]
"""
import os
import logging

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

from .base import BasePredictor
from config import settings


def _cfg(name, default):
    """Read an optional setting, falling back to a default if undefined."""
    return getattr(settings, name, default)


class TfLstmEngine(BasePredictor):
    """LSTM-based predictor for sequence learning, GPU-accelerated."""

    def __init__(self, sequence_length=settings.LSTM_SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.valid_numbers = list(settings.VALID_NUMBERS)
        self.num_classes = len(self.valid_numbers)

        # Stable number <-> class-index mapping (no sklearn dependency).
        self._num_to_idx = {n: i for i, n in enumerate(self.valid_numbers)}
        self._idx_to_num = {i: n for n, i in self._num_to_idx.items()}

        self._mixed_precision = False
        self._maybe_enable_mixed_precision()

        self.model = self._build_model()
        self.history_metrics = {"loss": [], "accuracy": [], "val_accuracy": []}
        self._trained = False

        # Reuse a previously GPU-trained model if one was saved before.
        self.try_load()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _maybe_enable_mixed_precision(self):
        """Enable float16 compute on GPUs with Tensor Cores (RTX 3080 = Ampere).
        Falls back silently to full precision on CPU or older cards."""
        if not _cfg("LSTM_USE_MIXED_PRECISION", True):
            return
        try:
            if tf.config.list_physical_devices("GPU"):
                mixed_precision.set_global_policy("mixed_float16")
                self._mixed_precision = True
                logging.info("TF-LSTM: mixed_float16 enabled (GPU Tensor Cores).")
        except Exception as e:
            logging.warning(f"TF-LSTM: mixed precision unavailable: {e}")

    def _build_model(self):
        emb_dim = int(_cfg("LSTM_EMBEDDING_DIM", 16))
        units = list(_cfg("LSTM_UNITS", [128, 64]))
        dense_units = int(_cfg("LSTM_DENSE_UNITS", 64))
        dropout = float(_cfg("LSTM_DROPOUT", 0.2))

        layers = [
            Input(shape=(self.sequence_length,), dtype="int32"),
            Embedding(input_dim=self.num_classes, output_dim=emb_dim),
        ]
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            layers.append(LSTM(int(u), return_sequences=return_seq))
            layers.append(Dropout(dropout))
        layers.append(Dense(dense_units, activation="relu"))
        # Keep the softmax in float32 even under mixed precision (stability).
        layers.append(Dense(self.num_classes, activation="softmax", dtype="float32"))

        model = Sequential(layers)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode(self, numbers):
        return [self._num_to_idx[n] for n in numbers if n in self._num_to_idx]

    def prepare_data(self, sequence_list):
        encoded = self._encode(sequence_list)
        if len(encoded) <= self.sequence_length:
            return np.array([]), np.array([])
        X, y = [], []
        for i in range(len(encoded) - self.sequence_length):
            X.append(encoded[i:i + self.sequence_length])
            y.append(encoded[i + self.sequence_length])
        return np.array(X, dtype="int32"), np.array(y, dtype="int32")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, history_list, epochs=1, bulk=False):
        """Train incrementally (default) or do a heavy bulk train.

        bulk=True uses many epochs + a validation split + early stopping,
        which is where a fast GPU really pays off."""
        X, y = self.prepare_data(history_list)
        if X is None or len(X) == 0:
            return False

        if bulk:
            epochs = int(_cfg("LSTM_BULK_EPOCHS", 60))
            batch_size = int(_cfg("LSTM_BATCH_SIZE", 64))
            val_split = float(_cfg("LSTM_VALIDATION_SPLIT", 0.15))
            patience = int(_cfg("LSTM_EARLY_STOP_PATIENCE", 8))
            use_val = len(X) >= 40
            callbacks = []
            if use_val:
                callbacks.append(
                    EarlyStopping(
                        monitor="val_accuracy",
                        patience=patience,
                        restore_best_weights=True,
                    )
                )
            hist = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=min(batch_size, len(X)),
                validation_split=val_split if use_val else 0.0,
                callbacks=callbacks,
                verbose=0,
            )
        else:
            hist = self.model.fit(X, y, epochs=epochs, verbose=0)

        self._record_metrics(hist)
        self._trained = True
        return True

    def _record_metrics(self, hist):
        h = getattr(hist, "history", {}) or {}
        for k in ("loss", "accuracy", "val_accuracy"):
            if k in h:
                self.history_metrics.setdefault(k, []).extend(h[k])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_next(self, history):
        encoded = self._encode(history)
        if len(encoded) < self.sequence_length:
            prob = 1.0 / self.num_classes
            return [{"number": n, "confidence": prob} for n in self.valid_numbers]

        seq = np.array([encoded[-self.sequence_length:]], dtype="int32")
        probs = self.model.predict(seq, verbose=0)[0]

        predictions = [
            {"number": self._idx_to_num[i], "confidence": float(p)}
            for i, p in enumerate(probs)
        ]
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions

    # ------------------------------------------------------------------
    # Honest evaluation (the real point of a powerful model)
    # ------------------------------------------------------------------
    def evaluate_backtest(self, history_list, min_train=40):
        """Walk-forward test: train on the past, predict the next, and compare
        the model's top-1 accuracy with the naive 'most frequent so far'
        baseline. This is the honest 'is the GPU model worth it?' check.

        Returns a dict with model_acc, baseline_acc, lift, verdict."""
        encoded = self._encode(history_list)
        n = len(encoded)
        result = {
            "n_eval": 0, "model_acc": None, "baseline_acc": None,
            "lift": None, "verdict": "insufficient",
        }
        if n < min_train + self.sequence_length + 1:
            return result

        split = max(min_train, int(n * 0.7))
        train_nums = [self._idx_to_num[i] for i in encoded[:split]]
        self.train(train_nums, bulk=True)

        correct_model = correct_base = total = 0
        for i in range(split, n):
            window = [self._idx_to_num[x]
                      for x in encoded[i - self.sequence_length:i]]
            preds = self.predict_next(window)
            top = preds[0]["number"] if preds else None
            actual = self._idx_to_num[encoded[i]]
            past = encoded[:i]
            base = self._idx_to_num[max(set(past), key=past.count)]
            correct_model += int(top == actual)
            correct_base += int(base == actual)
            total += 1

        if total == 0:
            return result
        m = correct_model / total
        b = correct_base / total
        result.update({
            "n_eval": total,
            "model_acc": m,
            "baseline_acc": b,
            "lift": m - b,
            "verdict": "edge" if m > b else "no_edge",
        })
        return result

    # ------------------------------------------------------------------
    # Persistence (train once on the GPU, reuse forever)
    # ------------------------------------------------------------------
    def _model_path(self):
        path = _cfg("LSTM_MODEL_PATH", "models/lstm_spinwheel.keras")
        if not os.path.isabs(path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        return path

    def save(self):
        try:
            path = self._model_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            return True
        except Exception as e:
            logging.warning(f"TF-LSTM: could not save model: {e}")
            return False

    def try_load(self):
        try:
            path = self._model_path()
            if os.path.exists(path):
                self.model = load_model(path)
                self._trained = True
                logging.info(f"TF-LSTM: loaded saved model from {path}")
                return True
        except Exception as e:
            logging.warning(f"TF-LSTM: could not load saved model: {e}")
        return False
