# -*- coding: utf-8 -*-
"""
predictors/tf_lstm_engine.py - Deep LSTM sequence predictor (TensorFlow/Keras).

v1.18.0 upgrade (PROMPT 11): longer context + attention + feature engineering.
  * Default context window LSTM_SEQUENCE_LENGTH = 10 (was 5).
  * Optional self-attention (MultiHeadAttention) between two LSTM stacks.
  * Per-timestep engineered features (onehot, trailing-5 freq, time-since-same,
    session position) fed ALONGSIDE the learned number embedding.
  * Train-time augmentation (drop-random-prefix) for regularization.
  * Categorical EMBEDDING input (0 reserved as PAD), optional mixed precision,
    bulk training w/ validation split + early stopping, model persistence,
    honest walk-forward backtest vs the 'always most frequent' baseline.

Backward compatibility:
  * If LSTM_USE_ATTENTION is False the attention block is skipped (plain
    stacked-LSTM over embedding[+features]).
  * Public interface is unchanged:
      - train(history_list, epochs=1, bulk=False) -> bool
      - predict_next(history: list) -> [{"number": int, "confidence": float}, ...]
      - evaluate_backtest(history_list) -> dict
      - save() / try_load()
      - history_metrics dict, _trained flag
  * NOTE: the feature/seq-len change makes models saved by <=v1.17.0
    incompatible; try_load() detects the input-shape mismatch and ignores the
    stale file so a fresh model is trained.
"""
import os
import logging

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dropout, Dense, Concatenate,
    MultiHeadAttention, Add, LayerNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping

from .base import BasePredictor
from . import lstm_features as lf
from config import settings


def _cfg(name, default):
    """Read an optional setting, falling back to a default if undefined."""
    return getattr(settings, name, default)


class TfLstmEngine(BasePredictor):
    """LSTM-based predictor for sequence learning, GPU-accelerated."""

    def __init__(self, sequence_length=None, use_attention=None, use_features=None):
        self.sequence_length = int(
            sequence_length if sequence_length is not None
            else _cfg("LSTM_SEQUENCE_LENGTH", 10)
        )
        self.use_attention = bool(
            use_attention if use_attention is not None
            else _cfg("LSTM_USE_ATTENTION", True)
        )
        self.use_features = bool(
            use_features if use_features is not None
            else _cfg("LSTM_USE_FEATURES", True)
        )
        self.valid_numbers = list(settings.VALID_NUMBERS)
        self.num_classes = len(self.valid_numbers)
        self.feat_width = lf.feature_width(self.num_classes)

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
        """Enable float16 compute on GPUs with Tensor Cores (Ampere/RTX 30xx).
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
        heads = int(_cfg("LSTM_ATTENTION_HEADS", 4))
        key_dim = int(_cfg("LSTM_ATTENTION_DIM", 32))

        # Embedding ids are shifted +1 (0 reserved as PAD) -> input_dim+1.
        num_in = Input(shape=(self.sequence_length,), dtype="int32", name="num")
        x = Embedding(input_dim=self.num_classes + 1, output_dim=emb_dim,
                      name="emb")(num_in)

        inputs = [num_in]
        if self.use_features:
            feat_in = Input(shape=(self.sequence_length, self.feat_width),
                            dtype="float32", name="feat")
            inputs.append(feat_in)
            x = Concatenate(axis=-1, name="emb_feat")([x, feat_in])

        # First LSTM always returns sequences (needed for attention + 2nd LSTM).
        u0 = int(units[0]) if units else 128
        u1 = int(units[1]) if len(units) > 1 else 64
        x = LSTM(u0, return_sequences=True, name="lstm_1")(x)
        x = Dropout(dropout)(x)

        if self.use_attention:
            attn = MultiHeadAttention(num_heads=heads, key_dim=key_dim,
                                      name="mha")(x, x)
            x = Add(name="attn_residual")([x, attn])
            x = LayerNormalization(name="attn_norm")(x)

        x = LSTM(u1, name="lstm_2")(x)
        x = Dropout(dropout)(x)
        x = Dense(dense_units, activation="relu", name="dense")(x)
        # Keep the softmax in float32 even under mixed precision (stability).
        out = Dense(self.num_classes, activation="softmax", dtype="float32",
                    name="out")(x)

        model = Model(inputs=inputs if len(inputs) > 1 else inputs[0],
                      outputs=out)
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    # ------------------------------------------------------------------
    # Encoding / data prep
    # ------------------------------------------------------------------
    def _encode(self, numbers):
        return [self._num_to_idx[n] for n in numbers if n in self._num_to_idx]

    def _model_inputs(self, Xn, Xf):
        """Match the model's input arity (features optional)."""
        return [Xn, Xf] if self.use_features else Xn

    def prepare_data(self, sequence_list):
        """Returns (X_num, X_feat, y). Kept name-compatible with older callers;
        X_feat is None-safe (zeros) when features are disabled."""
        encoded = self._encode(sequence_list)
        feats = lf.compute_features(encoded, self.num_classes)
        return lf.make_windows(encoded, feats, self.sequence_length)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, history_list, epochs=1, bulk=False):
        """Train incrementally (default) or do a heavy bulk train.

        bulk=True uses many epochs + a validation split + early stopping +
        optional drop-prefix augmentation, where a fast GPU really pays off."""
        Xn, Xf, y = self.prepare_data(history_list)
        if Xn is None or len(Xn) == 0:
            return False

        if bulk:
            epochs = int(_cfg("LSTM_BULK_EPOCHS", 60))
            batch_size = int(_cfg("LSTM_BATCH_SIZE", 64))
            val_split = float(_cfg("LSTM_VALIDATION_SPLIT", 0.15))
            patience = int(_cfg("LSTM_EARLY_STOP_PATIENCE", 8))
            if _cfg("LSTM_AUGMENT", True):
                Xn, Xf, y = lf.augment_drop_random_prefix(
                    Xn, Xf, y,
                    drop_prob=float(_cfg("LSTM_AUGMENT_PROB", 0.5)),
                    max_drop_frac=float(_cfg("LSTM_AUGMENT_MAX_FRAC", 0.5)),
                )
            use_val = len(Xn) >= 40
            callbacks = []
            if use_val:
                callbacks.append(EarlyStopping(
                    monitor="val_accuracy", patience=patience,
                    restore_best_weights=True))
            hist = self.model.fit(
                self._model_inputs(Xn, Xf), y,
                epochs=epochs,
                batch_size=min(batch_size, len(Xn)),
                validation_split=val_split if use_val else 0.0,
                callbacks=callbacks,
                verbose=0,
            )
        else:
            hist = self.model.fit(self._model_inputs(Xn, Xf), y,
                                  epochs=epochs, verbose=0)

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

        feats = lf.compute_features(encoded, self.num_classes)
        win = lf.make_predict_window(encoded, feats, self.sequence_length)
        Xn, Xf = win
        probs = self.model.predict(self._model_inputs(Xn, Xf), verbose=0)[0]

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
        """Walk-forward test: train on the past, predict the next, compare the
        model's top-1 accuracy with the naive 'most frequent so far' baseline.

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
            "n_eval": total, "model_acc": m, "baseline_acc": b,
            "lift": m - b, "verdict": "edge" if m > b else "no_edge",
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

    def _expected_num_len(self):
        """The seq-len the current architecture expects on its number input."""
        return self.sequence_length

    def _model_is_compatible(self, model):
        """Reject stale models whose input shape no longer matches this config
        (e.g. old seq_len=5 / no-feature models saved by <=v1.17.0)."""
        try:
            shapes = model.input_shape
            if isinstance(shapes, list):
                num_shape = shapes[0]
                has_feat_input = len(shapes) > 1
            else:
                num_shape = shapes
                has_feat_input = False
            if num_shape[-1] != self._expected_num_len():
                return False
            if has_feat_input != self.use_features:
                return False
            return True
        except Exception:
            return False

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
                loaded = load_model(path)
                if not self._model_is_compatible(loaded):
                    logging.warning(
                        "TF-LSTM: saved model is incompatible with the current "
                        "architecture (seq_len/features changed); ignoring it.")
                    return False
                self.model = loaded
                self._trained = True
                logging.info(f"TF-LSTM: loaded saved model from {path}")
                return True
        except Exception as e:
            logging.warning(f"TF-LSTM: could not load saved model: {e}")
        return False


def compare_attention_vs_vanilla(history_list, min_train=40):
    """Build BOTH an attention model and a vanilla (attention-off) model, run the
    honest walk-forward backtest on the SAME data, and report the lift of
    attention over vanilla. Requires TensorFlow; intended to be run on the GPU
    machine via run_backtest.py / train_lstm.py.
    """
    att = TfLstmEngine(use_attention=True)
    van = TfLstmEngine(use_attention=False)
    r_att = att.evaluate_backtest(history_list, min_train=min_train)
    r_van = van.evaluate_backtest(history_list, min_train=min_train)
    lift = None
    if r_att.get("model_acc") is not None and r_van.get("model_acc") is not None:
        lift = r_att["model_acc"] - r_van["model_acc"]
    return {
        "attention": r_att,
        "vanilla": r_van,
        "attention_minus_vanilla": lift,
        "verdict": ("attention_better" if (lift is not None and lift > 0)
                    else "no_gain" if lift is not None else "insufficient"),
    }
