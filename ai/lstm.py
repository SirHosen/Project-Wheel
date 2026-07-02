# -*- coding: utf-8 -*-
"""A clean, well-documented LSTM next-result model + an HONEST walk-forward
backtest. This is the AI-training playground -- now on PyTorch so it can use an
NVIDIA GPU on native Windows (no WSL needed).

Architecture (small and readable):
    Embedding(9 -> EMBEDDING_DIM) -> LSTM(LSTM_UNITS) -> Dropout -> Linear(9)

The backtest is the most important part: it compares the model's top-1 accuracy
against a trivial "always guess the most frequent number so far" baseline, in a
walk-forward (train-on-past, test-on-next) fashion. On a FAIR wheel the model
will NOT beat the baseline -- that is the real lesson. On data with a genuine
pattern (see ai/dataset.synthetic_markov) it can, and you will see it.

PyTorch is optional: if it is not installed, available() says so and the trainer
raises a clear, friendly error instead of crashing the whole app. When a CUDA
GPU is present the model trains on it automatically (the panel goes GREEN).
"""
import numpy as np

from config import (BATCH_SIZE, DROPOUT, EARLY_STOP_PATIENCE, EMBEDDING_DIM,
                    EPOCHS, LSTM_UNITS, MODEL_PATH, SEQUENCE_LENGTH,
                    VALIDATION_SPLIT)
from ai.dataset import NUM_CLASSES, from_indices, make_windows, to_indices

try:
    import torch
    import torch.nn as nn
    _TORCH_ERR = None
except Exception as e:  # pragma: no cover - env dependent
    torch = None
    nn = None
    _TORCH_ERR = e


def available():
    return torch is not None


def device():
    """The torch device training will run on (cuda if present, else cpu)."""
    if torch is None:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def status_line():
    if not available():
        name = type(_TORCH_ERR).__name__ if _TORCH_ERR else "missing"
        return (f"pytorch NOT available ({name})"
                "  -> pip install -r requirements-ai.txt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return f"pytorch {torch.__version__} on {dev} ({name})"


if available():
    class SpinLSTM(nn.Module):
        """Embedding -> LSTM -> Dropout -> Linear classifier."""

        def __init__(self, num_classes=NUM_CLASSES, embed=EMBEDDING_DIM,
                     hidden=LSTM_UNITS, dropout=DROPOUT):
            super().__init__()
            self.embed = nn.Embedding(num_classes, embed)
            self.lstm = nn.LSTM(embed, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            x = self.embed(x)              # (B, T) -> (B, T, E)
            out, _ = self.lstm(x)          # (B, T, H)
            out = out[:, -1, :]            # last timestep
            return self.fc(self.drop(out))  # (B, C) logits
else:  # pragma: no cover - keeps import working without torch
    SpinLSTM = None


def build_model(seq_len=SEQUENCE_LENGTH):
    """Build the small LSTM classifier on the active device."""
    if not available():
        raise RuntimeError(status_line())
    return SpinLSTM().to(device())


def _accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1)
        return float((pred == y).float().mean().item())


def _fit(model, X, y, epochs, batch_size, lr=1e-3, val=None,
         patience=EARLY_STOP_PATIENCE, monitor_loss=False, verbose=0):
    """Generic training loop with optional early stopping + best-weight restore.
    Returns a history dict: {loss, accuracy, val_accuracy}."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = X.shape[0]
    hist = {"loss": [], "accuracy": [], "val_accuracy": []}
    best_metric = 1e18 if monitor_loss else -1e18
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=X.device)
        running = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(X[b])
            loss = loss_fn(logits, y[b])
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(b)
        epoch_loss = running / max(1, n)
        hist["loss"].append(epoch_loss)
        hist["accuracy"].append(_accuracy(model, X, y))
        if val is not None:
            vacc = _accuracy(model, val[0], val[1])
            hist["val_accuracy"].append(vacc)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss={epoch_loss:.4f}"
                  + (f" val_acc={hist['val_accuracy'][-1]:.3f}" if val is not None else ""))
        # Early stopping.
        metric = epoch_loss if monitor_loss else (hist["val_accuracy"][-1]
                                                   if val is not None else -epoch_loss)
        improved = (metric < best_metric) if monitor_loss else (metric > best_metric)
        if improved:
            best_metric = metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def _to_tensors(X, y, dev):
    Xt = torch.as_tensor(X, dtype=torch.long, device=dev)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    return Xt, yt


def train(numbers, seq_len=SEQUENCE_LENGTH, epochs=EPOCHS, verbose=1,
          save_path=MODEL_PATH):
    """Train on a result history. Returns (model, history_dict)."""
    if not available():
        raise RuntimeError(status_line())
    X, y = make_windows(numbers, seq_len)
    if len(X) < 20:
        raise ValueError(f"need >=20 windows to train, got {len(X)} "
                         f"(feed more results; you have {len(numbers)})")
    dev = device()
    Xt, yt = _to_tensors(X, y, dev)
    # Keras-style validation_split: hold out the LAST fraction (no shuffle).
    n_val = int(len(Xt) * VALIDATION_SPLIT)
    if n_val >= 5:
        Xtr, ytr, val = Xt[:-n_val], yt[:-n_val], (Xt[-n_val:], yt[-n_val:])
    else:
        Xtr, ytr, val = Xt, yt, None
    model = build_model(seq_len)
    hist = _fit(model, Xtr, ytr, epochs=epochs, batch_size=BATCH_SIZE,
                val=val, verbose=(1 if verbose else 0))
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model, hist


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
    clean = from_indices(idx)
    if len(idx) < min_train + seq_len + 20:
        raise ValueError("not enough data for a trustworthy backtest "
                         f"(have {len(idx)}, need >= {min_train + seq_len + 20})")
    dev = device()
    model = None
    model_hits = base_hits = total = 0
    last_train_at = -10**9
    for t in range(min_train, len(idx) - 1):
        if model is None or (t - last_train_at) >= step_retrain:
            X, y = make_windows(clean[:t], seq_len)
            if len(X) >= 20:
                Xt, yt = _to_tensors(X, y, dev)
                model = build_model(seq_len)
                _fit(model, Xt, yt, epochs=epochs, batch_size=BATCH_SIZE,
                     monitor_loss=True, patience=3, verbose=0)
                last_train_at = t
        truth = idx[t]
        if model is not None:
            window = torch.as_tensor([idx[t - seq_len:t]], dtype=torch.long, device=dev)
            model.eval()
            with torch.no_grad():
                pred = int(model(window).argmax(dim=1).item())
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
