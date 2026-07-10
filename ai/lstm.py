# -*- coding: utf-8 -*-
"""A clean, well-documented LSTM next-result model + an HONEST walk-forward
backtest. This is the AI-training playground -- on PyTorch so it can use an
NVIDIA GPU on native Windows (no WSL needed).

Architecture (small and readable):
    Embedding(9 -> EMBEDDING_DIM) [+ richer per-step features]
        -> LSTM(LSTM_UNITS) -> Dropout -> Linear(9)

Beyond the raw number index (all an Embedding sees), we can feed richer causal
features per timestep -- running frequency, recency gap, fair base rate and a
repeat flag (see ai/dataset.compute_features). They are concatenated onto the
embedding before the LSTM, are OPT-IN via `use_features`, and are computed
causally so nothing leaks from the future.

The backtest is the most important part. It is honest in three ways:
  1. walk-forward (train on the past, predict the very next spin);
  2. it compares the LSTM against SEVERAL baselines (most-frequent, moving
     frequency, persistence, and the fair-design argmax) on the SAME points;
  3. it reports probabilistic calibration (log-loss + Brier), not just top-1
     accuracy, so you can see whether the predicted probabilities are actually
     trustworthy. On a FAIR wheel the LSTM will NOT beat the baselines -- that
     is the real lesson. On genuinely sequential data (synthetic_markov) it can.

PyTorch is optional: if it is not installed, available() says so and the model
rows are simply omitted -- the baselines and the frequency-model calibration
still run so you always get a comparison.
"""
import numpy as np

from config import (BATCH_SIZE, DROPOUT, EARLY_STOP_PATIENCE, EMBEDDING_DIM,
                    EPOCHS, LSTM_UNITS, MODEL_PATH, SEQUENCE_LENGTH,
                    VALIDATION_SPLIT)
from ai.dataset import (FEATURE_DIM, NUM_CLASSES, NUMBER, compute_features,
                        from_indices, make_feature_windows, make_windows,
                        to_indices)
from ai import metrics
from core.wheel import design_distribution

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
        """Embedding [+ features] -> LSTM -> Dropout -> Linear classifier.

        When feat_dim > 0 the forward pass expects a (B, T, feat_dim) tensor of
        per-timestep features, concatenated onto the embedding before the LSTM.
        """

        def __init__(self, num_classes=NUM_CLASSES, embed=EMBEDDING_DIM,
                     hidden=LSTM_UNITS, dropout=DROPOUT, feat_dim=0):
            super().__init__()
            self.feat_dim = int(feat_dim)
            self.embed = nn.Embedding(num_classes, embed)
            self.lstm = nn.LSTM(embed + self.feat_dim, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x, feats=None):
            e = self.embed(x)              # (B, T) -> (B, T, E)
            if self.feat_dim:
                if feats is None:
                    raise ValueError("model expects features but got none")
                e = torch.cat([e, feats], dim=-1)  # (B, T, E+F)
            out, _ = self.lstm(e)          # (B, T, H)
            out = out[:, -1, :]            # last timestep
            return self.fc(self.drop(out))  # (B, C) logits
else:  # pragma: no cover - keeps import working without torch
    SpinLSTM = None


def build_model(seq_len=SEQUENCE_LENGTH, feat_dim=0):
    """Build the small LSTM classifier on the active device."""
    if not available():
        raise RuntimeError(status_line())
    return SpinLSTM(feat_dim=feat_dim).to(device())


def _accuracy(model, X, y, F=None):
    model.eval()
    with torch.no_grad():
        logits = model(X, F)
        pred = logits.argmax(dim=1)
        return float((pred == y).float().mean().item())


def _fit(model, X, y, epochs, batch_size, lr=1e-3, val=None, F=None,
         valF=None, patience=EARLY_STOP_PATIENCE, monitor_loss=False,
         verbose=0):
    """Generic training loop with optional early stopping + best-weight restore.
    `F` / `valF` are optional per-timestep feature tensors aligned with X / val.
    Returns a history dict: {loss, accuracy, val_accuracy}.
    """
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
            logits = model(X[b], F[b] if F is not None else None)
            loss = loss_fn(logits, y[b])
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(b)
        epoch_loss = running / max(1, n)
        hist["loss"].append(epoch_loss)
        hist["accuracy"].append(_accuracy(model, X, y, F))
        if val is not None:
            vacc = _accuracy(model, val[0], val[1], valF)
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


def _feat_tensor(F, dev):
    return None if F is None else torch.as_tensor(F, dtype=torch.float32, device=dev)


def train(numbers, seq_len=SEQUENCE_LENGTH, epochs=EPOCHS, verbose=1,
          save_path=MODEL_PATH, use_features=True):
    """Train on a result history. Returns (model, history_dict)."""
    if not available():
        raise RuntimeError(status_line())
    if use_features:
        X, Xf, y = make_feature_windows(numbers, seq_len)
        feat_dim = FEATURE_DIM
    else:
        X, y = make_windows(numbers, seq_len)
        Xf, feat_dim = None, 0
    if len(X) < 20:
        raise ValueError(f"need >=20 windows to train, got {len(X)} "
                         f"(feed more results; you have {len(numbers)})")
    dev = device()
    Xt, yt = _to_tensors(X, y, dev)
    Ft = _feat_tensor(Xf, dev)
    # Keras-style validation_split: hold out the LAST fraction (no shuffle).
    n_val = int(len(Xt) * VALIDATION_SPLIT)
    if n_val >= 5:
        Xtr, ytr = Xt[:-n_val], yt[:-n_val]
        val = (Xt[-n_val:], yt[-n_val:])
        Ftr = None if Ft is None else Ft[:-n_val]
        valF = None if Ft is None else Ft[-n_val:]
    else:
        Xtr, ytr, val = Xt, yt, None
        Ftr, valF = Ft, None
    model = build_model(seq_len, feat_dim=feat_dim)
    hist = _fit(model, Xtr, ytr, epochs=epochs, batch_size=BATCH_SIZE,
                val=val, F=Ftr, valF=valF, verbose=(1 if verbose else 0))
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model, hist


def _most_frequent_so_far(idx_history):
    """Baseline prediction: the most common class seen so far."""
    counts = np.bincount(idx_history, minlength=NUM_CLASSES)
    return int(np.argmax(counts))


def _freq_probs(idx_arr, t):
    """Laplace-smoothed empirical probability vector from history idx_arr[:t].
    Always well-defined (no zeros), so it is a safe probabilistic baseline."""
    counts = np.bincount(idx_arr[:t], minlength=NUM_CLASSES).astype(np.float64)
    return (counts + 1.0) / (counts.sum() + NUM_CLASSES)


def _baseline_predictors(recent_window=50):
    """Return {name: fn(idx_arr, t) -> predicted class index}. Each baseline is
    CAUSAL: it may only look at idx_arr[:t] (and idx_arr[t-1] for persistence).

      most_frequent : the single most common number over ALL history so far
      moving_freq   : most common number over only the last `recent_window`
      persistence   : just repeat the previous number (wins iff the stream is
                      sequential, e.g. synthetic_markov)
      design_argmax : the number the FAIR design makes most likely (static)
    """
    design = design_distribution()
    design_idx = int(np.argmax([design[NUMBER[i]] for i in range(NUM_CLASSES)]))

    def most_frequent(idx_arr, t):
        return int(np.argmax(np.bincount(idx_arr[:t], minlength=NUM_CLASSES)))

    def moving_freq(idx_arr, t):
        lo = max(0, t - recent_window)
        return int(np.argmax(np.bincount(idx_arr[lo:t], minlength=NUM_CLASSES)))

    def persistence(idx_arr, t):
        return int(idx_arr[t - 1])

    def design_argmax(idx_arr, t):
        return design_idx

    return {
        "most_frequent": most_frequent,
        f"moving_freq_{recent_window}": moving_freq,
        "persistence": persistence,
        "design_argmax": design_argmax,
    }


def _run_walk_forward(numbers, seq_len, min_train, step_retrain, epochs,
                      use_features, with_model, recent_window=50):
    """Shared walk-forward engine. Scores every baseline (always) and, when
    `with_model` and PyTorch are available, the LSTM too -- all on the SAME
    evaluation points so results are directly comparable. Also accumulates
    probabilistic calibration (log-loss + Brier) for the frequency baseline
    (always) and the LSTM (when it runs).

    Returns {n_eval, baseline_acc:{name:acc}, prob:{freq:{log_loss,brier},
    model?:{...}}, model_acc?:float}.
    """
    idx = to_indices(numbers)
    clean = from_indices(idx)
    if len(idx) < min_train + seq_len + 20:
        raise ValueError("not enough data for a trustworthy backtest "
                         f"(have {len(idx)}, need >= {min_train + seq_len + 20})")
    idx_arr = np.asarray(idx)
    preds = _baseline_predictors(recent_window)
    hits = {name: 0 for name in preds}
    run_model = bool(with_model) and available()
    model_hits = total = 0
    ll_freq = br_freq = ll_model = br_model = 0.0
    feat_dim = FEATURE_DIM if use_features else 0
    dev = device() if run_model else None
    model = None
    last_train_at = -10**9
    for t in range(min_train, len(idx) - 1):
        if run_model and (model is None or (t - last_train_at) >= step_retrain):
            if use_features:
                X, Xf, y = make_feature_windows(clean[:t], seq_len)
            else:
                X, y = make_windows(clean[:t], seq_len)
                Xf = None
            if len(X) >= 20:
                Xt, yt = _to_tensors(X, y, dev)
                Ft = _feat_tensor(Xf, dev)
                model = build_model(seq_len, feat_dim=feat_dim)
                _fit(model, Xt, yt, epochs=epochs, batch_size=BATCH_SIZE,
                     F=Ft, monitor_loss=True, patience=3, verbose=0)
                last_train_at = t
        truth = idx[t]
        for name, fn in preds.items():
            hits[name] += int(fn(idx_arr, t) == truth)
        # Probabilistic baseline (always available, no torch needed).
        pf = _freq_probs(idx_arr, t)
        ll_freq += metrics.log_loss_single(pf, truth)
        br_freq += metrics.brier_single(pf, truth)
        if run_model and model is not None:
            window = torch.as_tensor([idx[t - seq_len:t]], dtype=torch.long, device=dev)
            wfeat = None
            if use_features:
                fall = compute_features(clean[:t], seq_len)[-seq_len:]
                wfeat = torch.as_tensor(fall[None, :, :], dtype=torch.float32, device=dev)
            model.eval()
            with torch.no_grad():
                logits = model(window, wfeat)
                pred = int(logits.argmax(dim=1).item())
                pm = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            model_hits += int(pred == truth)
            ll_model += metrics.log_loss_single(pm, truth)
            br_model += metrics.brier_single(pm, truth)
        total += 1

    denom = max(1, total)
    out = {"n_eval": total,
           "baseline_acc": {n: (hits[n] / denom) for n in preds},
           "prob": {"freq": {"log_loss": ll_freq / denom,
                             "brier": br_freq / denom}}}
    if run_model:
        out["model_acc"] = model_hits / denom
        out["prob"]["model"] = {"log_loss": ll_model / denom,
                                "brier": br_model / denom}
    return out


def walk_forward_backtest(numbers, seq_len=SEQUENCE_LENGTH, min_train=80,
                          step_retrain=40, epochs=8, verbose=0,
                          use_features=True):
    """Honest walk-forward evaluation: train on the past, predict the next.

    Returns a dict with the model's top-1 accuracy, the most-frequent baseline
    accuracy, the lift, and the sample size. If the model does not beat the
    baseline by a clear margin, there is no real edge -- exactly what you should
    expect on a fair wheel.
    """
    if not available():
        raise RuntimeError(status_line())
    res = _run_walk_forward(numbers, seq_len, min_train, step_retrain, epochs,
                            use_features, with_model=True)
    model_acc = res["model_acc"]
    base_acc = res["baseline_acc"]["most_frequent"]
    return {"n_eval": res["n_eval"], "model_acc": model_acc,
            "baseline_acc": base_acc, "lift": model_acc - base_acc,
            "verdict": ("model beats baseline (real sequential signal)"
                        if model_acc - base_acc > 0.02 else
                        "NO edge: model ~= baseline (expected on a fair wheel)")}


def walk_forward_table(numbers, seq_len=SEQUENCE_LENGTH, min_train=80,
                       step_retrain=40, epochs=8, use_features=True,
                       recent_window=50):
    """Compare the LSTM against SEVERAL baselines on the same walk-forward split.

    Returns {n_eval, rows:[{name, acc, lift}], best, verdict}. `lift` is each
    strategy's accuracy minus the strongest BASELINE (so a positive model lift
    means the LSTM genuinely adds something over simple heuristics). Works even
    without PyTorch -- the model row is simply omitted.
    """
    res = _run_walk_forward(numbers, seq_len, min_train, step_retrain, epochs,
                            use_features, with_model=available(),
                            recent_window=recent_window)
    accs = dict(res["baseline_acc"])
    best_baseline = max(accs.values()) if accs else 0.0
    rows = [{"name": n, "acc": a, "lift": a - best_baseline} for n, a in accs.items()]
    if "model_acc" in res:
        rows.append({"name": "lstm_model", "acc": res["model_acc"],
                     "lift": res["model_acc"] - best_baseline})
    rows.sort(key=lambda r: r["acc"], reverse=True)
    best = rows[0]["name"] if rows else None
    if "model_acc" in res and res["model_acc"] - best_baseline > 0.02:
        verdict = "LSTM beats every baseline (real sequential signal)"
    elif "model_acc" not in res:
        verdict = "baselines only (PyTorch not installed)"
    else:
        verdict = "NO edge: a simple baseline is as good as the LSTM"
    return {"n_eval": res["n_eval"], "rows": rows, "best": best, "verdict": verdict}


def probabilistic_report(numbers, seq_len=SEQUENCE_LENGTH, min_train=80,
                         step_retrain=40, epochs=8, use_features=True):
    """Walk-forward probabilistic calibration (log-loss + Brier) for the
    frequency baseline and, when PyTorch is installed, the LSTM. Lower is
    better; the uniform reference is log-loss=ln(9)~=2.197, Brier=1-1/9~=0.889.
    Returns {n_eval, freq:{log_loss,brier}, model?:{log_loss,brier}}.
    """
    res = _run_walk_forward(numbers, seq_len, min_train, step_retrain, epochs,
                            use_features, with_model=available())
    rep = {"n_eval": res["n_eval"], "freq": res["prob"]["freq"]}
    if "model" in res["prob"]:
        rep["model"] = res["prob"]["model"]
    return rep


def format_backtest_table(report):
    """Render walk_forward_table() output as a compact fixed-width text table."""
    lines = [f"strategy          accuracy   lift_vs_best_baseline  (n={report['n_eval']})",
             "-" * 62]
    for r in report["rows"]:
        star = "  <-- best" if r["name"] == report["best"] else ""
        lines.append(f"{r['name']:<16}  {r['acc']:>7.3f}   {r['lift']:>+8.3f}{star}")
    lines.append(f"verdict: {report['verdict']}")
    return "\n".join(lines)


def format_probabilistic_report(report):
    """Render probabilistic_report() output (log-loss + Brier) as text."""
    uni_ll = float(np.log(NUM_CLASSES))
    uni_br = 1.0 - 1.0 / NUM_CLASSES
    lines = [f"probabilistic calibration (lower=better, n={report['n_eval']}):",
             f"  uniform reference   log_loss={uni_ll:.3f}  brier={uni_br:.3f}",
             f"  frequency baseline  log_loss={report['freq']['log_loss']:.3f}  "
             f"brier={report['freq']['brier']:.3f}"]
    if "model" in report:
        lines.append(f"  lstm model          log_loss={report['model']['log_loss']:.3f}  "
                     f"brier={report['model']['brier']:.3f}")
    return "\n".join(lines)
