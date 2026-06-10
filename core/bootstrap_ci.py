# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals for prediction engines (PROMPT 8).

The Bayesian engine derives ci_low/ci_high/support analytically from its
Dirichlet posterior. Heuristic and LSTM engines expose only a point
confidence, so we quantify their uncertainty with a non-parametric bootstrap:
resample the observed history WITH REPLACEMENT (default 200x), recompute each
engine's confidence, and take the 2.5 / 97.5 percentiles.

This is an honest uncertainty proxy, not a guarantee of calibration -- a wide
band means "this engine's confidence is very sensitive to which spins it saw".
"""
import math
import random


def _percentile(sorted_vals, q):
    """Linear-interpolation percentile (q in [0,1]) of a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def bootstrap_distributions(engine, history, n_boot=200, seed=42):
    """Return a list of {number: confidence} dicts, one per bootstrap resample."""
    rng = random.Random(seed)
    n = len(history)
    samples = []
    for _ in range(int(n_boot)):
        if n == 0:
            resampled = []
        else:
            resampled = [history[rng.randrange(n)] for _ in range(n)]
        preds = engine.predict_next(resampled) or []
        samples.append(
            {int(d["number"]): float(d.get("confidence", 0.0)) for d in preds}
        )
    return samples


def bootstrap_ci_for_numbers(engine, history, numbers, n_boot=200, alpha=0.05, seed=42):
    """Map each requested number -> (ci_low, ci_high) at the (1-alpha) level."""
    samples = bootstrap_distributions(engine, history, n_boot=n_boot, seed=seed)
    out = {}
    for num in numbers:
        vals = sorted(s.get(int(num), 0.0) for s in samples)
        out[int(num)] = (
            _percentile(vals, alpha / 2.0),
            _percentile(vals, 1.0 - alpha / 2.0),
        )
    return out


def attach_confidence_intervals(
    engine, history, preds, n_boot=200, alpha=0.05, seed=42, top_k=3
):
    """Enrich predictions in-place with ci_low / ci_high / support.

    - Predictions that already carry ci_low/ci_high (e.g. Bayesian) are left
      untouched.
    - For the top_k predictions lacking a CI, run a single shared bootstrap.
    - support falls back to len(history) (the # of observations the estimate
      rests on) when the engine does not report its own support count.
    """
    if not preds:
        return preds
    targets = preds[:top_k]
    need = [p for p in targets if p.get("ci_low") is None or p.get("ci_high") is None]
    ci_map = {}
    if need:
        nums = [int(p["number"]) for p in need]
        ci_map = bootstrap_ci_for_numbers(
            engine, history, nums, n_boot=n_boot, alpha=alpha, seed=seed
        )
    for p in preds:
        num = int(p["number"])
        if (p.get("ci_low") is None or p.get("ci_high") is None) and num in ci_map:
            p["ci_low"], p["ci_high"] = ci_map[num]
        if p.get("support") is None:
            p["support"] = len(history)
    return preds


def support_label(support):
    """Return (label, color_key) for an evidence-strength badge.

    color_key matches UI_COLORS keys: error=red, secondary=gold, primary=green.
    """
    if support is None:
        return ("RAW (cold start)", "error")
    s = int(support)
    if s < 25:
        return ("RAW (cold start)", "error")
    if s <= 100:
        return ("WARMING UP", "secondary")
    return ("STABLE", "primary")
