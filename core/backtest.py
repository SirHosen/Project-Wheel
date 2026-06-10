# -*- coding: utf-8 -*-
"""core/backtest.py - Universal walk-forward backtester (PROMPT 2).

Pure standard library (no TensorFlow / pandas) so it can grade ANY engine that
implements `predict_next(history) -> [{"number", "confidence", ...}]` and run
anywhere (CI included). Heavy engines (LSTM / Ensemble) can still be passed in;
if importing or predicting fails the comparison simply skips them.

Honest metrics only:
  - top-1 / top-3 accuracy
  - multi-class Brier score + log-loss (calibration quality)
  - a naive 'most-frequent-so-far' baseline + two-proportion z vs that baseline
  - a SKIP-free unit-bet profit simulation + its Sharpe
A verdict of 'edge' REQUIRES statistically beating the baseline (z > 1.96) --
exactly the bar a fair wheel will (correctly) fail.
"""
from __future__ import annotations

import math
from collections import Counter

try:
    from config.settings import calculate_reward
except Exception:  # pragma: no cover
    def calculate_reward(token_bet, n):
        return (token_bet * (1 if n == 1 else n)) + token_bet

_EPS = 1e-12


class WalkForwardBacktester:
    """Grades engines walk-forward on a fixed sequence of real outcomes."""

    def __init__(self, actuals, valid_numbers, sequence):
        valid = set(valid_numbers)
        self.actuals = [a for a in actuals if a in valid]
        self.valid_numbers = list(valid_numbers)
        self.sequence = list(sequence)

    # ------------------------------------------------------------------ #
    def _prob_dist(self, engine, hist):
        """Normalized probability over valid_numbers from an engine's preds."""
        try:
            preds = engine.predict_next(hist) or []
        except Exception:
            preds = []
        dist = {n: 0.0 for n in self.valid_numbers}
        for p in preds:
            num = p.get("number")
            if num in dist:
                dist[num] = max(0.0, float(p.get("confidence", 0.0) or 0.0))
        total = sum(dist.values())
        if total <= 0:
            u = 1.0 / len(self.valid_numbers)
            return {n: u for n in self.valid_numbers}
        return {n: v / total for n, v in dist.items()}

    # ------------------------------------------------------------------ #
    def backtest_engine(self, engine, warmup=40, train_fn=None, refit_every=None):
        """Walk-forward grade one engine. Returns a metrics dict."""
        actuals = self.actuals
        n = len(actuals)
        res = {
            "n_eval": 0, "top1_acc": None, "top3_acc": None,
            "brier_score": None, "log_loss": None,
            "baseline_top1_acc": None, "lift": None, "z_score": None,
            "verdict": "insufficient",
            "simulated_profit_unit_bet": None, "sharpe": None,
            "note": "Butuh >= warmup+10 hasil untuk backtest.",
        }
        if n < warmup + 10:
            return res

        top1_hits = top3_hits = base_hits = 0
        brier_sum = logloss_sum = 0.0
        profits = []
        rounds = 0
        for i in range(warmup, n - 1):
            hist = actuals[: i + 1]
            actual = actuals[i + 1]
            if train_fn is not None and refit_every and (i - warmup) % refit_every == 0:
                try:
                    train_fn(engine, hist)
                except Exception:
                    pass
            dist = self._prob_dist(engine, hist)
            ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            top1 = ranked[0][0]
            top3 = {k for k, _ in ranked[:3]}
            base_top1 = Counter(hist).most_common(1)[0][0]

            if top1 == actual:
                top1_hits += 1
            if actual in top3:
                top3_hits += 1
            if base_top1 == actual:
                base_hits += 1
            for num, p in dist.items():
                ind = 1.0 if num == actual else 0.0
                brier_sum += (p - ind) ** 2
            logloss_sum += -math.log(max(_EPS, dist.get(actual, 0.0)))
            profits.append(calculate_reward(1, actual) - 1 if top1 == actual else -1)
            rounds += 1

        mk = top1_hits / rounds
        base = base_hits / rounds
        pooled = (top1_hits + base_hits) / (2 * rounds)
        denom = math.sqrt(max(_EPS, pooled * (1 - pooled) * (2 / rounds)))
        z = (mk - base) / denom if denom > 0 else 0.0
        verdict = "edge" if (z > 1.96 and mk > base) else "no_edge"
        mean_p = sum(profits) / rounds
        sd_p = math.sqrt(sum((x - mean_p) ** 2 for x in profits) / (rounds - 1)) if rounds > 1 else 0.0
        sharpe = (mean_p / sd_p * math.sqrt(252)) if sd_p > 0 else 0.0
        res.update({
            "n_eval": rounds,
            "top1_acc": mk, "top3_acc": top3_hits / rounds,
            "brier_score": brier_sum / rounds, "log_loss": logloss_sum / rounds,
            "baseline_top1_acc": base, "lift": mk - base, "z_score": z,
            "verdict": verdict,
            "simulated_profit_unit_bet": sum(profits),
            "sharpe": sharpe,
            "note": ("Edge statistik terdeteksi (model > baseline)." if verdict == "edge"
                     else "Belum ada edge: model tidak mengalahkan tebakan paling sering."),
        })
        return res

    # ------------------------------------------------------------------ #
    def compare_all_engines(self, engines, warmup=40):
        """Backtest a {name: engine} dict; return rows sorted best-first."""
        rows = []
        for name, eng in engines.items():
            r = self.backtest_engine(eng, warmup=warmup)
            r["engine"] = name
            rows.append(r)
        rows.sort(key=lambda x: (x["top1_acc"] is None, -(x["top1_acc"] or 0)))
        return rows

    # ------------------------------------------------------------------ #
    def calibration_curve(self, engine, warmup=40, bins=10):
        """Reliability of the engine's top-1 confidence vs its real hit-rate."""
        actuals = self.actuals
        n = len(actuals)
        pairs = []
        for i in range(warmup, n - 1):
            hist = actuals[: i + 1]
            actual = actuals[i + 1]
            dist = self._prob_dist(engine, hist)
            num, p = max(dist.items(), key=lambda kv: kv[1])
            pairs.append((p, 1.0 if num == actual else 0.0))
        curve = []
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            sel = [(p, h) for p, h in pairs
                   if (lo <= p < hi or (b == bins - 1 and p >= hi))]
            if not sel:
                continue
            curve.append({
                "bin_center": (lo + hi) / 2,
                "expected_acc": sum(p for p, _ in sel) / len(sel),
                "actual_acc": sum(h for _, h in sel) / len(sel),
                "n_in_bin": len(sel),
            })
        return curve
