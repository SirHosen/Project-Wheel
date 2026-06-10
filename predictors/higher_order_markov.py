# -*- coding: utf-8 -*-
"""predictors/higher_order_markov.py - Variable-order Markov predictor (PROMPT 3).

Extends the plain first-order MarkovEngine to context lengths 1..max_order
("after the sequence [a, b, c], what tends to come next?").

Three honest safeguards against overfitting on a near-random wheel:
  1. CROSS-VALIDATION auto-selects the order. We replay the recent history
     walk-forward for each candidate order and keep the order with the best
     predictive log-loss -- with a parsimony margin so a higher order must be
     *meaningfully* better, not just nominally.
  2. BACKOFF. At prediction time we use the highest order whose exact context
     has actually been observed enough times (>= min_support); otherwise we
     drop to a shorter context, finally to the wheel's frequency prior.
  3. LAPLACE smoothing toward that same frequency prior, so sparse contexts
     degrade gracefully instead of producing fake-confident spikes.

`support` in the output is the observed count of the chosen context, so the
betting / evidence-gate layer still refuses to act on small-sample noise.
"""
from collections import Counter
import math

from .base import BasePredictor
from config import settings

_EPS = 1e-12


class HigherOrderMarkovEngine(BasePredictor):
    def __init__(self, max_order: int = 4, alpha: float = 3.0, min_support: int = 8):
        self.max_order = max(1, int(max_order))
        self.alpha = alpha
        self.min_support = min_support
        self.valid_numbers = settings.VALID_NUMBERS
        self._valid_set = set(self.valid_numbers)
        self.last_selected_order = None

        seq = settings.SPINWHEEL_SEQUENCE
        counts = Counter(seq)
        total = len(seq) or 1
        self.prior = {n: counts.get(n, 0) / total for n in self.valid_numbers}

    # ------------------------------------------------------------------ #
    def _freq_predictions(self) -> list:
        preds = [{"number": n, "confidence": self.prior[n], "support": 0, "order": 0}
                 for n in self.valid_numbers]
        preds.sort(key=lambda x: x["confidence"], reverse=True)
        return preds

    def _build(self, hist, order):
        """context-tuple -> Counter(next) for a fixed order."""
        table = {}
        for t in range(order, len(hist)):
            ctx = tuple(hist[t - order:t])
            table.setdefault(ctx, Counter())[hist[t]] += 1
        return table

    def _smoothed(self, row):
        row_total = sum(row.values())
        probs = {
            n: (row.get(n, 0) + self.alpha * self.prior[n]) / (row_total + self.alpha)
            for n in self.valid_numbers
        }
        s = sum(probs.values()) or 1.0
        return {n: probs[n] / s for n in self.valid_numbers}, row_total

    def _predict_with_backoff(self, hist, order):
        """Use the highest order <= `order` whose context has enough support."""
        tables = {o: self._build(hist, o) for o in range(1, order + 1)}
        for o in range(order, 0, -1):
            ctx = tuple(hist[-o:])
            row = tables[o].get(ctx)
            if row and sum(row.values()) >= self.min_support:
                probs, support = self._smoothed(row)
                return probs, support, o
        return dict(self.prior), 0, 0

    # ------------------------------------------------------------------ #
    def _select_order(self, hist):
        """Walk-forward CV (incremental, O(n) per order) to pick the order."""
        n = len(hist)
        if n < 30:
            return 1
        cv = hist[-250:]
        m = len(cv)
        best_order, best_ll = 1, float("inf")
        for o in range(1, self.max_order + 1):
            counts = {}
            ll, cnt = 0.0, 0
            for t in range(o, m):
                if t >= 20:
                    ctx = tuple(cv[t - o:t])
                    row = counts.get(ctx)
                    if row and sum(row.values()) > 0:
                        rt = sum(row.values())
                        p = (row.get(cv[t], 0) + self.alpha * self.prior[cv[t]]) / (rt + self.alpha)
                    else:
                        p = self.prior.get(cv[t], _EPS)
                    ll += -math.log(max(_EPS, p))
                    cnt += 1
                counts.setdefault(tuple(cv[t - o:t]), Counter())[cv[t]] += 1
            if cnt == 0:
                continue
            avg = ll / cnt
            # parsimony: only upgrade order if clearly (>0.01 nats) better.
            if avg < best_ll - 0.01:
                best_ll, best_order = avg, o
        return best_order

    # ------------------------------------------------------------------ #
    def predict_next(self, history: list) -> list:
        hist = [h for h in history if h in self._valid_set]
        if not hist:
            return self._freq_predictions()
        order = self._select_order(hist)
        probs, support, used_order = self._predict_with_backoff(hist, order)
        self.last_selected_order = used_order
        preds = [{"number": n, "confidence": probs[n], "support": support,
                  "order": used_order}
                 for n in self.valid_numbers]
        preds.sort(key=lambda x: x["confidence"], reverse=True)
        return preds
