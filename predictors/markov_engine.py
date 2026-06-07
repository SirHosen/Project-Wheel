# -*- coding: utf-8 -*-
from collections import defaultdict, Counter

from .base import BasePredictor
from config import settings


class MarkovEngine(BasePredictor):
    """First-order Markov / transition-matrix predictor.

    Learns P(next | last_number) from the observed spin-result history. This is
    the engine designed to exploit *positional / autocorrelation* bias of the
    physical wheel ("after X, Y tends to come out").

    Sparse-data behaviour: additive (Laplace-style) smoothing pulls every row
    toward the wheel's natural frequency prior. So with little/no history the
    predictions degrade gracefully to "the most frequent numbers", which gives
    the best top-3 hit rate when no autocorrelation has emerged yet. As real
    data accumulates, the observed transitions take over and the model adapts.
    """

    def __init__(self, alpha: float = 3.0):
        # alpha = smoothing strength toward the frequency prior.
        # Higher alpha => trusts the frequency prior more until lots of data.
        self.alpha = alpha
        self.valid_numbers = settings.VALID_NUMBERS

        seq = settings.SPINWHEEL_SEQUENCE
        counts = Counter(seq)
        total = len(seq)
        # Frequency prior derived from the physical wheel layout.
        self.prior = {n: counts.get(n, 0) / total for n in self.valid_numbers}

    def _freq_predictions(self) -> list:
        # No transition evidence -> support 0 (betting layer treats this as
        # "not enough data to trust an +EV signal").
        preds = [{"number": n, "confidence": self.prior[n], "support": 0}
                 for n in self.valid_numbers]
        preds.sort(key=lambda x: x["confidence"], reverse=True)
        return preds

    def predict_next(self, history: list) -> list:
        if not history:
            return self._freq_predictions()

        # Build first-order transition counts from the result history.
        trans = defaultdict(Counter)
        for a, b in zip(history, history[1:]):
            if a in self.prior and b in self.prior:
                trans[a][b] += 1

        last = history[-1]
        if last not in self.prior:
            return self._freq_predictions()

        row = trans[last]
        row_total = sum(row.values())

        # Smoothed conditional probability:
        #   P(next | last) = (count + alpha * prior[next]) / (row_total + alpha)
        probs = {
            n: (row.get(n, 0) + self.alpha * self.prior[n]) / (row_total + self.alpha)
            for n in self.valid_numbers
        }

        # Defensive normalization (should already sum to ~1).
        s = sum(probs.values()) or 1.0
        # `support` = how many transitions we have actually observed FROM the
        # current last-number. The betting layer uses this to avoid acting on
        # confident-looking spikes that are really just small-sample noise.
        preds = [{"number": n, "confidence": probs[n] / s, "support": row_total}
                 for n in self.valid_numbers]
        preds.sort(key=lambda x: x["confidence"], reverse=True)
        return preds
