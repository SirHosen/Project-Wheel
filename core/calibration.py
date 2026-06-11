# -*- coding: utf-8 -*-
"""core/calibration.py - Probability reliability + isotonic calibration (PROMPT 4).

Why: an engine can be *accurate-ish* yet *over-confident* -- saying "80%" when
it's right 40% of the time. On a near-random wheel that mismatch is the norm,
and acting on inflated confidence is exactly how a bankroll dies. This module
measures calibration honestly and can re-map raw confidences onto empirically
observed hit-rates.

  - ReliabilityTracker records (predicted_prob, was_correct) per engine.
  - Brier score + log-loss + reliability bins quantify calibration.
  - Isotonic regression (monotone) learns p_raw -> p_calibrated.
      * Uses scikit-learn's IsotonicRegression if installed,
      * otherwise a pure-Python Pool-Adjacent-Violators (PAVA) fallback so it
        works everywhere (no hard dependency).
  - State persists to models/calibration_state.json (additive, drop-in).
"""
import json
import math
import os

_EPS = 1e-12
_MAX_PAIRS = 5000


def _pava(ys, ws):
    """Pool-Adjacent-Violators -> non-decreasing fit. ys already ordered by x."""
    # blocks: [sum_wy, sum_w, value]
    blocks = [[y * w, w, y] for y, w in zip(ys, ws)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2] + _EPS:
            # merge i and i+1
            swy = blocks[i][0] + blocks[i + 1][0]
            sw = blocks[i][1] + blocks[i + 1][1]
            merged = [swy, sw, swy / sw if sw else 0.0]
            blocks[i:i + 2] = [merged]
            if i > 0:
                i -= 1
        else:
            i += 1
    # blocks already carry the pooled, non-decreasing values; callers expand
    # them to per-point values as needed (audit V3 #5: removed dead `fitted`).
    return blocks


class ReliabilityTracker:
    def __init__(self, path="models/calibration_state.json", n_bins=10):
        self.path = path
        self.n_bins = n_bins
        self.pairs = []            # global (p, y)
        self.engine_pairs = {}     # engine -> [(p, y), ...]
        self._iso = None           # (xs, ys) monotone map
        self._sklearn_model = None
        self.load()

    # ------------------------------------------------------------------ #
    def record(self, predicted_prob, was_correct, engine=None):
        p = max(0.0, min(1.0, float(predicted_prob)))
        y = 1.0 if was_correct else 0.0
        self.pairs.append((p, y))
        if len(self.pairs) > _MAX_PAIRS:
            self.pairs = self.pairs[-_MAX_PAIRS:]
        if engine:
            lst = self.engine_pairs.setdefault(engine, [])
            lst.append((p, y))
            if len(lst) > _MAX_PAIRS:
                self.engine_pairs[engine] = lst[-_MAX_PAIRS:]

    def _pick(self, engine=None):
        if engine and self.engine_pairs.get(engine):
            return self.engine_pairs[engine]
        return self.pairs

    # ------------------------------------------------------------------ #
    def brier(self, engine=None):
        pairs = self._pick(engine)
        if not pairs:
            return None
        return sum((p - y) ** 2 for p, y in pairs) / len(pairs)

    def log_loss(self, engine=None):
        pairs = self._pick(engine)
        if not pairs:
            return None
        s = 0.0
        for p, y in pairs:
            pc = min(1 - _EPS, max(_EPS, p))
            s += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
        return s / len(pairs)

    def reliability_bins(self, engine=None):
        pairs = self._pick(engine)
        bins = []
        for b in range(self.n_bins):
            lo, hi = b / self.n_bins, (b + 1) / self.n_bins
            sel = [(p, y) for p, y in pairs
                   if (lo <= p < hi or (b == self.n_bins - 1 and p >= hi))]
            if not sel:
                continue
            bins.append({
                "bin_center": (lo + hi) / 2,
                "mean_predicted": sum(p for p, _ in sel) / len(sel),
                "observed_acc": sum(y for _, y in sel) / len(sel),
                "n": len(sel),
            })
        return bins

    def expected_calibration_error(self, engine=None):
        pairs = self._pick(engine)
        if not pairs:
            return None
        n = len(pairs)
        ece = 0.0
        for b in self.reliability_bins(engine):
            ece += (b["n"] / n) * abs(b["mean_predicted"] - b["observed_acc"])
        return ece

    # ------------------------------------------------------------------ #
    def fit_isotonic(self, engine=None):
        pairs = sorted(self._pick(engine), key=lambda t: t[0])
        if len(pairs) < 10:
            self._iso = None
            return False
        xs = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        # Try scikit-learn if available (better tie handling).
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(xs, ys)
            self._sklearn_model = ir
            self._iso = None
            return True
        except Exception:
            self._sklearn_model = None
        # Pure-python PAVA fallback.
        ws = [1.0] * len(ys)
        blocks = _pava(ys, ws)
        # Build a step map: cumulative x boundaries -> block value.
        knots_x, knots_y = [], []
        idx = 0
        for swy, sw, val in blocks:
            count = int(round(sw))
            # x at the end of this block
            end = min(len(xs) - 1, idx + count - 1)
            knots_x.append(xs[end])
            knots_y.append(val)
            idx += count
        self._iso = (knots_x, knots_y)
        return True

    def calibrate(self, p, engine=None):
        p = max(0.0, min(1.0, float(p)))
        if self._sklearn_model is not None:
            try:
                return float(self._sklearn_model.predict([p])[0])
            except Exception:
                pass
        if not self._iso:
            return p
        xs, ys = self._iso
        # piecewise-linear interpolation over the isotonic knots
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if p <= xs[i]:
                x0, x1 = xs[i - 1], xs[i]
                y0, y1 = ys[i - 1], ys[i]
                if x1 == x0:
                    return y1
                t = (p - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return ys[-1]

    def calibrate_predictions(self, predictions, engine=None, renormalize=True):
        """Return a copy of predictions with confidences isotonically calibrated."""
        out = []
        for p in predictions:
            q = dict(p)
            q["confidence_raw"] = p.get("confidence")
            q["confidence"] = self.calibrate(p.get("confidence", 0.0), engine)
            out.append(q)
        if renormalize:
            s = sum(q["confidence"] for q in out) or 1.0
            for q in out:
                q["confidence"] = q["confidence"] / s
        out.sort(key=lambda x: x["confidence"], reverse=True)
        return out

    # ------------------------------------------------------------------ #
    def summary(self, engine=None):
        return {
            "n": len(self._pick(engine)),
            "brier": self.brier(engine),
            "log_loss": self.log_loss(engine),
            "ece": self.expected_calibration_error(engine),
            "bins": self.reliability_bins(engine),
        }

    # ------------------------------------------------------------------ #
    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            state = {
                "pairs": self.pairs,
                "engine_pairs": self.engine_pairs,
                "isotonic": {"x": self._iso[0], "y": self._iso[1]} if self._iso else None,
            }
            with open(self.path, "w") as f:
                json.dump(state, f)
            return True
        except Exception:
            return False

    def load(self):
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                state = json.load(f)
            self.pairs = [tuple(t) for t in state.get("pairs", [])]
            self.engine_pairs = {k: [tuple(t) for t in v]
                                 for k, v in state.get("engine_pairs", {}).items()}
            iso = state.get("isotonic")
            self._iso = (iso["x"], iso["y"]) if iso else None
            return True
        except Exception:
            return False
