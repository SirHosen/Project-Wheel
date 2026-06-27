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
_GLOBAL_KEY = "__global__"  # JSON-safe key for the global (engine=None) map


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
    def __init__(self, path=None, n_bins=10):
        if path is None:
            from config import settings
            path = getattr(settings, "CALIBRATION_STATE_PATH",
                           "runtime/calibration_state.json")
        self.path = path
        self.n_bins = n_bins
        self.pairs = []            # global (p, y)
        self.engine_pairs = {}     # engine -> [(p, y), ...]
        # Isotonic maps are stored PER ENGINE (audit V5 bug fix). Previously a
        # single global attribute was overwritten every round by whichever
        # engine was confirmed last, so calibrate(engine=...) returned the wrong
        # engine's curve. Key None = global pool; string keys = per-engine.
        self._iso_by_engine = {}        # engine -> (xs, ys) PAVA monotone map
        self._sklearn_by_engine = {}    # engine -> fitted sklearn IsotonicRegression
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
        # Fit a monotone p_raw -> p_calibrated map for THIS engine and store it
        # under the engine key (audit V5 bug fix: no longer a single global map).
        pairs = sorted(self._pick(engine), key=lambda t: t[0])
        if len(pairs) < 10:
            self._iso_by_engine[engine] = None
            self._sklearn_by_engine[engine] = None
            return False
        xs = [p for p, _ in pairs]
        ys = [y for _, y in pairs]
        # Try scikit-learn if available (better tie handling).
        try:
            from sklearn.isotonic import IsotonicRegression  # type: ignore
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(xs, ys)
            self._sklearn_by_engine[engine] = ir
            self._iso_by_engine[engine] = None
            return True
        except Exception:
            self._sklearn_by_engine[engine] = None
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
        self._iso_by_engine[engine] = (knots_x, knots_y)
        return True

    def calibrate(self, p, engine=None):
        p = max(0.0, min(1.0, float(p)))
        # Per-engine lookup (audit V5 bug fix): use THIS engine's fitted map,
        # falling back to the global pool, instead of one clobbered global map.
        model = self._sklearn_by_engine.get(engine)
        if model is None:
            model = self._sklearn_by_engine.get(None)
        if model is not None:
            try:
                return float(model.predict([p])[0])
            except Exception:
                pass
        iso = self._iso_by_engine.get(engine)
        if iso is None:
            iso = self._iso_by_engine.get(None)
        if not iso:
            return p
        xs, ys = iso
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
            # Serialize the PAVA isotonic maps PER engine. JSON keys must be
            # strings, so the global (None) map uses the _GLOBAL_KEY sentinel.
            # NOTE: scikit-learn models are NOT serializable here; on envs WITH
            # sklearn the maps simply re-fit on the next confirmed round.
            iso_ser = {}
            for eng, iso in self._iso_by_engine.items():
                if not iso:
                    continue
                key = _GLOBAL_KEY if eng is None else str(eng)
                iso_ser[key] = {"x": iso[0], "y": iso[1]}
            state = {
                "pairs": self.pairs,
                "engine_pairs": self.engine_pairs,
                "isotonic_by_engine": iso_ser,
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
            self._iso_by_engine = {}
            iso_map = state.get("isotonic_by_engine")
            if isinstance(iso_map, dict):
                for key, iso in iso_map.items():
                    eng = None if key == _GLOBAL_KEY else key
                    if iso:
                        self._iso_by_engine[eng] = (iso["x"], iso["y"])
            else:
                # Back-compat: old single global "isotonic" key.
                iso = state.get("isotonic")
                if iso:
                    self._iso_by_engine[None] = (iso["x"], iso["y"])
            # Re-fit every isotonic map from the persisted (p, y) pairs so that
            # calibration SURVIVES a save/load round-trip on ANY backend. scikit-
            # learn's IsotonicRegression is NOT JSON-serializable, so without this
            # a fresh tracker on an env WITH sklearn would silently drop every
            # per-engine curve and calibrate() would decay to an identity no-op
            # (audit V6: test_isotonic_is_per_engine failed ONLY when sklearn was
            # installed). Re-fitting is deterministic -> identical curve -> exact
            # round-trip, and also makes calibration ready immediately at startup.
            self._refit_all()
            return True
        except Exception:
            return False

    def _refit_all(self):
        """Rebuild all isotonic maps from persisted pairs (global + per engine).

        Deterministic: same data + same backend -> identical curve, so a
        save/load round-trip reproduces calibrate() exactly. Engines with fewer
        than 10 pairs simply stay no-ops (fit_isotonic returns False).
        """
        for eng in [None] + list(self.engine_pairs.keys()):
            try:
                self.fit_isotonic(eng)
            except Exception:
                pass