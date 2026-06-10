# -*- coding: utf-8 -*-
"""
core/continuous_engine.py — Unified continuous-learning ensemble ("the brain").

Fuses every signal the project has into ONE continuously-updated probability
distribution, and — crucially — LEARNS HOW MUCH TO TRUST each signal over time
based on its recent walk-forward accuracy:

  1. PHYSICS prior   : area fraction of each number on the real wheel
                       (core/physics_wheel.py). This is the ground-truth
                       probability of a fair wheel.
  2. BAYES posterior : Dirichlet posterior over observed results, seeded by the
                       physics prior. Converges to the wheel you actually see.
  3. MARKOV          : transition model over recent results.
  4. TF-LSTM (GPU)   : deep sequence model, trained incrementally every spin.

Every confirmed spin calls observe(), which:
  - scores each sub-model (did its top pick match the real result?),
  - updates an exponential-moving-average accuracy per model,
  - re-derives each model's blend weight via a softmax of those accuracies,
  - persists the learning state to disk so it is CONTINUOUS across launches.

Honest note: on a fair wheel every model's accuracy converges to the same
frequency baseline, so the weights stay balanced and the ensemble converges to
the physics area-fractions. The machinery is real; it just cannot manufacture
an edge that the wheel does not have.
"""

import json
import math
import os
from collections import Counter

from config import settings
from core.physics_wheel import WheelPhysics


_EPS = 1e-12


def _normalize(d: dict) -> dict:
    total = float(sum(d.values()))
    if total <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: v / total for k, v in d.items()}


class ContinuousLearningEngine:
    """Adaptive ensemble that learns continuously from every spin."""

    MODELS = ("physics", "bayes", "markov", "lstm")

    def __init__(self, sequence=None, valid_numbers=None,
                 lstm_engine=None, markov_engine=None,
                 ema_lr: float = 0.08, temperature: float = 0.15):
        self.sequence = list(sequence if sequence is not None else settings.SPINWHEEL_SEQUENCE)
        self.valid_numbers = list(valid_numbers if valid_numbers is not None else settings.VALID_NUMBERS)
        self.lstm_engine = lstm_engine
        self.markov_engine = markov_engine
        self.ema_lr = float(ema_lr)
        self.temperature = float(temperature)

        self.physics = WheelPhysics(sequence=self.sequence, valid_numbers=self.valid_numbers)
        self._phys_prior = _normalize({int(k): v for k, v in self.physics.area_fractions().items()})

        # Bayesian prior counts seeded from the physical wheel layout (one wheel
        # of pseudo-observations) so the posterior is sane from spin #1.
        strength = len(self.sequence)
        self.prior_counts = {n: self._phys_prior[n] * strength for n in self.valid_numbers}

        # Learning state (persisted): EMA accuracy per model + spins seen.
        self.scores = {m: 0.0 for m in self.MODELS}
        # BMA (PROMPT 6): discounted predictive log-evidence per model.
        self.log_evidence = {m: 0.0 for m in self.MODELS}
        self.bma_discount = float(getattr(settings, "ENSEMBLE_BMA_DISCOUNT", 0.98))
        self.bma_temperature = float(getattr(settings, "ENSEMBLE_BMA_TEMPERATURE", 1.0))
        self.use_stacking = bool(getattr(settings, "ENSEMBLE_USE_STACKING", False))
        self.n_observed = 0
        self.prediction_log = []  # rolling [{i, actual, picks{model:number}, pa{model:prob}}]
        self._load_state()

    # ------------------------------------------------------------------ #
    # Individual signal distributions (all normalized over valid_numbers)
    # ------------------------------------------------------------------ #
    def physics_prior(self) -> dict:
        return dict(self._phys_prior)

    def bayes_posterior(self, history: list) -> dict:
        counts = Counter(history)
        post = {n: self.prior_counts.get(n, 0.0) + counts.get(n, 0) for n in self.valid_numbers}
        return _normalize(post)

    def _engine_distribution(self, engine, history: list) -> dict:
        """Turn an engine.predict_next() list into a normalized dict, or None."""
        if engine is None:
            return None
        try:
            preds = engine.predict_next(history)
        except Exception:
            return None
        if not preds:
            return None
        d = {n: 0.0 for n in self.valid_numbers}
        for p in preds:
            num = int(p.get("number"))
            if num in d:
                d[num] = max(0.0, float(p.get("confidence", 0.0)))
        return _normalize(d)

    def lstm_distribution(self, history: list) -> dict:
        eng = self.lstm_engine
        if eng is None or not getattr(eng, "_trained", False):
            return None
        return self._engine_distribution(eng, history)

    def markov_distribution(self, history: list) -> dict:
        return self._engine_distribution(self.markov_engine, history)

    def _all_distributions(self, history: list) -> dict:
        return {
            "physics": self.physics_prior(),
            "bayes": self.bayes_posterior(history),
            "markov": self.markov_distribution(history),
            "lstm": self.lstm_distribution(history),
        }

    def model_distributions(self, history: list) -> dict:
        """Public: each constituent model's normalized {number: prob}, dropping
        any that are unavailable (e.g. an untrained LSTM). Used by the PROMPT 17
        consensus filter as independent voters."""
        return {m: d for m, d in self._all_distributions(history).items() if d}

    # ------------------------------------------------------------------ #
    # Adaptive weights (learned continuously)
    # ------------------------------------------------------------------ #
    def weights(self) -> dict:
        """Bayesian Model Averaging (PROMPT 6).

        Each model's blend weight is its Bayesian posterior probability given the
        data: w_m is proportional to exp(log-evidence_m), where log-evidence is
        the accumulated *predictive* log-likelihood the model assigned to the
        spins that actually occurred (a prequential / discounted marginal
        likelihood). A forgetting factor lets the mix adapt if a model's edge
        changes over time -- this replaces the old softmax-of-EMA-accuracy.

        The maturity gate is retained: the trainable models (markov, lstm) only
        earn their full BMA weight as spins accumulate, so an undertrained,
        over-confident model can never hijack a cold-start prediction. Physics
        and bayes are valid from spin #1 and anchor the ensemble.
        """
        # Average ONLY over models that actually produced a prediction most
        # recently. A model that is absent (e.g. no LSTM loaded) must never win
        # BMA weight just because its neutral log-evidence (0) outranks the
        # NEGATIVE log-evidence of the models that genuinely predicted. Without
        # this guard a phantom model would dominate the reported mix.
        active = {"physics", "bayes"}
        if self.prediction_log:
            last = self.prediction_log[-1]
            active |= set(last.get("pa", last.get("picks", {})).keys())
        candidates = [m for m in self.MODELS if m in active]
        le = self.log_evidence
        mx = max(le[m] for m in candidates)
        temp = max(1e-6, self.bma_temperature)
        base = {m: math.exp((le[m] - mx) / temp) for m in candidates}
        warmup = float(max(1, getattr(settings, "ENSEMBLE_WARMUP_SPINS", 30)))
        warm = min(1.0, self.n_observed / warmup)
        gate = {"physics": 1.0, "bayes": 1.0, "markov": warm, "lstm": warm}
        gated = {m: base[m] * gate[m] for m in candidates}
        bma = _normalize(gated)
        if self.use_stacking:
            stk = self.stacking_weights()
            if stk is not None:
                blended = {m: (0.5 * bma.get(m, 0.0) + 0.5 * stk.get(m, 0.0)) * gate[m]
                           for m in candidates}
                bma = _normalize(blended)
        return {m: bma.get(m, 0.0) for m in self.MODELS}

    def stacking_weights(self, window: int = 200, iters: int = 100) -> dict:
        """Optional log-loss-optimal stacking via EM mixture-weight fitting.

        Finds weights that minimize the ensemble negative log-likelihood on
        recent spins: -sum_t log( sum_m w_m * p_m(actual_t) ), using only the
        per-model probability assigned to each realized outcome (stored in the
        prediction log). Returns None if there is not enough logged data.
        """
        entries = [e for e in self.prediction_log if e.get("pa")]
        if len(entries) < 20:
            return None
        entries = entries[-window:]
        present = set(self.MODELS)
        for e in entries:
            present &= set(e["pa"].keys())
        present = [m for m in self.MODELS if m in present]
        if len(present) < 2:
            return None
        w = {m: 1.0 / len(present) for m in present}
        for _ in range(iters):
            num = {m: 0.0 for m in present}
            for e in entries:
                pa = e["pa"]
                denom = sum(w[m] * max(_EPS, pa[m]) for m in present) or _EPS
                for m in present:
                    num[m] += (w[m] * max(_EPS, pa[m])) / denom
            total = sum(num.values()) or 1.0
            neww = {m: num[m] / total for m in present}
            if max(abs(neww[m] - w[m]) for m in present) < 1e-6:
                w = neww
                break
            w = neww
        return {m: w.get(m, 0.0) for m in self.MODELS}

    def ensemble_probabilities(self, history: list) -> dict:
        dists = self._all_distributions(history)
        w = self.weights()
        avail = {m: d for m, d in dists.items() if d is not None}
        wsum = sum(w[m] for m in avail) or 1.0
        blended = {n: 0.0 for n in self.valid_numbers}
        for m, d in avail.items():
            wm = w[m] / wsum
            for n in self.valid_numbers:
                blended[n] += wm * d[n]
        return _normalize(blended)

    def predict_next(self, history: list) -> list:
        """BasePredictor-compatible: returns [{number, confidence}] sorted desc."""
        probs = self.ensemble_probabilities(history)
        out = [{"number": n, "confidence": probs[n]} for n in self.valid_numbers]
        out.sort(key=lambda x: x["confidence"], reverse=True)
        return out

    # ------------------------------------------------------------------ #
    # Continuous learning step
    # ------------------------------------------------------------------ #
    def observe(self, actual: int, history_before: list):
        """Update model scores/weights from one confirmed spin. history_before is
        the result list BEFORE this spin (used to grade each model fairly)."""
        dists = self._all_distributions(history_before)
        picks = {}
        pa_log = {}
        for m, d in dists.items():
            if d is None:
                continue
            pick = max(d, key=d.get)
            picks[m] = pick
            hit = 1.0 if pick == actual else 0.0
            self.scores[m] = (1.0 - self.ema_lr) * self.scores[m] + self.ema_lr * hit
            pa = max(_EPS, float(d.get(actual, 0.0)))
            pa_log[m] = pa
            # discounted (forgetting) predictive log-evidence for BMA
            self.log_evidence[m] = self.bma_discount * self.log_evidence[m] + math.log(pa)
        self.n_observed += 1
        self.prediction_log.append({
            "i": self.n_observed, "actual": int(actual), "picks": picks,
            "pa": {m: round(v, 6) for m, v in pa_log.items()},
        })
        if len(self.prediction_log) > 500:
            self.prediction_log = self.prediction_log[-500:]
        self._save_state()

    # ------------------------------------------------------------------ #
    # Status (for the live-learning UI panel)
    # ------------------------------------------------------------------ #
    def learning_status(self) -> dict:
        w = self.weights()
        return {
            "n_observed": self.n_observed,
            "method": "BMA+stacking" if self.use_stacking else "BMA",
            "weights": {m: round(w[m], 4) for m in self.MODELS},
            "accuracy": {m: round(self.scores[m], 4) for m in self.MODELS},
            "log_evidence": {m: round(self.log_evidence[m], 3) for m in self.MODELS},
            "lstm_ready": bool(self.lstm_engine is not None and getattr(self.lstm_engine, "_trained", False)),
        }

    # ------------------------------------------------------------------ #
    # Persistence (continuity across launches)
    # ------------------------------------------------------------------ #
    def _state_path(self) -> str:
        rel = getattr(settings, "LEARNING_STATE_PATH", "models/continuous_state.json")
        if os.path.isabs(rel):
            return rel
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, rel)

    def _save_state(self):
        try:
            path = self._state_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "scores": self.scores,
                    "log_evidence": self.log_evidence,
                    "n_observed": self.n_observed,
                    "prediction_log": self.prediction_log[-200:],
                }, f)
        except Exception:
            pass

    def _load_state(self):
        try:
            path = self._state_path()
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            for m in self.MODELS:
                if m in state.get("scores", {}):
                    self.scores[m] = float(state["scores"][m])
                if m in state.get("log_evidence", {}):
                    self.log_evidence[m] = float(state["log_evidence"][m])
            self.n_observed = int(state.get("n_observed", 0))
            self.prediction_log = list(state.get("prediction_log", []))
        except Exception:
            pass
