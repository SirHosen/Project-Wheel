# -*- coding: utf-8 -*-
"""
predictors/bayesian_optimal.py — The provably-optimal predictor for this wheel.

WHY THIS IS "THE BEST" (and why it is NOT a bigger neural net):
--------------------------------------------------------------
A spin wheel's result is an i.i.d. draw from a fixed (but unknown) categorical
distribution over VALID_NUMBERS. For that data-generating process the Bayes-
optimal next-step predictor is *closed form*: the posterior-predictive of a
Dirichlet-Multinomial model. No LSTM / Markov / "pattern" model can beat it,
because there is no sequential signal to exploit -- the user's own 235-spin logs
showed the LSTM (27%) losing badly to the simple frequency baseline (44%).
Bigger models here only overfit noise. The optimal move is to estimate the true
frequencies as tightly as possible and quantify our uncertainty honestly.

WHAT IT ADDS OVER A PLAIN FREQUENCY COUNT:
  1. Dirichlet prior seeded from the physical wheel layout => sane from spin #1,
     converges to the *real* frequencies you actually observe.
  2. Per-number Beta credible intervals => we know how sure we are.
  3. EV / edge engine: every number has a payout. A bet is only profitable if
         p * (payout + 1) - 1 > 0.
     We flag a number as a REAL, actionable edge ONLY when the *lower* credible
     bound clears break-even -- i.e. even our pessimistic estimate says +EV.
     This is the single honest path to "grinding" tokens: it will bet big the
     moment a genuine wheel bias appears, and correctly SKIP on a fair wheel
     (where every bet is -EV) instead of bleeding tokens on noisy spikes.

This is the difference between "predict the most likely number" (frequency) and
"find the bet that actually makes money" (this engine).
"""

import json
import math
import os
from collections import Counter

from .base import BasePredictor
from config import settings
from core.betting import net_multiplier, ev_per_token, kelly_fraction_for


class BayesianOptimalEngine(BasePredictor):
    """Dirichlet-Multinomial posterior predictor + statistically-gated EV engine."""

    def __init__(self, prior_strength: float = None, ci_z: float = None,
                 wheel_prior_path: str = None):
        self.valid_numbers = list(settings.VALID_NUMBERS)

        # Frequency prior from the physical wheel layout (segment area fraction).
        seq = settings.SPINWHEEL_SEQUENCE
        counts = Counter(seq)
        total = float(len(seq))
        self.prior = {n: counts.get(n, 0) / total for n in self.valid_numbers}

        # prior_strength = how many pseudo-observations the layout prior is worth.
        # One physical wheel (54 segments) keeps cold-start sane without
        # drowning out real data after a few dozen spins.
        self.prior_strength = float(
            prior_strength if prior_strength is not None
            else getattr(settings, "BAYES_OPT_PRIOR_STRENGTH", len(seq))
        )
        # z for credible interval (1.96 ~ 95%).
        self.ci_z = float(ci_z if ci_z is not None else getattr(settings, "BAYES_OPT_CI_Z", 1.96))
        self.ev_margin = float(getattr(settings, "BAYES_OPT_EV_MARGIN", 0.05))
        self.min_obs = int(getattr(settings, "BAYES_OPT_MIN_OBS", 25))

        # Camera-observed physical spins (vision learning loop). These are REAL
        # i.i.d. draws from the wheel, so they enter the Dirichlet posterior as
        # extra observed counts -- they do NOT touch betting win-rate stats.
        # Missing/corrupt file => no extra evidence (graceful, backward compat).
        self.wheel_prior_path = wheel_prior_path
        self.observed_counts = self._load_observed_counts()

    # ------------------------------------------------------------------ #
    # Vision learning loop: fold camera-observed spins into the posterior
    # ------------------------------------------------------------------ #
    def _default_wheel_prior_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "models", "wheel_prior.json")

    def _load_observed_counts(self) -> dict:
        counts = {n: 0 for n in self.valid_numbers}
        path = self.wheel_prior_path or self._default_wheel_prior_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in (data.get("counts") or {}).items():
                    try:
                        num = int(k)
                    except (TypeError, ValueError):
                        continue
                    if num in counts:
                        counts[num] += int(v)
        except Exception:
            pass  # corrupt/unreadable => behave exactly as before (no extra data)
        return counts

    def reload_observed_counts(self):
        """Re-read the learned wheel prior (call after learn_from_vision.py)."""
        self.observed_counts = self._load_observed_counts()
        return self.observed_counts

    # ------------------------------------------------------------------ #
    # Posterior
    # ------------------------------------------------------------------ #
    def _posterior(self, history: list) -> dict:
        """Return per-number dict with mean, ci_low, ci_high, alpha, beta, n."""
        obs = Counter(x for x in history if x in self.prior)
        # Camera-observed physical spins count as real evidence too.
        cam = self.observed_counts
        n = int(sum(obs.values())) + int(sum(cam.values()))
        # Dirichlet concentration: alpha_n = prior_strength * prior[n]
        #                                   + camera_count[n] + betting_count[n]
        alpha = {
            k: self.prior_strength * self.prior[k] + cam.get(k, 0) + obs.get(k, 0)
            for k in self.valid_numbers
        }
        a0 = float(sum(alpha.values()))
        out = {}
        for k in self.valid_numbers:
            a = alpha[k]
            b = a0 - a  # Beta marginal of a Dirichlet component
            mean = a / a0 if a0 > 0 else 0.0
            var = (a * b) / (a0 * a0 * (a0 + 1.0)) if a0 > 0 else 0.0
            sd = math.sqrt(max(0.0, var))
            out[k] = {
                "mean": mean,
                "sd": sd,
                "ci_low": max(0.0, mean - self.ci_z * sd),
                "ci_high": min(1.0, mean + self.ci_z * sd),
                "n": n,
            }
        return out

    # ------------------------------------------------------------------ #
    # BasePredictor API: posterior-mean probabilities (optimal accuracy)
    # ------------------------------------------------------------------ #
    def predict_next(self, history: list) -> list:
        post = self._posterior(history or [])
        n = post[self.valid_numbers[0]]["n"] if self.valid_numbers else 0
        preds = []
        for k in self.valid_numbers:
            preds.append({
                "number": k,
                "confidence": post[k]["mean"],
                "ci_low": post[k]["ci_low"],
                "ci_high": post[k]["ci_high"],
                # support gates the shared betting layer; the posterior mean is
                # only trustworthy once we have a few dozen real observations.
                "support": n,
            })
        preds.sort(key=lambda x: x["confidence"], reverse=True)
        return preds

    # ------------------------------------------------------------------ #
    # Edge / EV engine (the "grinding" brain)
    # ------------------------------------------------------------------ #
    def edge_report(self, history: list) -> list:
        """For every number, compute EV at the posterior mean AND at the lower
        credible bound. `robust_positive` means even the pessimistic estimate is
        profitable => a real, bettable edge. Sorted by conservative EV desc."""
        post = self._posterior(history or [])
        n = post[self.valid_numbers[0]]["n"] if self.valid_numbers else 0
        rows = []
        for k in self.valid_numbers:
            m = net_multiplier(k)
            mean = post[k]["mean"]
            lo = post[k]["ci_low"]
            hi = post[k]["ci_high"]
            ev_mean = ev_per_token(mean, k)
            ev_cons = ev_per_token(lo, k)
            ev_opt = ev_per_token(hi, k)
            breakeven = 1.0 / (m + 1)
            robust = (n >= self.min_obs) and (ev_cons > self.ev_margin)
            rows.append({
                "number": k,
                "payout": m,
                "prob_mean": mean,
                "prob_low": lo,
                "prob_high": hi,
                "breakeven_prob": breakeven,
                "ev_mean": ev_mean,
                "ev_conservative": ev_cons,
                "ev_optimistic": ev_opt,
                "robust_positive": robust,
                "n": n,
            })
        rows.sort(key=lambda r: r["ev_conservative"], reverse=True)
        return rows

    def recommend(self, history, capital, risk_pct, kelly_fraction=0.5, max_bets=3):
        """Return UI-compatible allocations. Stakes ONLY on numbers whose lower
        credible bound is robustly +EV (sized with conservative fractional
        Kelly). On a fair wheel nothing clears the bar -> SKIP (token_bet 0).
        """
        capital = max(0, int(capital))
        budget = max(0, int(capital * float(risk_pct)))
        report = self.edge_report(history or [])

        actionable = [r for r in report if r["robust_positive"]]
        # Size with Kelly computed on the CONSERVATIVE probability (ci_low).
        for r in actionable:
            r["_kelly"] = max(0.0, kelly_fraction_for(r["prob_low"], r["number"]))
        actionable = [r for r in actionable if r["_kelly"] > 0]
        actionable.sort(key=lambda r: r["_kelly"], reverse=True)
        actionable = actionable[:max_bets]

        if actionable and budget > 0:
            raws = [capital * kelly_fraction * r["_kelly"] for r in actionable]
            total_raw = sum(raws) or 1.0
            scale = min(1.0, budget / total_raw)
            result = []
            for r, raw in zip(actionable, raws):
                result.append(self._alloc_row(r, max(0, int(raw * scale))))
            if result and all(x["token_bet"] == 0 for x in result):
                result[0]["token_bet"] = 1
            result = [x for x in result if x["token_bet"] > 0]
            if result:
                return result

        # SKIP recommendation: show the most likely numbers with full EV info.
        top = sorted(report, key=lambda r: r["prob_mean"], reverse=True)[:max_bets]
        return [self._alloc_row(r, 0) for r in top]

    @staticmethod
    def _alloc_row(r, token_bet):
        return {
            "number": r["number"],
            "confidence": r["prob_mean"],
            "ev_per_token": r["ev_mean"],
            "ev_conservative": r["ev_conservative"],
            "ci_low": r["prob_low"],
            "ci_high": r["prob_high"],
            "is_positive_ev": bool(r["robust_positive"]),
            "support": r["n"],
            "token_bet": int(token_bet),
        }
