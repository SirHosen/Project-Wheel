# -*- coding: utf-8 -*-
"""OnlineBiasTracker: the Bayesian "brain".

It keeps a Dirichlet posterior over the wheel's true number distribution,
starting from a prior that says "the wheel is fair" (the design distribution).
Crucially, it only claims a bias or a +EV edge once it has enough evidence
(>= MIN_OBS spins) AND the lower confidence bound clears the margin. That
honesty is the whole point: on a fair wheel it will keep saying SKIP.
"""
import math

from config import CI_Z, EV_MARGIN, MIN_OBS, PRIOR_STRENGTH, VALID_NUMBERS
from core.wheel import (breakeven_prob, chi_square_gof, design_distribution,
                        ev_per_token)


class OnlineBiasTracker:
    def __init__(self, valid_numbers=None, prior_strength=PRIOR_STRENGTH,
                 ci_z=CI_Z, ev_margin=EV_MARGIN, min_obs=MIN_OBS):
        self.valid = list(valid_numbers) if valid_numbers else list(VALID_NUMBERS)
        self.prior_strength = float(prior_strength)
        self.ci_z = float(ci_z)
        self.ev_margin = float(ev_margin)
        self.min_obs = int(min_obs)
        design = design_distribution(valid_numbers=self.valid)
        # Dirichlet prior alpha_n = prior_strength * fair probability of n.
        self.prior_alpha = {n: self.prior_strength * design[n] for n in self.valid}
        self.counts = {n: 0 for n in self.valid}
        self.n = 0

    # --- ingest ---
    def observe(self, number):
        if number in self.counts:
            self.counts[number] += 1
            self.n += 1

    def observe_many(self, numbers):
        for x in numbers:
            self.observe(x)

    # --- posterior ---
    def posterior(self):
        """Return {number: {mean, lo, hi}} using the Dirichlet posterior and a
        normal approximation to each Beta marginal for the confidence band."""
        alpha0 = sum(self.prior_alpha[n] + self.counts[n] for n in self.valid)
        out = {}
        for n in self.valid:
            a = self.prior_alpha[n] + self.counts[n]
            mean = a / alpha0
            b = alpha0 - a
            var = (a * b) / (alpha0 * alpha0 * (alpha0 + 1.0))
            sd = math.sqrt(max(var, 0.0))
            lo = max(0.0, mean - self.ci_z * sd)
            hi = min(1.0, mean + self.ci_z * sd)
            out[n] = {"mean": mean, "lo": lo, "hi": hi}
        return out

    # --- edges / betting ---
    def edges(self):
        """Per-number EV rows. `is_edge` is True only with enough data AND a
        conservative (lower-bound) EV above the margin."""
        post = self.posterior()
        rows = []
        for n in self.valid:
            p = post[n]
            ev = ev_per_token(p["mean"], n)
            ev_lo = ev_per_token(p["lo"], n)
            is_edge = (self.n >= self.min_obs) and (ev_lo > self.ev_margin)
            rows.append({"number": n, "mean": p["mean"], "lo": p["lo"],
                         "ev": ev, "ev_lo": ev_lo,
                         "breakeven": breakeven_prob(n), "is_edge": is_edge})
        return rows

    def best_bet(self):
        candidates = [e for e in self.edges() if e["is_edge"]]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["ev_lo"])

    # --- bias test ---
    def bias_test(self):
        """Chi-square goodness-of-fit vs the fair design. `biased` requires both
        enough data and a small p-value."""
        dof = len(self.valid) - 1
        if self.n == 0:
            return {"biased": False, "p_value": 1.0, "chi2": 0.0, "dof": dof, "n": 0}
        design = design_distribution(valid_numbers=self.valid)
        expected = {n: design[n] * self.n for n in self.valid}
        chi2, dof, p = chi_square_gof(self.counts, expected)
        biased = (self.n >= self.min_obs) and (p < 0.05)
        return {"biased": biased, "p_value": p, "chi2": chi2, "dof": dof, "n": self.n}

    # --- human summary ---
    def summary(self):
        bb = self.best_bet()
        if bb:
            rec = f"BET {bb['number']} (EV_lo={bb['ev_lo']:.3f})"
        elif self.n < self.min_obs:
            rec = f"SKIP (need >= {self.min_obs} spins, have {self.n})"
        else:
            rec = "SKIP (no robust +EV edge)"
        return {"n": self.n, "recommendation": rec, "best_bet": bb,
                "bias": self.bias_test()}
