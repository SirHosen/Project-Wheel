# -*- coding: utf-8 -*-
"""OnlineBiasTracker: the Bayesian "brain".

It keeps a Dirichlet posterior over the wheel's true number distribution,
starting from a prior that says "the wheel is fair" (the design distribution).
Crucially, it only claims a bias or a +EV edge once it has enough evidence
(>= MIN_OBS spins) AND the lower confidence bound clears the margin. That
honesty is the whole point: on a fair wheel it will keep saying SKIP.

Because we search all 9 numbers for an edge at once, the per-number test is
multiple-testing corrected (Sidak by default) so the family-wide false-positive
rate stays near EDGE_FAMILY_ALPHA instead of ~1-in-2 per session.
"""
from config import (CI_Z, EDGE_FAMILY_ALPHA, EV_MARGIN, MIN_OBS,
                    MULTIPLE_TEST_CORRECTION, PRIOR_STRENGTH, VALID_NUMBERS,
                    payout_multiplier)
from core.wheel import (beta_quantile, breakeven_prob, chi_square_gof,
                        design_distribution, ev_per_token, normal_cdf)


class OnlineBiasTracker:
    def __init__(self, valid_numbers=None, prior_strength=PRIOR_STRENGTH,
                 ci_z=CI_Z, ev_margin=EV_MARGIN, min_obs=MIN_OBS,
                 family_alpha=EDGE_FAMILY_ALPHA,
                 mt_correction=MULTIPLE_TEST_CORRECTION):
        self.valid = list(valid_numbers) if valid_numbers else list(VALID_NUMBERS)
        self.prior_strength = float(prior_strength)
        self.ci_z = float(ci_z)
        self.ev_margin = float(ev_margin)
        self.min_obs = int(min_obs)
        self.family_alpha = float(family_alpha)
        self.mt_correction = (mt_correction or "none").lower()
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
        """Return {number: {mean, lo, hi}} from the Dirichlet posterior.

        Each number's marginal is exactly Beta(a, alpha0 - a), so the credible
        band uses EXACT Beta quantiles instead of a normal approximation. This
        matters at the small sample sizes and near the [0, 1] edges where the
        normal approx is worst (and where a spurious +EV 'edge' could sneak in).
        The z-score `ci_z` (e.g. 1.96) is mapped to a two-sided credible mass
        via the normal CDF (1.96 -> ~95%).
        """
        alpha0 = sum(self.prior_alpha[n] + self.counts[n] for n in self.valid)
        upper_p = normal_cdf(self.ci_z)   # e.g. 1.96 -> 0.975
        lower_p = 1.0 - upper_p           #            -> 0.025
        out = {}
        for n in self.valid:
            a = self.prior_alpha[n] + self.counts[n]
            b = alpha0 - a
            mean = a / alpha0
            lo = beta_quantile(a, b, lower_p)
            hi = beta_quantile(a, b, upper_p)
            out[n] = {"mean": mean, "lo": lo, "hi": hi}
        return out

    # --- multiple-testing correction ---
    def _edge_tail(self):
        """Per-number ONE-SIDED lower-tail probability for the edge test, after
        correcting for searching all K numbers at once. Smaller = stricter.

          none        -> family_alpha (no correction)
          bonferroni  -> family_alpha / K
          sidak       -> 1 - (1 - family_alpha)^(1/K)
        """
        k = len(self.valid)
        base = self.family_alpha
        if self.mt_correction == "bonferroni":
            return base / k
        if self.mt_correction == "sidak":
            return 1.0 - (1.0 - base) ** (1.0 / k)
        return base

    # --- edges / betting ---
    def edges(self):
        """Per-number EV rows. `is_edge` is True only with enough data AND a
        conservative, multiple-testing-corrected lower-bound EV above the
        margin. `lo` is the corrected one-sided lower credible bound on the
        number's probability; `tail` is the per-number alpha used."""
        alpha0 = sum(self.prior_alpha[n] + self.counts[n] for n in self.valid)
        tail = self._edge_tail()
        rows = []
        for n in self.valid:
            a = self.prior_alpha[n] + self.counts[n]
            b = alpha0 - a
            mean = a / alpha0
            lo = beta_quantile(a, b, tail)
            ev = ev_per_token(mean, n)
            ev_lo = ev_per_token(lo, n)
            is_edge = (self.n >= self.min_obs) and (ev_lo > self.ev_margin)
            rows.append({"number": n, "mean": mean, "lo": lo, "tail": tail,
                         "ev": ev, "ev_lo": ev_lo,
                         "breakeven": breakeven_prob(n), "is_edge": is_edge})
        return rows

    def best_bet(self):
        candidates = [e for e in self.edges() if e["is_edge"]]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["ev_lo"])

    # --- fully Bayesian EV (Monte-Carlo over the Dirichlet posterior) ---
    def ev_samples(self, n_samples=4000, seed=0):
        """Monte-Carlo the EV of a 1-token bet on each number by drawing the
        whole probability vector from the Dirichlet posterior.

        Returns {number: {ev_mean, ev_lo, ev_hi, prob_positive}} where
        `prob_positive` = P(EV > 0) under the posterior. This is more honest
        than plugging a single lower confidence bound into the EV formula: it
        reports the FULL uncertainty of the edge, so you can require, say,
        P(EV > 0) >= 0.95 before ever betting. numpy is imported lazily so the
        analytic methods still work if numpy is somehow unavailable.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        alpha = np.array([self.prior_alpha[n] + self.counts[n]
                          for n in self.valid], dtype=float)
        draws = rng.dirichlet(alpha, size=int(n_samples))   # (S, K)
        upper_p = normal_cdf(self.ci_z)
        lo_q, hi_q = 1.0 - upper_p, upper_p
        out = {}
        for i, n in enumerate(self.valid):
            p = draws[:, i]
            ev = p * (payout_multiplier(n) + 1.0) - 1.0
            lo, hi = (float(v) for v in np.quantile(ev, [lo_q, hi_q]))
            out[n] = {"ev_mean": float(ev.mean()), "ev_lo": lo, "ev_hi": hi,
                      "prob_positive": float((ev > 0.0).mean())}
        return out

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
