# -*- coding: utf-8 -*-
"""Wheel math: the fair design distribution, payouts, EV, and a chi-square
goodness-of-fit test. Ships its own incomplete-gamma function so there is no
scipy dependency.
"""
import math

from config import VALID_NUMBERS, WHEEL_SEQUENCE, payout_multiplier


# --- incomplete gamma (chi-square p-values, no scipy) ---------------------
def _gammln(x):
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _gser(a, x, itmax=300, eps=3.0e-9):
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - _gammln(a))


def _gcf(a, x, itmax=300, eps=3.0e-9, fpmin=1.0e-30):
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - _gammln(a)) * h


def gammq(a, x):
    """Upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < 0.0 or a <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def design_distribution(sequence=None, valid_numbers=None):
    """Fair probability of each number = (its segment count) / (total segments)."""
    sequence = sequence if sequence is not None else WHEEL_SEQUENCE
    valid_numbers = valid_numbers if valid_numbers is not None else VALID_NUMBERS
    total = len(sequence)
    return {n: sequence.count(n) / total for n in valid_numbers}


def chi_square_gof(observed, expected):
    """Pearson chi-square goodness-of-fit. `observed`/`expected` are dicts keyed
    by number. Returns (chi2, dof, p_value)."""
    keys = list(expected.keys())
    chi2 = 0.0
    for k in keys:
        e = expected[k]
        o = observed.get(k, 0)
        if e > 0:
            chi2 += (o - e) ** 2 / e
    dof = max(1, len(keys) - 1)
    p = gammq(dof / 2.0, chi2 / 2.0)
    return chi2, dof, p


def breakeven_prob(n):
    """Win probability at which a bet on `n` is exactly break-even."""
    return 1.0 / (payout_multiplier(n) + 1.0)


def ev_per_token(p, n):
    """Expected value per 1 token staked on `n` given win probability `p`.
    A win returns (payout + 1) tokens (your stake back + the payout)."""
    return p * (payout_multiplier(n) + 1.0) - 1.0


def design_evs(sequence=None, valid_numbers=None):
    """EV per token for each number under the fair design distribution."""
    dist = design_distribution(sequence, valid_numbers)
    return {n: ev_per_token(dist[n], n) for n in dist}
