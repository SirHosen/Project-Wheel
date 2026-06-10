# -*- coding: utf-8 -*-
"""Session detection & temporal-drift statistics (PROMPT 9).

A "session" is a run of spins with no large idle gap between them. Real wheels
can behave differently across sessions (operator change, mechanical drift,
different table), so aggregating every spin into one linear pool can hide a
real change. This module splits history into sessions and tests whether the
outcome distribution drifted across them.

No scipy dependency: chi-square GOF and the 2-sample Kolmogorov-Smirnov test
(with the asymptotic p-value) are implemented from scratch.
"""
import bisect
import math
from collections import Counter
from datetime import datetime


def parse_ts(s):
    """Parse an ISO-8601 timestamp; return None on failure."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.fromisoformat(str(s).split(".")[0])
        except Exception:
            return None


def expected_probs(valid_numbers, sequence):
    """Theoretical wheel probabilities from the physical segment layout."""
    if sequence:
        c = Counter(sequence)
        tot = sum(c.values())
        if tot > 0:
            return {n: c.get(n, 0) / tot for n in valid_numbers}
    if valid_numbers:
        p = 1.0 / len(valid_numbers)
        return {n: p for n in valid_numbers}
    return {}


def chi_square_gof(actuals, valid_numbers, sequence):
    """Pearson chi-square goodness-of-fit vs the fair-wheel distribution."""
    if not actuals or not valid_numbers:
        return None
    exp = expected_probs(valid_numbers, sequence)
    n = len(actuals)
    obs = Counter(actuals)
    chi = 0.0
    for num in valid_numbers:
        e = exp.get(num, 0.0) * n
        if e > 0:
            chi += (obs.get(num, 0) - e) ** 2 / e
    return chi


def _ks_pvalue(d, n, m):
    """Asymptotic 2-sample KS p-value via the Kolmogorov distribution."""
    if n == 0 or m == 0:
        return 1.0
    en = math.sqrt(n * m / float(n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam <= 0:
        return 1.0
    s = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
        s += term
        if abs(term) < 1e-10:
            break
    return max(0.0, min(1.0, s))


def ks_2sample(a, b):
    """Two-sample Kolmogorov-Smirnov test. Returns (D statistic, p_value)."""
    if not a or not b:
        return (0.0, 1.0)
    sa = sorted(a)
    sb = sorted(b)
    na, nb = len(sa), len(sb)
    d = 0.0
    for v in sorted(set(sa) | set(sb)):
        ca = bisect.bisect_right(sa, v) / na
        cb = bisect.bisect_right(sb, v) / nb
        d = max(d, abs(ca - cb))
    return (d, _ks_pvalue(d, na, nb))


def detect_session_drift(session_actuals, alpha=0.05):
    """KS-test the pooled first half of sessions vs the second half.

    session_actuals: list of per-session lists of actual numbers.
    Returns dict with drift verdict, KS D, p_value.
    """
    sessions = [s for s in session_actuals if s]
    if len(sessions) < 2:
        return {"drift": False, "ks_d": None, "p_value": None,
                "alpha": alpha, "reason": "butuh >=2 sesi dengan data"}
    mid = len(sessions) // 2
    first = [x for s in sessions[:mid] for x in s] or sessions[0]
    second = [x for s in sessions[mid:] for x in s] or sessions[-1]
    d, p = ks_2sample(first, second)
    return {"drift": bool(p < alpha), "ks_d": d, "p_value": p, "alpha": alpha,
            "n_first": len(first), "n_second": len(second)}


def recency_weights(n, half_life=50.0):
    """Exponential-decay weights for n ordered records (oldest..newest).

    The most recent record gets weight 1.0; each step back halves every
    `half_life` records. Useful to let an engine trust a recent session more.
    Returns a list of length n summing-free weights (not normalized).
    """
    if n <= 0:
        return []
    if half_life <= 0:
        return [1.0] * n
    decay = math.log(2.0) / half_life
    # index 0 = oldest, n-1 = newest -> age = (n-1 - i)
    return [math.exp(-decay * ((n - 1) - i)) for i in range(n)]
