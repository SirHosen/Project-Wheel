# -*- coding: utf-8 -*-
"""EV-aware bet sizing (the profit engine).

Instead of blindly staking the top-3 predictions, this turns the engine's
predicted probabilities into bet sizes that maximize EXPECTED PROFIT.

Payout model (see config.settings.calculate_reward):
    net win per token on number n = n   (n >= 2)
                                  = 1   (n == 1)
    loss per token                = 1

For a predicted probability p on number n, with m = net multiplier:
    EV per token  = p * m - (1 - p) = p * (m + 1) - 1
    break-even p* = 1 / (m + 1)

We stake ONLY on numbers that clear an EV margin AND are backed by enough
evidence, sizing each stake with fractional Kelly:
    kelly(n) = p - (1 - p) / m      (fraction of bankroll)
clamped to >= 0, scaled by `kelly_fraction` (half-Kelly by default for safety),
then the whole set is capped to the user's risk budget (risk_pct * capital).

Evidence gate (added after analysing real game logs): a prediction may carry a
`support` field = how many observations back the current conditional estimate
(e.g. Markov transition count from the last number). Confident-looking spikes
from tiny samples are the #1 cause of false +EV bets that bleed the bankroll,
so a candidate must have `support >= min_support` before we will stake on it.
Predictions with no `support` key (e.g. other engines) are not gated.

If nothing clears the bar, we recommend SKIPPING the round (all stakes = 0). On
a fair wheel every bet is -EV, so skipping is the profit-maximizing move until a
real, well-supported bias pushes some number above break-even.
"""


def net_multiplier(number: int) -> int:
    """Net token profit per token staked, on a win."""
    return 1 if number == 1 else number


def ev_per_token(prob: float, number: int) -> float:
    m = net_multiplier(number)
    return prob * (m + 1) - 1.0


def kelly_fraction_for(prob: float, number: int) -> float:
    m = net_multiplier(number)
    return prob - (1.0 - prob) / m


def kelly_allocation(predictions, capital, risk_pct, kelly_fraction=0.5, max_bets=3,
                     min_support=10, ev_margin=0.10):
    """Return EV-aware, evidence-gated allocations.

    predictions: list of {"number", "confidence", optional "support"}.
    Returns a list of dicts with keys:
        number, confidence, ev_per_token, is_positive_ev, support, token_bet
    where `is_positive_ev` means ACTIONABLE (cleared EV margin + evidence gate).
    - If at least one actionable bet exists: returns those (token_bet > 0),
      sorted by stake desc, capped to `max_bets` and the risk budget.
    - Otherwise: returns the top-`max_bets` by confidence with token_bet = 0
      (a SKIP recommendation), still carrying EV info for transparency.
    """
    capital = max(0, int(capital))
    budget = max(0, int(capital * risk_pct))

    enriched = []
    for p in predictions:
        n = p["number"]
        conf = max(0.0, min(1.0, float(p["confidence"])))
        ev = ev_per_token(conf, n)
        kelly = max(0.0, kelly_fraction_for(conf, n))
        support = p.get("support")
        has_evidence = (support is None) or (support >= min_support)
        actionable = (ev > ev_margin) and has_evidence and kelly > 0
        enriched.append({
            "number": n,
            "confidence": conf,
            "ev_per_token": ev,
            "is_positive_ev": actionable,
            "support": support,
            "kelly": kelly,
        })

    positive = [e for e in enriched if e["is_positive_ev"]]
    positive.sort(key=lambda e: e["kelly"], reverse=True)
    positive = positive[:max_bets]

    if positive and budget > 0:
        raws = [capital * kelly_fraction * e["kelly"] for e in positive]
        total_raw = sum(raws)
        scale = min(1.0, budget / total_raw) if total_raw > 0 else 0.0
        result = []
        for e, raw in zip(positive, raws):
            result.append({
                "number": e["number"],
                "confidence": e["confidence"],
                "ev_per_token": e["ev_per_token"],
                "is_positive_ev": True,
                "support": e["support"],
                "token_bet": max(0, int(raw * scale)),
            })
        # If rounding zeroed everything but we do have an edge, stake 1 on the best.
        if result and all(r["token_bet"] == 0 for r in result):
            result[0]["token_bet"] = 1
        result = [r for r in result if r["token_bet"] > 0]
        if result:
            return result

    # Nothing actionable -> recommend SKIP (stake 0), show top by confidence.
    out = []
    for e in sorted(enriched, key=lambda x: x["confidence"], reverse=True)[:max_bets]:
        out.append({
            "number": e["number"],
            "confidence": e["confidence"],
            "ev_per_token": e["ev_per_token"],
            "is_positive_ev": e["is_positive_ev"],
            "support": e["support"],
            "token_bet": 0,
        })
    return out


import math as _math


def net_kelly_portfolio(predictions, capital, risk_pct, kelly_fraction=0.5,
                        max_bets=3, min_support=10, ev_margin=0.10):
    """Correlation-aware net-Kelly allocation for MUTUALLY-EXCLUSIVE outcomes.

    On a spin wheel exactly ONE number wins per spin, so stakes are negatively
    correlated: every token staked on A is lost whenever B wins. The legacy
    ``kelly_allocation`` sizes each number with single-bet Kelly and then merely
    rescales to the risk budget -- it ignores this correlation and therefore
    OVER-stakes when betting several numbers at once (it double-counts capital
    that can only ever win on one outcome).

    This allocator instead maximises expected LOG growth of the bankroll over
    the FULL predicted outcome distribution, which exactly encodes mutual
    exclusivity. If the realised number is ``o`` and we staked b_k tokens on
    each number k, the bankroll change is::

        profit(o) = b_o * net_multiplier(o) - sum_{k != o} b_k

    and we maximise  sum_o p_o * log(1 + profit(o) / W).  Because that objective
    is concave in integer token steps, a greedy marginal-gain search (add one
    token to whichever number most improves expected log-growth) converges to a
    near-optimal INTEGER allocation. Crucially the search STOPS as soon as no
    single extra token helps -- so over-betting correlated outcomes is impossible
    even with a large risk budget.

    Fractional Kelly is applied by scaling the effective bankroll used in the
    growth objective (W = capital * kelly_fraction). A SMALLER effective
    bankroll makes each token represent a larger risked fraction, so the optimum
    halts at fewer tokens -- reproducing the safer fractional-Kelly size (at the
    single-bet limit this matches the legacy half-Kelly sizing exactly) without
    distorting the relative allocation across numbers.

    Same EV + evidence gate as ``kelly_allocation``: a number is eligible only
    if it clears ``ev_margin`` AND (support is None or support >= min_support).
    Returns the same dict shape: number, confidence, ev_per_token,
    is_positive_ev, support, token_bet. If nothing is eligible -> SKIP (all
    token_bet = 0, top by confidence).
    """
    capital = max(0, int(capital))
    budget = max(0, int(capital * risk_pct))
    kf = min(1.0, max(1e-6, float(kelly_fraction)))

    # Probability over ALL predicted numbers (needed because numbers we do NOT
    # bet on still cause a loss of every stake when they win).
    prob = {}
    for p in predictions:
        n = int(p["number"])
        prob[n] = max(0.0, min(1.0, float(p["confidence"])))

    # Eligible numbers: +EV beyond margin AND evidence-backed.
    enriched = []
    for p in predictions:
        n = int(p["number"])
        conf = max(0.0, min(1.0, float(p["confidence"])))
        ev = ev_per_token(conf, n)
        support = p.get("support")
        has_evidence = (support is None) or (support >= min_support)
        eligible = (ev > ev_margin) and has_evidence
        enriched.append({
            "number": n, "confidence": conf, "ev_per_token": ev,
            "support": support, "eligible": eligible,
        })

    candidates = [e for e in enriched if e["eligible"]]
    # Keep the strongest-EV candidates (cap the simultaneous bet count).
    candidates.sort(key=lambda e: e["ev_per_token"], reverse=True)
    candidates = candidates[:max(0, int(max_bets))]
    bet_numbers = [e["number"] for e in candidates]

    def expected_log_growth(bets, staked, W):
        g = 0.0
        for o, p_o in prob.items():
            if p_o <= 0.0:
                continue
            won = bets.get(o, 0) * net_multiplier(o)
            profit = won - (staked - bets.get(o, 0))
            ratio = 1.0 + profit / W
            if ratio <= 0.0:
                return float("-inf")  # would risk ruin on this outcome
            g += p_o * _math.log(ratio)
        return g

    if bet_numbers and budget > 0 and capital > 0:
        W = capital * kf  # deflated bankroll -> fractional-Kelly sizing
        bets = {n: 0 for n in bet_numbers}
        staked = 0
        cur = expected_log_growth(bets, staked, W)  # all-zero baseline (= 0.0)
        # Greedy halts at the Kelly optimum; cap iterations for safety only.
        max_iter = min(budget, 100000)
        for _ in range(max_iter):
            if staked >= budget:
                break
            best_n, best_val, best_gain = None, None, 1e-12
            for n in bet_numbers:
                bets[n] += 1
                val = expected_log_growth(bets, staked + 1, W)
                bets[n] -= 1
                gain = val - cur
                if gain > best_gain:
                    best_gain, best_val, best_n = gain, val, n
            if best_n is None:
                break  # no extra token improves growth -> optimal
            bets[best_n] += 1
            staked += 1
            cur = best_val

        result = []
        for e in candidates:
            tb = bets.get(e["number"], 0)
            if tb > 0:
                result.append({
                    "number": e["number"], "confidence": e["confidence"],
                    "ev_per_token": e["ev_per_token"], "is_positive_ev": True,
                    "support": e["support"], "token_bet": int(tb),
                })
        if result:
            result.sort(key=lambda r: r["token_bet"], reverse=True)
            return result

    # Nothing actionable -> SKIP recommendation (stake 0), top by confidence.
    out = []
    for e in sorted(enriched, key=lambda x: x["confidence"], reverse=True)[:max_bets]:
        out.append({
            "number": e["number"], "confidence": e["confidence"],
            "ev_per_token": e["ev_per_token"], "is_positive_ev": False,
            "support": e["support"], "token_bet": 0,
        })
    return out
