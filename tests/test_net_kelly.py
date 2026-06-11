# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Test PROMPT 7: correlation-aware net-Kelly portfolio (mutually-exclusive).

The CORRECT optimality property is NOT "net stakes fewer tokens" -- betting on
several mutually-exclusive +EV outcomes can be good diversification. The right
invariant is that the net-Kelly portfolio achieves expected LOG-GROWTH at least
as high as the naive independent per-number sizing, evaluated under the true
mutually-exclusive payout model (only one number wins, all other stakes lost).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.betting import net_kelly_portfolio, kelly_allocation, net_multiplier


def _total(allocs):
    return sum(a["token_bet"] for a in allocs)


def _expected_log_growth(allocs, preds, W):
    """True log-growth under mutual exclusivity: exactly one number wins."""
    bets = {a["number"]: a["token_bet"] for a in allocs}
    staked = sum(bets.values())
    prob = {int(p["number"]): float(p["confidence"]) for p in preds}
    g = 0.0
    for o, p_o in prob.items():
        if p_o <= 0:
            continue
        profit = bets.get(o, 0) * net_multiplier(o) - (staked - bets.get(o, 0))
        ratio = 1.0 + profit / W
        if ratio <= 0:
            return float("-inf")
        g += p_o * math.log(ratio)
    return g


def test_beats_independent_kelly_in_log_growth():
    # Two simultaneously +EV numbers under full Kelly (kf=1.0), uncapped budget.
    preds = [
        {"number": 2, "confidence": 0.50},   # m=2 -> EV = 0.50
        {"number": 5, "confidence": 0.25},   # m=5 -> EV = 0.50
        {"number": 1, "confidence": 0.25},   # filler (-EV, m=1)
    ]
    cap, risk = 1000, 0.50
    net = net_kelly_portfolio(preds, cap, risk, kelly_fraction=1.0)
    ind = kelly_allocation(preds, cap, risk, kelly_fraction=1.0)
    g_net = _expected_log_growth(net, preds, cap)
    g_ind = _expected_log_growth(ind, preds, cap)
    assert _total(net) > 0
    assert g_net >= g_ind - 1e-9, (g_net, g_ind, _total(net), _total(ind))
    print(f"OK net-Kelly log-growth {g_net:.6f} >= independent {g_ind:.6f} "
          f"(net={_total(net)}tok, indep={_total(ind)}tok)")


def test_fractional_kelly_reduces_stake():
    # Smaller kelly_fraction must stake fewer (or equal) tokens than full Kelly.
    preds = [{"number": 2, "confidence": 0.60}, {"number": 1, "confidence": 0.40}]
    full = _total(net_kelly_portfolio(preds, 2000, 0.9, kelly_fraction=1.0))
    half = _total(net_kelly_portfolio(preds, 2000, 0.9, kelly_fraction=0.5))
    assert 0 < half < full, (half, full)
    print(f"OK fractional knob works: half-Kelly={half} < full-Kelly={full} tokens")


def test_budget_is_respected_under_strong_edge():
    preds = [{"number": 2, "confidence": 0.80}, {"number": 1, "confidence": 0.20}]
    cap, risk = 5000, 0.50
    tot = _total(net_kelly_portfolio(preds, cap, risk))
    assert 0 < tot <= int(cap * risk)
    print(f"OK budget respected: staked {tot} of budget {int(cap*risk)}")


def test_skip_when_no_edge():
    preds = [
        {"number": 1, "confidence": 0.37},
        {"number": 2, "confidence": 0.24},
        {"number": 5, "confidence": 0.13},
        {"number": 8, "confidence": 0.07},
    ]
    net = net_kelly_portfolio(preds, 1000, 0.30)
    assert _total(net) == 0, net
    assert all(not a["is_positive_ev"] for a in net)
    print("OK no-edge -> SKIP (all stakes 0)")


def test_evidence_gate_blocks_unsupported():
    preds = [{"number": 5, "confidence": 0.40, "support": 2}]
    net = net_kelly_portfolio(preds, 1000, 0.30, min_support=10)
    assert _total(net) == 0
    print("OK evidence gate blocks thin-support +EV bet")


def test_single_edge_stakes():
    preds = [{"number": 2, "confidence": 0.60}, {"number": 1, "confidence": 0.40}]
    net = net_kelly_portfolio(preds, 1000, 0.30)
    staked = [a for a in net if a["token_bet"] > 0]
    assert len(staked) == 1 and staked[0]["number"] == 2
    print(f"OK single +EV edge staked: {staked[0]['token_bet']} tokens on 2")


if __name__ == "__main__":
    test_beats_independent_kelly_in_log_growth()
    test_fractional_kelly_reduces_stake()
    test_budget_is_respected_under_strong_edge()
    test_skip_when_no_edge()
    test_evidence_gate_blocks_unsupported()
    test_single_edge_stakes()
    print("\nALL PROMPT 7 TESTS PASSED")
