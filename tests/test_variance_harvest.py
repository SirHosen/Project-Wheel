# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""PROMPT 12: Variance Harvest mode (opt-in, default OFF).

Verifies the pure-Python harvest logic: tiny bets, high-multiplier-only picks,
capped pick count, EV gate bypass, random skip, and a 500-spin simulation that
shows a small steady bleed (NOT capital wipeout) with rare big-multiplier hits.
Also runs the honest 217-spin-style simulation report.
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from core import harvest

_failures = []


def check(name, cond):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")
        _failures.append(name)
    assert cond, name  # pytest: surface failures as assertion errors


def test_default_off():
    check("HARVEST_MODE_DEFAULT is False (opt-in)", settings.HARVEST_MODE_DEFAULT is False)
    for nm in ("HARVEST_MIN_CONFIDENCE", "HARVEST_TARGET_MULTIPLIERS",
               "HARVEST_TOKEN_PCT", "HARVEST_MAX_PICKS", "HARVEST_SKIP_RATE"):
        check(f"setting {nm} present", hasattr(settings, nm))


def test_picks_skip_low_multipliers():
    # 1 and 2 must be skipped even with high confidence; 5/10 are eligible.
    preds = [
        {"number": 1, "confidence": 0.9},
        {"number": 2, "confidence": 0.8},
        {"number": 5, "confidence": 0.2},
        {"number": 10, "confidence": 0.15},
    ]
    picks = harvest.harvest_picks(preds, capital=1000)
    nums = {p["number"] for p in picks}
    check("skips #1 and #2 (low multiplier)", 1 not in nums and 2 not in nums)
    check("keeps high-multiplier numbers", nums.issubset({5, 8, 10, 15, 20, 30, 40}))


def test_pick_cap_and_stake():
    preds = [{"number": n, "confidence": 0.3} for n in (5, 8, 10, 15, 20)]
    picks = harvest.harvest_picks(preds, capital=1000)
    check("at most HARVEST_MAX_PICKS picks", len(picks) <= settings.HARVEST_MAX_PICKS)
    # fixed tiny stake = round(1000 * 0.02) = 20
    check("stake is a fixed tiny size (2% of capital)",
          all(p["token_bet"] == 20 for p in picks))
    check("every pick tagged mode=harvest", all(p["mode"] == "harvest" for p in picks))


def test_min_confidence_gate():
    preds = [{"number": 10, "confidence": 0.01}]  # below 5% threshold
    picks = harvest.harvest_picks(preds, capital=1000)
    check("below-threshold confidence is dropped", len(picks) == 0)
    preds2 = [{"number": 10, "confidence": 0.06}]
    check("above-threshold confidence is kept", len(harvest.harvest_picks(preds2, 1000)) == 1)


def test_min_stake_floor():
    # tiny capital -> stake floored to 1 (never 0)
    picks = harvest.harvest_picks([{"number": 10, "confidence": 0.5}], capital=10)
    check("stake floored to >=1 on tiny capital", picks and picks[0]["token_bet"] >= 1)


def test_round_profit():
    picks = [{"number": 10, "token_bet": 20}, {"number": 5, "token_bet": 20}]
    # 10 hits: +20*10 = 200 ; 5 loses: -20 -> net 180
    check("round_profit pays multiplier on win, loses stake otherwise",
          harvest.round_profit(picks, 10) == 180)
    # nothing hits -> -40
    check("round_profit all-miss = -sum(stakes)", harvest.round_profit(picks, 1) == -40)


def test_skip_rate_statistical():
    rng = random.Random(0)
    skips = sum(harvest.should_skip_round(0.20, rng) for _ in range(5000))
    check("skip rate ~20% over 5000 draws", 0.16 < skips / 5000 < 0.24)


def _fair_conf_map():
    # true area fractions for the wheel (sums ~1)
    return {1: 0.370, 2: 0.241, 5: 0.130, 8: 0.074, 10: 0.074,
            15: 0.037, 20: 0.037, 30: 0.019, 40: 0.019}


def test_sim_500_small_bleed():
    rng = random.Random(7)
    nums = list(_fair_conf_map().keys())
    weights = list(_fair_conf_map().values())
    actuals = rng.choices(nums, weights=weights, k=500)
    res = harvest.simulate(actuals, _fair_conf_map(), starting_capital=1000, seed=7)
    print(f"    sim500: profit={res['profit']:.0f} final={res['final_capital']:.0f} "
          f"harvest_rounds={res['n_harvest_rounds']} skipped={res['n_skipped']} "
          f"win_rate={res['win_rate']*100:.1f}% big_wins={res['big_wins']} "
          f"maxDD={res['max_drawdown']:.0f} sharpe={res['sharpe']}")
    # capital must NOT be wiped out by small lottery tickets
    check("capital not wiped out (>0) after 500 spins", res["final_capital"] > 0)
    check("retains a meaningful fraction of bankroll (>40%)",
          res["final_capital"] > 0.40 * 1000)
    # ~20% of rounds skipped by design (+ any no-pick rounds)
    check("a sizeable fraction of rounds skipped", res["n_skipped"] >= 0.15 * 500)
    check("sharpe computed", res["sharpe"] is not None)


def test_sim_big_wins_captured():
    # A confidence map where a high-multiplier number clears the 5% gate, then a
    # stream where that number keeps hitting -> big wins must register.
    # (With the fair map, only 10/5 clear the gate, so we set an explicit map.)
    conf = {20: 0.10, 10: 0.08, 5: 0.06}
    actuals = [20] * 60
    res = harvest.simulate(actuals, conf, starting_capital=1000,
                           seed=1, skip_rate=0.0)
    check("big multiplier hits captured", res["big_wins"] > 0)
    check("profit positive when big multipliers keep hitting", res["profit"] > 0)


if __name__ == "__main__":
    print("== PROMPT 12: Variance Harvest mode ==")
    test_default_off()
    test_picks_skip_low_multipliers()
    test_pick_cap_and_stake()
    test_min_confidence_gate()
    test_min_stake_floor()
    test_round_profit()
    test_skip_rate_statistical()
    test_sim_500_small_bleed()
    test_sim_big_wins_captured()
    if _failures:
        print(f"\nFAILED ({len(_failures)}): {_failures}")
        sys.exit(1)
    print("\nALL PASSED")
