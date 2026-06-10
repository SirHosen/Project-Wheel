# -*- coding: utf-8 -*-
"""core/harvest.py - Variance Harvest mode (OPT-IN, default OFF).

HONEST FRAMING (read this before trusting the mode):
  Every number on this wheel is long-run NEGATIVE expected value. Variance
  Harvest does NOT change that. It deliberately places TINY 'lottery ticket'
  stakes on high-multiplier numbers, accepting a small steady bleed in exchange
  for rare high-variance payoffs. Over many rounds you should EXPECT to lose
  slowly; any profit is luck (variance), not edge. This mode exists because the
  conservative engine correctly SKIPs everything, and some users explicitly
  want controlled lottery exposure. It is gated, capped, and OFF by default.

The sizing/selection/skip logic here is pure-Python so it can be unit-tested
without a GUI or TensorFlow.
"""
import math
import random

from core.betting import net_multiplier
from config import settings


def _cfg(name, default):
    return getattr(settings, name, default)


def harvest_picks(predictions, capital,
                  target_multipliers=None, min_confidence=None,
                  token_pct=None, max_picks=None):
    """Choose small lottery-ticket stakes for Variance Harvest mode.

    Rules (all configurable via settings):
      * Only numbers whose multiplier is in HARVEST_TARGET_MULTIPLIERS
        (skip #1/#2 -- payouts too small to be worth the variance).
      * Keep only confidence >= HARVEST_MIN_CONFIDENCE (bypasses the EV gate).
      * Stake a FIXED tiny size = round(capital * HARVEST_TOKEN_PCT), min 1.
      * At most HARVEST_MAX_PICKS numbers, ranked by lottery value
        (confidence * multiplier).

    Returns a list of alloc dicts compatible with the prediction cards:
      {number, token_bet, confidence, multiplier, ev_per_token,
       is_positive_ev, support, mode:"harvest"}
    """
    target = set(target_multipliers if target_multipliers is not None
                 else _cfg("HARVEST_TARGET_MULTIPLIERS", [5, 8, 10, 15, 20, 30, 40]))
    min_conf = float(min_confidence if min_confidence is not None
                     else _cfg("HARVEST_MIN_CONFIDENCE", 0.05))
    tok_pct = float(token_pct if token_pct is not None
                    else _cfg("HARVEST_TOKEN_PCT", 0.02))
    picks_cap = int(max_picks if max_picks is not None
                    else _cfg("HARVEST_MAX_PICKS", 2))

    stake = max(1, round(capital * tok_pct))
    candidates = []
    for p in predictions or []:
        try:
            num = int(p["number"])
        except (KeyError, TypeError, ValueError):
            continue
        conf = float(p.get("confidence", 0.0) or 0.0)
        mult = net_multiplier(num)
        if num not in target:
            continue
        if conf < min_conf:
            continue
        candidates.append((conf * mult, num, conf, mult))

    candidates.sort(reverse=True)
    picks = []
    for _, num, conf, mult in candidates[:picks_cap]:
        # EV per token is honestly reported (still negative for a fair wheel).
        ev = conf * (mult + 1) - 1
        picks.append({
            "number": num,
            "token_bet": stake,
            "confidence": conf,
            "multiplier": mult,
            "ev_per_token": ev,
            "is_positive_ev": ev > 0,
            "support": p_support(predictions, num),
            "mode": "harvest",
        })
    return picks


def p_support(predictions, num):
    for p in predictions or []:
        try:
            if int(p["number"]) == num:
                return p.get("support")
        except (KeyError, TypeError, ValueError):
            continue
    return None


def should_skip_round(skip_rate=None, rng=None):
    """Randomly skip a fraction of rounds (capital preservation)."""
    rate = float(skip_rate if skip_rate is not None
                 else _cfg("HARVEST_SKIP_RATE", 0.20))
    r = rng.random() if rng is not None else random.random()
    return r < rate


def round_profit(picks, actual_number):
    """Net token change for a single spin given the harvest picks.

    A pick wins net (token_bet * multiplier) if its number hits, else it loses
    its token_bet. Exactly one number wins per spin.
    """
    total = 0
    for b in picks:
        stake = int(b.get("token_bet", 0))
        if int(b["number"]) == int(actual_number):
            total += stake * net_multiplier(int(b["number"]))
        else:
            total -= stake
    return total


def _sharpe(returns):
    """Sharpe-equivalent: mean per-round return / population stddev."""
    n = len(returns)
    if n == 0:
        return None
    mean = sum(returns) / n
    var = sum((x - mean) ** 2 for x in returns) / n
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return mean / sd


def simulate(actuals, confidence_map, starting_capital=1000,
             seed=42, **overrides):
    """Walk a sequence of actual spin results through Variance Harvest mode.

    confidence_map: {number: confidence} used every round to pick lottery
    tickets (a fixed belief; harvest does not need a smart model). Override any
    of target_multipliers/min_confidence/token_pct/max_picks/skip_rate via
    kwargs for testing.

    Returns a dict: final_capital, profit, n_rounds, n_harvest_rounds,
    n_skipped, n_wins, win_rate, big_wins (multiplier>=10 hits), max_drawdown,
    sharpe.
    """
    rng = random.Random(seed)
    skip_rate = overrides.get("skip_rate", _cfg("HARVEST_SKIP_RATE", 0.20))
    preds = [{"number": int(n), "confidence": float(c)}
             for n, c in confidence_map.items()]

    capital = float(starting_capital)
    peak = capital
    max_dd = 0.0
    returns = []
    n_harvest = n_skip = n_wins = big_wins = 0

    for actual in actuals:
        if should_skip_round(skip_rate, rng):
            n_skip += 1
            returns.append(0)
            continue
        picks = harvest_picks(
            preds, capital,
            target_multipliers=overrides.get("target_multipliers"),
            min_confidence=overrides.get("min_confidence"),
            token_pct=overrides.get("token_pct"),
            max_picks=overrides.get("max_picks"),
        )
        if not picks:
            n_skip += 1
            returns.append(0)
            continue
        n_harvest += 1
        pnl = round_profit(picks, actual)
        capital += pnl
        returns.append(pnl)
        won = [b for b in picks if int(b["number"]) == int(actual)]
        if won:
            n_wins += 1
            if net_multiplier(int(actual)) >= 10:
                big_wins += 1
        peak = max(peak, capital)
        max_dd = max(max_dd, peak - capital)

    return {
        "final_capital": capital,
        "profit": capital - starting_capital,
        "n_rounds": len(actuals),
        "n_harvest_rounds": n_harvest,
        "n_skipped": n_skip,
        "n_wins": n_wins,
        "win_rate": (n_wins / n_harvest) if n_harvest else 0.0,
        "big_wins": big_wins,
        "max_drawdown": max_dd,
        "sharpe": _sharpe(returns),
    }
