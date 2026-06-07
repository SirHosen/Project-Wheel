# -*- coding: utf-8 -*-
"""Rigorous cross-checks for EV/Kelly betting + bias detection (no GUI deps)."""
import math
import random
from collections import Counter

from config import settings
from core.betting import net_multiplier, ev_per_token, kelly_allocation
from predictors.markov_engine import MarkovEngine

SEQ = settings.SPINWHEEL_SEQUENCE
VALID = settings.VALID_NUMBERS
freq = {n: SEQ.count(n) / len(SEQ) for n in VALID}

print("=== 1) EV + break-even per number (fair wheel) ===")
for n in VALID:
    m = net_multiplier(n)
    be = 1 / (m + 1)
    ev = ev_per_token(freq[n], n)
    # cross-check vs explicit reward math
    bet = 100
    payout = settings.calculate_reward(bet, n)          # includes stake
    net_win = payout - bet                               # net profit on win
    assert net_win == bet * m, (n, net_win, bet * m)
    ev_explicit = (freq[n] * net_win - (1 - freq[n]) * bet) / bet
    assert abs(ev_explicit - ev) < 1e-9, (n, ev_explicit, ev)
    print(f"n={n:>2} freq={freq[n]*100:5.1f}%  break-even={be*100:5.2f}%  EV/token={ev:+.3f}")

all_neg = all(ev_per_token(freq[n], n) < 0 for n in VALID)
print("All numbers -EV on a FAIR wheel:", all_neg)
assert all_neg, "Fair wheel should have no +EV bet"

print("\n=== 2) kelly_allocation behaviour ===")
# (a) Fair wheel confidences -> SKIP (every token_bet == 0)
fair_preds = [{"number": n, "confidence": freq[n]} for n in VALID]
fair_alloc = kelly_allocation(fair_preds, capital=1000, risk_pct=0.30)
print("Fair-wheel alloc bets:", [(a["number"], a["token_bet"]) for a in fair_alloc])
assert all(a["token_bet"] == 0 for a in fair_alloc), "Fair wheel must recommend SKIP"

# (b) Strong bias: model is 70% sure of number 1 -> should BET (+EV)
bias_preds = [{"number": 1, "confidence": 0.70}, {"number": 2, "confidence": 0.15},
              {"number": 5, "confidence": 0.05}]
bias_alloc = kelly_allocation(bias_preds, capital=1000, risk_pct=0.30)
bets = [(a["number"], a["token_bet"], round(a["ev_per_token"], 3)) for a in bias_alloc]
print("Biased alloc bets:", bets)
assert any(a["token_bet"] > 0 for a in bias_alloc), "Should bet when +EV exists"
total = sum(a["token_bet"] for a in bias_alloc)
assert total <= int(1000 * 0.30), ("budget exceeded", total)
# EV(0.7 on n=1) = 0.7*2 - 1 = 0.4 ; half-Kelly f=0.4 -> raw=1000*0.5*0.4=200 (<=budget 300)
n1 = next(a for a in bias_alloc if a["number"] == 1)
assert abs(n1["ev_per_token"] - 0.4) < 1e-9
assert 198 <= n1["token_bet"] <= 200, n1["token_bet"]  # ~200, floored (conservative)
print("Bet on n=1 =", n1["token_bet"], "(~200, half-Kelly, within budget)")

# (c) High-payout long shot becomes +EV above its tiny break-even
long_preds = [{"number": 40, "confidence": 0.05}, {"number": 1, "confidence": 0.40}]
long_alloc = kelly_allocation(long_preds, capital=1000, risk_pct=0.30)
print("Longshot alloc:", [(a["number"], a["token_bet"], round(a["ev_per_token"],3)) for a in long_alloc])
# 40 at 5% -> EV = 0.05*41 - 1 = +1.05 (+EV); 1 at 40% -> EV = 0.4*2-1 = -0.20 (-EV, skip)
assert any(a["number"] == 40 and a["token_bet"] > 0 for a in long_alloc)
assert all(a["number"] != 1 or a["token_bet"] == 0 for a in long_alloc)

print("\n=== 3) Bias walk-forward (Markov vs baseline top-1) ===")

def walk_forward(history):
    mk = MarkovEngine()
    base_top1 = max(mk.prior, key=mk.prior.get)
    mk_hits = base_hits = rounds = 0
    for i in range(6, len(history) - 1):
        preds = mk.predict_next(history[: i + 1])
        actual = history[i + 1]
        if preds and preds[0]["number"] == actual:
            mk_hits += 1
        if base_top1 == actual:
            base_hits += 1
        rounds += 1
    mk_rate = mk_hits / rounds
    base_rate = base_hits / rounds
    pooled = (mk_hits + base_hits) / (2 * rounds)
    denom = math.sqrt(max(1e-9, pooled * (1 - pooled) * (2 / rounds)))
    z = (mk_rate - base_rate) / denom if denom > 0 else 0.0
    edge = z > 1.96 and mk_rate > base_rate
    return mk_rate, base_rate, z, edge, rounds

random.seed(7)
# RANDOM (fair) sequence sampled by wheel frequency -> expect NO edge
weights = [SEQ.count(n) for n in VALID]
rnd_hist = random.choices(VALID, weights=weights, k=600)
mk_r, base_r, z, edge, rnds = walk_forward(rnd_hist)
print(f"RANDOM: model={mk_r*100:.1f}% base={base_r*100:.1f}% z={z:.2f} edge={edge} ({rnds} ronde)")
assert not edge, "Fair random data must NOT show a significant edge"

# BIASED (strong momentum: next often equals previous) -> expect EDGE
bias_hist = [random.choice(VALID)]
for _ in range(600):
    if random.random() < 0.6:
        bias_hist.append(bias_hist[-1])           # sticky / momentum
    else:
        bias_hist.append(random.choices(VALID, weights=weights, k=1)[0])
mk_r, base_r, z, edge, rnds = walk_forward(bias_hist)
print(f"BIASED: model={mk_r*100:.1f}% base={base_r*100:.1f}% z={z:.2f} edge={edge} ({rnds} ronde)")
assert edge, "Strong autocorrelation must be detected as an edge"

print("\nALL CHECKS PASSED")
