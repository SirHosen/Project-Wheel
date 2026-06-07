# -*- coding: utf-8 -*-
"""Pure-logic test for the live adaptive prior (no GUI / no tensorflow).

Replicates lock_manual_percentages + refresh_live_percentages math and proves
the displayed % self-tunes from the user's input toward the wheel's REAL
behaviour as spins accumulate.
"""
from collections import Counter
from config import settings

STRENGTH = len(settings.SPINWHEEL_SEQUENCE)  # 54
VN = settings.VALID_NUMBERS


def lock(manual_pct):
    total = sum(manual_pct.values())
    if total <= 0:
        c = Counter(settings.SPINWHEEL_SEQUENCE); L = len(settings.SPINWHEEL_SEQUENCE)
        manual_pct = {n: c.get(n, 0) / L * 100 for n in VN}
        total = sum(manual_pct.values())
    manual_pct = {n: v / total * 100 for n, v in manual_pct.items()}
    prior_counts = {n: manual_pct[n] / 100 * STRENGTH for n in VN}
    return prior_counts


def live(prior_counts, history):
    c = Counter(history)
    denom = STRENGTH + len(history)
    return {n: (prior_counts.get(n, 0.0) + c.get(n, 0)) / denom * 100 for n in VN}


# User WRONGLY believes 8 is hot (50%); reality: number 1 dominates (~37%).
user_input = {n: 0.0 for n in VN}
user_input[8] = 50.0
user_input[1] = 50.0
prior = lock(user_input)

import random
random.seed(7)
seq = settings.SPINWHEEL_SEQUENCE
real_stream = [random.choice(seq) for _ in range(400)]  # fair wheel draws

print("prior%% (locked):   1=%.1f  8=%.1f" % (prior[1] / STRENGTH * 100, prior[8] / STRENGTH * 100))
for k in (0, 5, 20, 100, 400):
    lv = live(prior, real_stream[:k])
    true1 = Counter(seq)[1] / len(seq) * 100
    print(f"after {k:4d} spins: 1={lv[1]:5.2f}%  8={lv[8]:5.2f}%   (roda asli 1~{true1:.1f}%)")

lv_start = live(prior, [])
lv_final = live(prior, real_stream)
true1 = Counter(seq)[1] / len(seq) * 100
true8 = Counter(seq)[8] / len(seq) * 100
assert abs(sum(lv_final.values()) - 100) < 1e-6, "live% must sum to 100"
# After data, both numbers must move TOWARD their real wheel frequency.
assert abs(lv_final[1] - true1) < abs(lv_start[1] - true1), "1 must move toward real freq"
assert abs(lv_final[8] - true8) < abs(lv_start[8] - true8), "8 must move toward real freq"
assert lv_final[1] > lv_final[8], "1 (real freq) must end above user's wrong 8 bet"
print("\nALL CHECKS PASSED - live prior converges from input -> real wheel.")
