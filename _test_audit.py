# -*- coding: utf-8 -*-
"""Smoke + sanity test for core/diagnostics audit report (no GUI/tf)."""
import random
from datetime import datetime, timedelta
from config import settings
from core.diagnostics import compute_report, report_to_markdown, export_audit_bundle

random.seed(11)
seq = settings.SPINWHEEL_SEQUENCE

# Build ~120 rounds resembling real play: mostly skips/small bets + a few wins.
history = []
t = datetime(2026, 6, 6, 4, 46, 0)
for i in range(120):
    actual = random.choice(seq)
    t += timedelta(seconds=random.randint(3, 30))
    # ~65% skip (profit 0), else bet: small loss or occasional win
    roll = random.random()
    if roll < 0.65:
        predicted, profit, win = None, 0, False
    elif roll < 0.90:
        predicted, profit, win = None, -random.choice([1, 2, 3]), False
    else:
        predicted = actual
        profit, win = settings.calculate_reward(2, actual) - 4, True
    history.append({
        "timestamp": t.isoformat(),
        "actual_number": actual,
        "predicted_number": predicted,
        "profit_change": profit,
        "is_win": win,
    })

profit_total = sum(h["profit_change"] for h in history)
data = {
    "current_capital": 20 + profit_total,
    "total_predictions": len(history),
    "wins": sum(h["is_win"] for h in history),
    "losses": sum(not h["is_win"] for h in history),
    "profit": profit_total,
    "history": history,
}

r = compute_report(data, settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS, app_version=settings.APP_VERSION)

# Structural assertions.
for key in ("meta", "bankroll", "performance", "round_stats", "streaks",
            "per_number", "fairness", "autocorrelation", "markov_edge", "data_quality"):
    assert key in r, f"missing section {key}"
assert len(r["per_number"]) == len(settings.VALID_NUMBERS)
assert r["meta"]["total_rounds"] == 120
assert r["fairness"]["verdict"] in ("fair", "biased", "insufficient")
assert abs(r["bankroll"]["realized_profit"] - profit_total) < 1e-6

md = report_to_markdown(r)
assert "Laporan Audit" in md and "chi-square" in md.lower()

files = export_audit_bundle(data, "/tmp/audit_demo.md", settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS, app_version=settings.APP_VERSION)
print("Files written:", files)
print("=" * 60)
print(md)
print("=" * 60)
print("ALL CHECKS PASSED - audit report OK.")
