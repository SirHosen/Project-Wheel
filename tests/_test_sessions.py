# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Test PROMPT 9: session detection + temporal-drift statistics."""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sessions import (
    chi_square_gof, ks_2sample, detect_session_drift, recency_weights, parse_ts,
)
from core import diagnostics
from data.tracker import Tracker
from config import settings

VN = settings.VALID_NUMBERS
SEQ = settings.SPINWHEEL_SEQUENCE


def _rec(ts, actual, predicted=None, profit=0, is_win=False):
    return {"timestamp": ts.isoformat(), "actual_number": actual,
            "predicted_number": predicted, "profit_change": profit,
            "is_win": is_win, "bets": []}


def _build_tracker(records):
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "_test_sessions_history.json")
    t = Tracker(history_file=path)
    t.data["history"] = records
    return t


def test_get_sessions_splits_on_gap():
    base = datetime(2026, 6, 9, 1, 0, 0)
    recs = []
    # Session A: 5 spins 2 min apart.
    for i in range(5):
        recs.append(_rec(base + timedelta(minutes=2 * i), 1))
    # 90-minute gap -> new session B: 4 spins.
    b2 = base + timedelta(minutes=2 * 4 + 90)
    for i in range(4):
        recs.append(_rec(b2 + timedelta(minutes=3 * i), 2))
    t = _build_tracker(recs)
    sessions = t.get_sessions(gap_minutes=30)
    assert len(sessions) == 2, len(sessions)
    assert len(sessions[0]) == 5 and len(sessions[1]) == 4
    print(f"OK get_sessions split into {len(sessions)} sessions on >=30min gap")


def test_per_session_stats_fields():
    base = datetime(2026, 6, 9, 1, 0, 0)
    recs = [_rec(base + timedelta(minutes=2 * i), 1, predicted=1, profit=5, is_win=True) for i in range(6)]
    recs += [_rec(base + timedelta(minutes=200 + 2 * i), 2, predicted=5, profit=-10) for i in range(4)]
    t = _build_tracker(recs)
    stats = t.per_session_stats(gap_minutes=30)
    assert len(stats) == 2
    s0 = stats[0]
    assert s0["n_spins"] == 6 and s0["dominant_num"] == 1
    assert abs(s0["win_rate"] - 100.0) < 1e-6 and s0["profit"] == 30
    assert stats[1]["dominant_num"] == 2 and stats[1]["profit"] == -40
    print("OK per_session_stats: n_spins/dominant/win_rate/profit correct")


def test_chi_square_detects_skew():
    fair = [n for n in SEQ] * 3                 # mirrors wheel -> low chi^2
    skewed = [40] * 50                          # all rare number -> huge chi^2
    chi_fair = chi_square_gof(fair, VN, SEQ)
    chi_skew = chi_square_gof(skewed, VN, SEQ)
    assert chi_skew > chi_fair
    assert chi_square_gof([], VN, SEQ) is None
    print(f"OK chi-square GOF: skewed {chi_skew:.0f} >> fair {chi_fair:.1f}")


def test_ks_2sample_identifies_difference():
    same_a = [1, 2, 5, 8, 10] * 20
    same_b = [1, 2, 5, 8, 10] * 20
    d_same, p_same = ks_2sample(same_a, same_b)
    diff_a = [1] * 100
    diff_b = [40] * 100
    d_diff, p_diff = ks_2sample(diff_a, diff_b)
    assert d_same < 0.05 and p_same > 0.05
    assert d_diff > 0.9 and p_diff < 0.05
    print(f"OK KS 2-sample: identical p={p_same:.2f} (no diff), disjoint p={p_diff:.4f} (diff)")


def test_detect_drift_verdict():
    # Early sessions dominated by 1; late sessions dominated by 40 -> drift.
    early = [[1, 1, 2, 1, 1] for _ in range(3)]
    late = [[40, 40, 30, 40, 40] for _ in range(3)]
    res = detect_session_drift(early + late, alpha=0.05)
    assert res["drift"] is True and res["p_value"] < 0.05
    # Single session -> cannot decide.
    res1 = detect_session_drift([[1, 2, 5]], alpha=0.05)
    assert res1["drift"] is False and res1["p_value"] is None
    print(f"OK drift verdict: shift detected (p={res['p_value']:.4f}); single-session abstains")


def test_recency_weights_monotone():
    w = recency_weights(10, half_life=5)
    assert len(w) == 10
    assert all(w[i] <= w[i + 1] + 1e-12 for i in range(len(w) - 1)), "newest must weigh most"
    assert abs(w[-1] - 1.0) < 1e-9                       # newest = 1.0
    assert abs(w[-6] / w[-1] - 0.5) < 1e-6               # one half-life back = 0.5
    assert recency_weights(0) == []
    print("OK recency_weights monotone increasing toward newest; half-life halves")


def test_diagnostics_section_11_renders():
    base = datetime(2026, 6, 9, 1, 0, 0)
    recs = [_rec(base + timedelta(minutes=2 * i), 1, 1, 5, True) for i in range(5)]
    recs += [_rec(base + timedelta(minutes=300 + 2 * i), 40, 5, -10) for i in range(5)]
    data = {"current_capital": 1000, "history": recs, "total_predictions": 10,
            "wins": 5, "losses": 5, "profit": -25}
    sa = diagnostics.session_analysis(data, SEQ, VN)
    assert sa["n_sessions"] == 2
    rep = diagnostics.compute_report(data, SEQ, VN, app_version="test")
    md = diagnostics.report_to_markdown(rep)
    assert "## 11. Analisis per-sesi" in md
    assert "sesi" in md.lower()
    print("OK diagnostics section 11 computes + renders in markdown")


if __name__ == "__main__":
    test_get_sessions_splits_on_gap()
    test_per_session_stats_fields()
    test_chi_square_detects_skew()
    test_ks_2sample_identifies_difference()
    test_detect_drift_verdict()
    test_recency_weights_monotone()
    test_diagnostics_section_11_renders()
    print("\nALL PROMPT 9 TESTS PASSED")
