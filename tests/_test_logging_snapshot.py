# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Test PROMPT 1: full per-round bet-snapshot logging in data/tracker.py."""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.tracker import Tracker
from config import settings


def _new_tracker():
    d = tempfile.mkdtemp()
    return Tracker(history_file=os.path.join(d, "history.json")), d


def test_snapshot_stored_on_win_and_loss():
    t, d = _new_tracker()
    try:
        snap = [
            {"number": 5, "token_bet": 2, "confidence": 0.3,
             "ev_per_token": 0.1, "is_positive_ev": True, "support": 30},
            {"number": 1, "token_bet": 1, "confidence": 0.2,
             "ev_per_token": -0.5, "is_positive_ev": False, "support": 80},
        ]
        # WIN on 5
        t.record_result(5, 5, settings.calculate_reward(2, 5) - 3,
                         bet_snapshot=snap, engine_used="Markov")
        # LOSS on 8 -> snapshot must STILL be stored
        t.record_result(8, None, -3, bet_snapshot=snap, engine_used="Markov")

        hist = t.data["history"]
        assert len(hist) == 2
        assert hist[0]["bets"] and hist[1]["bets"], "snapshot must persist on win AND loss"
        assert hist[0].get("engine_used") == "Markov"
        assert hist[1]["bets"][0]["number"] == 5
        assert hist[0]["bets"][0]["support"] == 30
        print("OK snapshot stored on win and loss")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_backward_compat_no_snapshot():
    t, d = _new_tracker()
    try:
        t.record_result(2, None, -1)  # legacy call, no snapshot
        rec = t.data["history"][0]
        assert rec["bets"] == [], "legacy records must have empty bets list"
        assert t.get_per_number_bet_stats() == {}
        assert t.get_engine_bet_distribution()  # 'unknown' engine bucket exists
        print("OK backward compatible (no snapshot)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_per_number_stats_correct():
    t, d = _new_tracker()
    try:
        snap = [
            {"number": 5, "token_bet": 2, "is_positive_ev": True},
            {"number": 1, "token_bet": 1, "is_positive_ev": False},
        ]
        # Round 1: actual 5 -> bet on 5 wins, bet on 1 loses.
        t.record_result(5, 5, settings.calculate_reward(2, 5) - 3,
                         bet_snapshot=snap, engine_used="Markov")
        # Round 2: actual 8 -> both lose.
        t.record_result(8, None, -3, bet_snapshot=snap, engine_used="Markov")

        stats = t.get_per_number_bet_stats()
        assert stats[5]["bets"] == 2 and stats[5]["wins"] == 1
        assert stats[5]["total_staked"] == 4
        assert abs(stats[5]["hit_rate"] - 50.0) < 1e-6
        expected_net5 = (settings.calculate_reward(2, 5) - 2) - 2
        assert stats[5]["net_profit"] == expected_net5, (stats[5]["net_profit"], expected_net5)
        assert stats[1]["bets"] == 2 and stats[1]["wins"] == 0
        assert stats[1]["net_profit"] == -2

        # Per-number net profit must sum to total realized profit (consistency).
        total_net = sum(s["net_profit"] for s in stats.values())
        realized = sum(h["profit_change"] for h in t.data["history"])
        assert total_net == realized, (total_net, realized)

        dist = t.get_engine_bet_distribution()
        assert dist["Markov"]["rounds"] == 2
        assert dist["Markov"]["bet_rounds"] == 2
        print("OK per-number stats + engine distribution correct")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_snapshot_stored_on_win_and_loss()
    test_backward_compat_no_snapshot()
    test_per_number_stats_correct()
    print("\nALL PROMPT 1 TESTS PASSED")
