# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Test PROMPT 5: AdaptiveRiskManager."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.risk_manager import AdaptiveRiskManager


def test_full_risk_when_healthy():
    rm = AdaptiveRiskManager(1000)
    assert rm.risk_multiplier() == 1.0
    assert rm.effective_risk_pct(0.30) == 0.30
    print("OK healthy -> full risk")


def test_hard_drawdown_stops():
    rm = AdaptiveRiskManager(1000)
    rm.update(-400, is_win=False)  # 40% drawdown > hard 35%
    stop, reason = rm.should_stop()
    assert stop and "drawdown" in reason.lower()
    assert rm.risk_multiplier() == 0.0
    print("OK hard drawdown -> stop")


def test_soft_drawdown_scales_down():
    rm = AdaptiveRiskManager(1000)
    rm.update(-220, is_win=False)  # 22% dd: between soft(20)/hard(35), below daily(25)
    m = rm.risk_multiplier()
    assert 0.0 < m < 1.0, m
    print(f"OK soft drawdown -> scaled {m:.2f}")


def test_losing_streak_brake():
    rm = AdaptiveRiskManager(100000)  # big bankroll so drawdown stays tiny
    for _ in range(5):
        rm.update(-1, is_win=False)
    assert rm.loss_streak == 5
    assert rm.risk_multiplier() <= 0.5 + 1e-9
    print("OK 5-loss streak -> brake")


def test_winning_streak_boost():
    rm = AdaptiveRiskManager(100000)
    for _ in range(8):
        rm.update(+1, is_win=True)
    assert rm.win_streak == 8
    assert rm.risk_multiplier() > 1.0
    print(f"OK 8-win streak -> boost {rm.risk_multiplier():.2f}")


def test_daily_stop():
    rm = AdaptiveRiskManager(1000)
    # lose 26% in the day but keep bankroll above hard drawdown by winning back later? 
    # simplest: single 26% loss triggers daily stop (also < hard dd 35%)
    rm.update(-260, is_win=False)
    stop, reason = rm.should_stop()
    assert stop, reason
    print("OK daily loss >= 25% -> stop")


def test_persistence():
    path = os.path.join(tempfile.gettempdir(), "_risk_test.json")
    if os.path.exists(path):
        os.remove(path)
    rm = AdaptiveRiskManager(1000, path=path)
    rm.update(+50, is_win=True)
    rm.update(-20, is_win=False)
    assert rm.save()
    rm2 = AdaptiveRiskManager(1000, path=path)
    assert rm2.load()
    assert abs(rm2.current - rm.current) < 1e-9
    print("OK persistence round-trip")
    os.remove(path)


if __name__ == "__main__":
    test_full_risk_when_healthy()
    test_hard_drawdown_stops()
    test_soft_drawdown_scales_down()
    test_losing_streak_brake()
    test_winning_streak_boost()
    test_daily_stop()
    test_persistence()
    print("\nALL PROMPT 5 TESTS PASSED")
