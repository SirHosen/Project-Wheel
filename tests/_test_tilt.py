# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Unit tests for the PROMPT 16 anti-tilt guard (core/tilt.py).

Pure / headless: no TensorFlow, no GUI. Run with: python _test_tilt.py
"""
from datetime import datetime, timedelta

from core.tilt import is_rising, detect_tilt, TiltDetector

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def _ev(dt, is_win, staked):
    return {"timestamp": dt, "is_win": is_win, "staked": staked}


T0 = datetime(2026, 6, 9, 4, 0, 0)


# --------------------------------------------------------------------------- #
# is_rising
# --------------------------------------------------------------------------- #
def test_is_rising():
    check(is_rising([1, 2, 3]) is True, "strictly increasing rises")
    check(is_rising([10, 10, 10]) is False, "flat is not rising (strict)")
    check(is_rising([1, 2, 2], strict=True) is False, "plateau breaks strict rise")
    check(is_rising([1, 2, 2], strict=False) is True, "non-strict allows plateau if net up")
    check(is_rising([3, 2, 1]) is False, "decreasing not rising")
    check(is_rising([5]) is False, "single element not rising")
    check(is_rising([], strict=False) is False, "empty not rising")
    check(is_rising([1, 3, 2], strict=False) is False, "a dip breaks non-strict rise")


# --------------------------------------------------------------------------- #
# detect_tilt
# --------------------------------------------------------------------------- #
def test_detect_trigger():
    events = [
        _ev(T0, False, 10),
        _ev(T0 + timedelta(minutes=1), False, 20),
        _ev(T0 + timedelta(minutes=2), False, 30),
    ]
    res = detect_tilt(events)
    check(res["triggered"] is True, "3 rising losses in 2 min trips")
    check("menit" in res["reason"], "reason mentions window")
    check(abs(res["span_minutes"] - 2.0) < 1e-6, "span computed = 2 min")


def test_detect_no_trigger_window():
    events = [
        _ev(T0, False, 10),
        _ev(T0 + timedelta(minutes=3), False, 20),
        _ev(T0 + timedelta(minutes=8), False, 30),
    ]
    res = detect_tilt(events, window_minutes=5.0)
    check(res["triggered"] is False, "8-min span exceeds 5-min window")
    check(res["reason"] == "di luar jendela waktu", "reason = outside window")


def test_detect_no_trigger_win():
    events = [
        _ev(T0, False, 10),
        _ev(T0 + timedelta(minutes=1), True, 20),
        _ev(T0 + timedelta(minutes=2), False, 30),
    ]
    res = detect_tilt(events)
    check(res["triggered"] is False, "a win in the window blocks trigger")
    check(res["reason"] == "ada menang di jendela", "reason = win present")


def test_detect_no_trigger_not_rising():
    events = [
        _ev(T0, False, 30),
        _ev(T0 + timedelta(minutes=1), False, 20),
        _ev(T0 + timedelta(minutes=2), False, 10),
    ]
    res = detect_tilt(events)
    check(res["triggered"] is False, "falling stakes do not trip")
    check(res["reason"] == "taruhan tidak naik", "reason = not rising")


def test_detect_rising_disabled():
    # Same falling stakes, but require_rising=False -> trips on losses alone.
    events = [
        _ev(T0, False, 30),
        _ev(T0 + timedelta(minutes=1), False, 20),
        _ev(T0 + timedelta(minutes=2), False, 10),
    ]
    res = detect_tilt(events, require_rising=False)
    check(res["triggered"] is True, "require_rising=False trips on losses only")
    check("menaik" not in res["reason"], "reason omits rising note when disabled")


def test_detect_no_bet_round():
    events = [
        _ev(T0, False, 0),
        _ev(T0 + timedelta(minutes=1), False, 20),
        _ev(T0 + timedelta(minutes=2), False, 30),
    ]
    res = detect_tilt(events)
    check(res["triggered"] is False, "a zero-stake round blocks trigger")
    check(res["reason"] == "ada ronde tanpa taruhan", "reason = no-bet round")


def test_detect_insufficient_data():
    events = [_ev(T0, False, 10), _ev(T0 + timedelta(minutes=1), False, 20)]
    res = detect_tilt(events, n_losses=3)
    check(res["triggered"] is False, "fewer than n_losses cannot trip")
    check(res["reason"] == "data kurang", "reason = insufficient data")


def test_detect_uses_tail_only():
    # Old wins far in the past must not block a fresh 3-loss tail.
    events = [
        _ev(T0 - timedelta(hours=1), True, 5),
        _ev(T0, False, 10),
        _ev(T0 + timedelta(minutes=1), False, 20),
        _ev(T0 + timedelta(minutes=2), False, 30),
    ]
    res = detect_tilt(events)
    check(res["triggered"] is True, "only the last n_losses events matter")


def test_detect_iso_strings():
    events = [
        _ev(T0.isoformat(), False, 10),
        _ev((T0 + timedelta(minutes=1)).isoformat(), False, 20),
        _ev((T0 + timedelta(minutes=2)).isoformat(), False, 30),
    ]
    res = detect_tilt(events)
    check(res["triggered"] is True, "ISO timestamp strings are parsed")


def test_detect_n_losses_param():
    events = [
        _ev(T0, False, 10),
        _ev(T0 + timedelta(minutes=1), False, 20),
    ]
    res = detect_tilt(events, n_losses=2)
    check(res["triggered"] is True, "n_losses=2 trips on 2 rising losses")


# --------------------------------------------------------------------------- #
# TiltDetector (stateful)
# --------------------------------------------------------------------------- #
def test_detector_trips_and_cooldown():
    d = TiltDetector(n_losses=3, window_minutes=5.0, cooldown_seconds=60.0)
    d.record(T0, False, 10)
    d.record(T0 + timedelta(minutes=1), False, 20)
    st = d.record(T0 + timedelta(minutes=2), False, 30)
    check(st["tilt_triggered"] is True, "third rising loss trips detector")
    check(st["in_cooldown"] is True, "cooldown active right after trip")
    check(d.trigger_count == 1, "trigger_count incremented")
    # 30s later still in cooldown (60s window).
    st2 = d.status(now=T0 + timedelta(minutes=2, seconds=30))
    check(st2["in_cooldown"] is True, "still cooling at +30s")
    check(28 <= st2["remaining_seconds"] <= 32, "≈30s remaining")
    # 61s later cooldown expired.
    st3 = d.status(now=T0 + timedelta(minutes=3, seconds=1))
    check(st3["in_cooldown"] is False, "cooldown auto-expires after 60s")
    check(st3["remaining_seconds"] == 0.0, "no remaining time after expiry")


def test_detector_no_trip_on_wins():
    d = TiltDetector()
    d.record(T0, True, 10)
    d.record(T0 + timedelta(minutes=1), True, 20)
    st = d.record(T0 + timedelta(minutes=2), True, 30)
    check(st["tilt_triggered"] is False, "all wins never trip")
    check(st["in_cooldown"] is False, "no cooldown on wins")


def test_detector_clear_cooldown():
    d = TiltDetector(cooldown_seconds=60.0)
    d.record(T0, False, 10)
    d.record(T0 + timedelta(minutes=1), False, 20)
    d.record(T0 + timedelta(minutes=2), False, 30)
    check(d.is_in_cooldown(now=T0 + timedelta(minutes=2, seconds=5)) is True,
          "in cooldown before clear")
    d.clear_cooldown()
    check(d.is_in_cooldown(now=T0 + timedelta(minutes=2, seconds=5)) is False,
          "clear_cooldown ends the breather")


def test_detector_disabled():
    d = TiltDetector(enabled=False, cooldown_seconds=60.0)
    d.record(T0, False, 10)
    d.record(T0 + timedelta(minutes=1), False, 20)
    st = d.record(T0 + timedelta(minutes=2), False, 30)
    check(st["tilt_triggered"] is False, "disabled detector never trips")
    check(st["in_cooldown"] is False, "disabled detector has no cooldown")
    check(st.get("enabled") is False, "status reports enabled=False")


def test_detector_reset():
    d = TiltDetector()
    d.record(T0, False, 10)
    d.record(T0 + timedelta(minutes=1), False, 20)
    d.record(T0 + timedelta(minutes=2), False, 30)
    d.reset()
    check(d.cooldown_until is None, "reset clears cooldown")
    check(d.trigger_count == 0, "reset zeroes trigger_count")
    check(d._events == [], "reset empties the buffer")


def test_detector_buffer_cap():
    d = TiltDetector(n_losses=3, buffer_size=5)
    for i in range(20):
        d.record(T0 + timedelta(seconds=i), True, 1)
    check(len(d._events) == 5, "buffer never exceeds buffer_size")


def test_detector_buffer_min():
    # buffer_size smaller than n_losses must be raised to n_losses.
    d = TiltDetector(n_losses=3, buffer_size=1)
    check(d.buffer_size >= 3, "buffer_size floored at n_losses")


def test_detector_recovery_then_retrip():
    d = TiltDetector(n_losses=3, cooldown_seconds=60.0)
    # First trip.
    d.record(T0, False, 10)
    d.record(T0 + timedelta(minutes=1), False, 20)
    d.record(T0 + timedelta(minutes=2), False, 30)
    check(d.trigger_count == 1, "first trip counted")
    # A win breaks the streak; no new trip.
    st = d.record(T0 + timedelta(minutes=3), True, 40)
    check(st["tilt_triggered"] is False, "win after trip does not re-trip")
    # Three fresh rising losses re-trip.
    d.record(T0 + timedelta(minutes=4), False, 10)
    d.record(T0 + timedelta(minutes=5), False, 20)
    st2 = d.record(T0 + timedelta(minutes=6), False, 30)
    check(st2["tilt_triggered"] is True, "fresh rising losses re-trip")
    check(d.trigger_count == 2, "trigger_count now 2")


if __name__ == "__main__":
    for fn in [
        test_is_rising,
        test_detect_trigger,
        test_detect_no_trigger_window,
        test_detect_no_trigger_win,
        test_detect_no_trigger_not_rising,
        test_detect_rising_disabled,
        test_detect_no_bet_round,
        test_detect_insufficient_data,
        test_detect_uses_tail_only,
        test_detect_iso_strings,
        test_detect_n_losses_param,
        test_detector_trips_and_cooldown,
        test_detector_no_trip_on_wins,
        test_detector_clear_cooldown,
        test_detector_disabled,
        test_detector_reset,
        test_detector_buffer_cap,
        test_detector_buffer_min,
        test_detector_recovery_then_retrip,
    ]:
        fn()
    total = PASS + FAIL
    print(f"\n_test_tilt: {PASS}/{total} checks passed"
          + ("" if FAIL == 0 else f"  ({FAIL} FAILED)"))
    raise SystemExit(1 if FAIL else 0)
