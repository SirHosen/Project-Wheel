# -*- coding: utf-8 -*-
"""PROMPT 16: anti-tilt guard (TiltDetector).

"Tilt" = chasing losses emotionally -- betting AGAIN right after a loss, and
betting MORE each time. On a fair, memoryless wheel that is the fastest path to
ruin: a losing streak never makes a win "due", and escalating stakes only raise
variance and expected loss.

This module watches the recent betting cadence and, when it sees the classic
tilt pattern (N losing bet-rounds inside a short time window WITH rising
stakes), trips a mandatory cooldown. While the cooldown is active the UI should
refuse / strongly discourage the next bet until the user takes a breather.

Pure & GUI-agnostic (stdlib only). The detector keeps a small rolling buffer of
recent (timestamp, is_win, staked) events and exposes a deterministic status.

HONEST NOTE: this protects bankroll and discipline, NOT a profit feature. It
cannot make a fair wheel beatable; it only stops emotion-driven overbetting.
"""
from datetime import datetime, timedelta

from core.sessions import parse_ts


def _to_dt(ts):
    """Coerce a datetime or ISO string to datetime; None if unparseable."""
    if isinstance(ts, datetime):
        return ts
    if ts is None:
        return None
    return parse_ts(ts)


def is_rising(stakes, strict=True):
    """True if the stake sequence escalates (tilt = chasing with bigger bets).

    strict=True  -> every step must strictly increase.
    strict=False -> no step may decrease AND the last must exceed the first.
    """
    nums = [float(s) for s in stakes]
    if len(nums) < 2:
        return False
    if strict:
        return all(b > a for a, b in zip(nums, nums[1:]))
    if any(b < a for a, b in zip(nums, nums[1:])):
        return False
    return nums[-1] > nums[0]


def detect_tilt(events, n_losses=3, window_minutes=5.0, require_rising=True,
                strict_rising=True):
    """Inspect the most recent events for the tilt pattern.

    ``events``: iterable of dicts with keys ``timestamp``, ``is_win``,
    ``staked`` (most recent LAST). Returns a dict with at least ``triggered``
    and ``reason``.
    """
    norm = []
    for e in events:
        norm.append({
            "dt": _to_dt(e.get("timestamp")),
            "is_win": bool(e.get("is_win")),
            "staked": float(e.get("staked", 0) or 0),
        })
    if len(norm) < n_losses:
        return {"triggered": False, "reason": "data kurang"}
    tail = norm[-n_losses:]
    if any(e["is_win"] for e in tail):
        return {"triggered": False, "reason": "ada menang di jendela"}
    if any(e["staked"] <= 0 for e in tail):
        return {"triggered": False, "reason": "ada ronde tanpa taruhan"}
    dts = [e["dt"] for e in tail]
    if any(d is None for d in dts):
        return {"triggered": False, "reason": "timestamp tidak valid"}
    span = (dts[-1] - dts[0]).total_seconds() / 60.0
    if span > window_minutes:
        return {"triggered": False, "reason": "di luar jendela waktu"}
    if require_rising and not is_rising([e["staked"] for e in tail],
                                       strict=strict_rising):
        return {"triggered": False, "reason": "taruhan tidak naik"}
    return {
        "triggered": True,
        "reason": (f"{n_losses} kalah beruntun dalam {span:.1f} menit"
                   + (" dengan taruhan menaik" if require_rising else "")),
        "last_ts": dts[-1],
        "n_losses": n_losses,
        "span_minutes": span,
    }


class TiltDetector:
    """Stateful anti-tilt guard with a mandatory cooldown."""

    def __init__(self, n_losses=3, window_minutes=5.0, cooldown_seconds=60.0,
                 require_rising=True, strict_rising=True, enabled=True,
                 buffer_size=20):
        self.n_losses = int(n_losses)
        self.window_minutes = float(window_minutes)
        self.cooldown_seconds = float(cooldown_seconds)
        self.require_rising = bool(require_rising)
        self.strict_rising = bool(strict_rising)
        self.enabled = bool(enabled)
        self.buffer_size = max(int(buffer_size), self.n_losses)
        self._events = []
        self.cooldown_until = None
        self.last_trigger = None
        self.trigger_count = 0

    def reset(self):
        self._events = []
        self.cooldown_until = None
        self.last_trigger = None
        self.trigger_count = 0

    def clear_cooldown(self):
        """User acknowledged the warning / took a break."""
        self.cooldown_until = None

    def record(self, timestamp, is_win, staked):
        """Register a finished round; may trip a cooldown. Returns a status dict."""
        dt = _to_dt(timestamp) or datetime.now()
        self._events.append({
            "timestamp": dt, "is_win": bool(is_win),
            "staked": float(staked or 0),
        })
        if len(self._events) > self.buffer_size:
            self._events = self._events[-self.buffer_size:]
        if not self.enabled:
            st = self.status(now=dt)
            st["reason"] = "nonaktif"
            return st
        res = detect_tilt(self._events, n_losses=self.n_losses,
                          window_minutes=self.window_minutes,
                          require_rising=self.require_rising,
                          strict_rising=self.strict_rising)
        if res.get("triggered"):
            self.cooldown_until = dt + timedelta(seconds=self.cooldown_seconds)
            self.last_trigger = res
            self.trigger_count += 1
        st = self.status(now=dt)
        st["tilt_triggered"] = bool(res.get("triggered"))
        st["reason"] = res.get("reason")
        return st

    def status(self, now=None):
        """Current cooldown status. Auto-expires a finished cooldown."""
        now = _to_dt(now) or datetime.now()
        base = {"in_cooldown": False, "remaining_seconds": 0.0,
                "tilt_triggered": False, "trigger_count": self.trigger_count,
                "enabled": self.enabled}
        if self.cooldown_until is None:
            return base
        remaining = (self.cooldown_until - now).total_seconds()
        if remaining <= 0:
            self.cooldown_until = None
            return base
        base["in_cooldown"] = True
        base["remaining_seconds"] = remaining
        return base

    def is_in_cooldown(self, now=None):
        return self.status(now=now)["in_cooldown"]
