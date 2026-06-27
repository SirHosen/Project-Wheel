# -*- coding: utf-8 -*-
"""[EXPERIMENTAL / NOT WIRED INTO APP FLOW] core/risk_manager.py - Adaptive bankroll risk control (PROMPT 5).

Kept for experimentation (see experimental/README.md). This class is currently
NOT connected to the ViewModel; the live app sizes stakes via core/betting.py
(net_kelly_portfolio). Do not assume it affects in-app betting (audit V3 dead-code).

The single biggest cause of ruin on a near-random game is NOT a bad model --
it's betting too much after losses (chasing) and pressing too hard after wins.
This manager scales the base risk_pct up/down from objective bankroll state and
hard-stops trading when limits are breached. It is intentionally conservative:
when in doubt it bets LESS.

Levers (all configurable):
  - Soft drawdown (default 0.20): start linearly cutting stake.
  - Hard drawdown (default 0.35): STOP. No new bets.
  - Losing-streak brake (default 5 losses): halve stake.
  - Winning-streak boost (default 8 wins): modest +25% (capped) -- never wild.
  - Daily stop (default 0.25 of the day's starting bankroll): STOP for the day.

State persists to models/risk_state.json (additive / drop-in).
"""
import json
import os
from datetime import datetime, date


class AdaptiveRiskManager:
    def __init__(self, starting_bankroll, soft_dd=0.20, hard_dd=0.35,
                 loss_streak_brake=5, win_streak_boost=8, daily_stop=0.25,
                 brake_factor=0.5, boost_factor=1.25, max_multiplier=1.5,
                 path=None):
        if path is None:
            from config import settings
            path = getattr(settings, "RISK_STATE_PATH", "runtime/risk_state.json")
        sb = float(starting_bankroll) if starting_bankroll else 0.0
        self.starting_bankroll = sb
        self.peak = sb
        self.current = sb
        self.soft_dd = soft_dd
        self.hard_dd = hard_dd
        self.loss_streak_brake = loss_streak_brake
        self.win_streak_boost = win_streak_boost
        self.daily_stop = daily_stop
        self.brake_factor = brake_factor
        self.boost_factor = boost_factor
        self.max_multiplier = max_multiplier
        self.path = path

        self.loss_streak = 0
        self.win_streak = 0
        self._day = None
        self._day_start_bankroll = sb
        self._manual_daily_lock = False

    # ------------------------------------------------------------------ #
    def _roll_day(self, when):
        # NOTE: datetime is a subclass of date, so check datetime FIRST.
        if when is None:
            when = datetime.now()
        if isinstance(when, datetime):
            d = when.date()
        elif isinstance(when, date):
            d = when
        else:
            d = datetime.now().date()
        if self._day != d:
            self._day = d
            self._day_start_bankroll = self.current
            self._manual_daily_lock = False

    @property
    def drawdown(self):
        if self.peak <= 0:
            return 0.0
        return max(0.0, (self.peak - self.current) / self.peak)

    @property
    def daily_loss_frac(self):
        base = self._day_start_bankroll or self.starting_bankroll
        if base <= 0:
            return 0.0
        return max(0.0, (base - self.current) / base)

    # ------------------------------------------------------------------ #
    def update(self, profit, when=None, is_win=None):
        """Record one round's net profit and update streaks/bankroll."""
        when = when or datetime.now()
        self._roll_day(when)
        self.current += float(profit)
        if self.current > self.peak:
            self.peak = self.current
        won = is_win if is_win is not None else (profit > 0)
        if profit == 0 and is_win is None:
            # a pure no-bet / break-even round doesn't move streaks
            return self.status()
        if won:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
        return self.status()

    # ------------------------------------------------------------------ #
    def should_stop(self, when=None):
        self._roll_day(when or datetime.now())
        if self.drawdown >= self.hard_dd:
            return True, (f"STOP: drawdown {self.drawdown*100:.1f}% >= batas keras "
                          f"{self.hard_dd*100:.0f}%. Lindungi modal, jeda dulu.")
        if self.daily_loss_frac >= self.daily_stop or self._manual_daily_lock:
            return True, (f"STOP HARIAN: rugi {self.daily_loss_frac*100:.1f}% "
                          f">= {self.daily_stop*100:.0f}% modal awal hari ini.")
        return False, ""

    def risk_multiplier(self, when=None):
        stop, _ = self.should_stop(when)
        if stop:
            return 0.0
        m = 1.0
        dd = self.drawdown
        if dd >= self.soft_dd:
            span = max(1e-9, self.hard_dd - self.soft_dd)
            m *= max(0.0, (self.hard_dd - dd) / span)
        if self.loss_streak >= self.loss_streak_brake:
            m *= self.brake_factor
        if self.win_streak >= self.win_streak_boost:
            m = min(self.max_multiplier, m * self.boost_factor)
        return max(0.0, min(self.max_multiplier, m))

    def effective_risk_pct(self, base_risk_pct, when=None):
        return max(0.0, float(base_risk_pct) * self.risk_multiplier(when))

    def lock_day(self):
        self._manual_daily_lock = True

    # ------------------------------------------------------------------ #
    def status(self):
        stop, reason = self.should_stop()
        return {
            "current": self.current, "peak": self.peak,
            "drawdown": self.drawdown, "daily_loss_frac": self.daily_loss_frac,
            "loss_streak": self.loss_streak, "win_streak": self.win_streak,
            "risk_multiplier": self.risk_multiplier(),
            "should_stop": stop, "reason": reason,
        }

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({
                    "starting_bankroll": self.starting_bankroll,
                    "peak": self.peak, "current": self.current,
                    "loss_streak": self.loss_streak, "win_streak": self.win_streak,
                    "day": self._day.isoformat() if self._day else None,
                    "day_start_bankroll": self._day_start_bankroll,
                }, f)
            return True
        except Exception:
            return False

    def load(self):
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                s = json.load(f)
            self.starting_bankroll = s.get("starting_bankroll", self.starting_bankroll)
            self.peak = s.get("peak", self.peak)
            self.current = s.get("current", self.current)
            self.loss_streak = s.get("loss_streak", 0)
            self.win_streak = s.get("win_streak", 0)
            self._day_start_bankroll = s.get("day_start_bankroll", self.current)
            d = s.get("day")
            self._day = date.fromisoformat(d) if d else None
            return True
        except Exception:
            return False
