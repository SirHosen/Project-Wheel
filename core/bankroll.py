# -*- coding: utf-8 -*-
"""PROMPT 15: daily & per-session bankroll reporting.

Pure aggregation over the recorded spin history. Groups the realized P/L by
calendar day and by play-session (idle gap >= gap_minutes) so the user can see
WHEN they actually made or lost tokens, not just one linear blob.

No external deps -- only the stdlib + core.sessions.parse_ts. GUI-agnostic and
fully unit-testable headless.

HONEST NOTE: this is bookkeeping, not an edge. Seeing green days does not mean
the system "works"; on a fair wheel daily P/L is dominated by variance. The
report exists so the user can audit drawdowns, streaks of losing days, and
whether profits are concentrated in a few lucky sessions.
"""
import math
from collections import OrderedDict

from core.sessions import parse_ts


def _history(data):
    """Accept either the full tracker dict or a bare history list."""
    if isinstance(data, dict):
        return data.get("history", []) or []
    return data or []


def _day_key(rec):
    """Calendar date (YYYY-MM-DD) for a record, or None if unparseable."""
    ts = parse_ts(rec.get("timestamp", ""))
    return ts.date().isoformat() if ts is not None else None


def _drawdown(returns):
    """Max peak-to-trough drawdown of a cumulative return series."""
    cum = peak = 0.0
    max_dd = 0.0
    for x in returns:
        cum += x
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def _bucket_stats(records):
    """Shared per-bucket aggregation (one day or one session)."""
    returns = [float(r.get("profit_change", 0) or 0) for r in records]
    n = len(records)
    wins = sum(1 for r in records if r.get("is_win"))
    # A "bet round" is one where tokens were actually staked.
    bet_rounds = 0
    for r in records:
        bets = r.get("bets") or []
        if any(float(b.get("token_bet", 0) or 0) > 0 for b in bets):
            bet_rounds += 1
        elif float(r.get("profit_change", 0) or 0) != 0:
            bet_rounds += 1
    profit = sum(returns)
    return {
        "rounds": n,
        "wins": wins,
        "win_rate": (wins / n * 100.0) if n else 0.0,
        "bet_rounds": bet_rounds,
        "profit": profit,
        "avg_profit": (profit / n) if n else 0.0,
        "best": max(returns) if returns else 0.0,
        "worst": min(returns) if returns else 0.0,
        "max_drawdown": _drawdown(returns),
    }


def daily_report(data):
    """Per-calendar-day bankroll rows, ascending by date.

    Each row carries the per-day stats plus a running ``cum_profit`` (realized
    P/L from the first recorded day through the end of that day).
    """
    history = _history(data)
    buckets = OrderedDict()
    for rec in history:
        key = _day_key(rec)
        if key is None:
            continue
        buckets.setdefault(key, []).append(rec)

    rows = []
    cum = 0.0
    for day in sorted(buckets.keys()):
        stats = _bucket_stats(buckets[day])
        cum += stats["profit"]
        stats["date"] = day
        stats["cum_profit"] = cum
        rows.append(stats)
    return rows


def split_sessions(history, gap_minutes=30):
    """Split records into sessions whenever the idle gap >= gap_minutes."""
    sessions = []
    cur = []
    prev_ts = None
    for rec in history:
        ts = parse_ts(rec.get("timestamp", ""))
        if cur and prev_ts is not None and ts is not None:
            if (ts - prev_ts).total_seconds() / 60.0 >= gap_minutes:
                sessions.append(cur)
                cur = []
        cur.append(rec)
        if ts is not None:
            prev_ts = ts
    if cur:
        sessions.append(cur)
    return sessions


def session_report(data, gap_minutes=30):
    """Per-session bankroll rows (idle gap >= gap_minutes defines a session)."""
    history = _history(data)
    rows = []
    for i, sess in enumerate(split_sessions(history, gap_minutes=gap_minutes)):
        stats = _bucket_stats(sess)
        stats["session_id"] = i
        stats["start"] = sess[0].get("timestamp") if sess else None
        stats["end"] = sess[-1].get("timestamp") if sess else None
        rows.append(stats)
    return rows


def overall_summary(data):
    """High-level roll-up across all recorded days."""
    days = daily_report(data)
    if not days:
        return {
            "n_days": 0, "total_profit": 0.0, "profit_days": 0, "loss_days": 0,
            "flat_days": 0, "avg_daily_profit": 0.0, "best_day": None,
            "worst_day": None, "win_rate": 0.0, "max_daily_drawdown": 0.0,
            "longest_losing_streak": 0,
        }
    profits = [d["profit"] for d in days]
    total = sum(profits)
    profit_days = sum(1 for x in profits if x > 0)
    loss_days = sum(1 for x in profits if x < 0)
    flat_days = sum(1 for x in profits if x == 0)
    total_rounds = sum(d["rounds"] for d in days)
    total_wins = sum(d["wins"] for d in days)
    best = max(days, key=lambda d: d["profit"])
    worst = min(days, key=lambda d: d["profit"])
    # Longest run of consecutive losing days.
    longest = cur = 0
    for x in profits:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    return {
        "n_days": len(days),
        "total_profit": total,
        "profit_days": profit_days,
        "loss_days": loss_days,
        "flat_days": flat_days,
        "avg_daily_profit": total / len(days),
        "best_day": {"date": best["date"], "profit": best["profit"]},
        "worst_day": {"date": worst["date"], "profit": worst["profit"]},
        "win_rate": (total_wins / total_rounds * 100.0) if total_rounds else 0.0,
        "max_daily_drawdown": _drawdown(profits),
        "longest_losing_streak": longest,
    }


def format_report_text(data, gap_minutes=30):
    """Plain-text daily + session bankroll report for the GUI panel."""
    days = daily_report(data)
    sess = session_report(data, gap_minutes=gap_minutes)
    s = overall_summary(data)
    L = []
    L.append("RINGKASAN BANKROLL")
    L.append("=" * 60)
    if s["n_days"] == 0:
        L.append("Belum ada data ber-timestamp untuk dilaporkan.")
        return "\n".join(L)
    L.append(f"Total hari aktif   : {s['n_days']}")
    L.append(f"Profit total       : {s['total_profit']:+.0f} token")
    L.append(f"Hari profit/rugi   : {s['profit_days']} hijau / {s['loss_days']} merah "
             f"/ {s['flat_days']} datar")
    L.append(f"Rata-rata per hari : {s['avg_daily_profit']:+.1f} token")
    L.append(f"Win-rate keseluruhan: {s['win_rate']:.1f}%")
    L.append(f"Hari terbaik       : {s['best_day']['date']} ({s['best_day']['profit']:+.0f})")
    L.append(f"Hari terburuk      : {s['worst_day']['date']} ({s['worst_day']['profit']:+.0f})")
    L.append(f"Max drawdown harian: {s['max_daily_drawdown']:.0f} token")
    L.append(f"Rentet hari rugi   : {s['longest_losing_streak']} hari beruntun")
    L.append("")
    L.append("PER HARI")
    L.append("-" * 60)
    L.append(f"{'Tanggal':<12}{'Ronde':>6}{'WinR':>7}{'Profit':>9}{'Kumulatif':>11}")
    for d in days:
        L.append(f"{d['date']:<12}{d['rounds']:>6}{d['win_rate']:>6.0f}%"
                 f"{d['profit']:>+9.0f}{d['cum_profit']:>+11.0f}")
    L.append("")
    L.append(f"PER SESI (jeda >= {gap_minutes} menit)")
    L.append("-" * 60)
    L.append(f"{'Sesi':<6}{'Mulai':<20}{'Ronde':>6}{'WinR':>7}{'Profit':>9}")
    for r in sess:
        start = (r["start"] or "")[:19]
        L.append(f"{r['session_id']:<6}{start:<20}{r['rounds']:>6}"
                 f"{r['win_rate']:>6.0f}%{r['profit']:>+9.0f}")
    L.append("")
    L.append("CATATAN JUJUR: hari hijau bukan bukti sistem menang. Di roda adil,")
    L.append("P/L harian didominasi variance. Pakai ini untuk audit drawdown &")
    L.append("apakah profit cuma menumpuk di segelintir sesi beruntung.")
    return "\n".join(L)
