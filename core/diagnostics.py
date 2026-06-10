# -*- coding: utf-8 -*-
"""
core/diagnostics.py - Audit & diagnostics engine.

Turns the recorded game history into a precise, detailed report so upgrades,
updates and audits are easy and evidence-based. Pure standard library (no
TensorFlow / CustomTkinter) so it can run anywhere, including CI.

The report covers:
  - Session metadata + app version (traceability)
  - Bankroll: starting/current, realized profit, ROI, max drawdown
  - Performance: top-pick win-rate, betting win-rate, skip ratio
  - Per-round profit stats (best/worst/avg/std)
  - Streaks (max win / max loss / current)
  - Per-number table: observed vs theoretical wheel frequency + deviation
  - Fairness test: chi-square goodness-of-fit vs the 54-segment wheel
  - Autocorrelation: lag-1 repeat rate vs random expectation
  - Markov edge: walk-forward top-1 hit vs naive baseline (two-proportion z)
  - Data-quality flags + concrete recommendations for the next upgrade
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime

# Chi-square 0.05 critical values by degrees of freedom (df 1..20).
_CHI2_CRIT_05 = {
    1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592,
    7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307, 11: 19.675, 12: 21.026,
    13: 22.362, 14: 23.685, 15: 24.996, 16: 26.296, 17: 27.587, 18: 28.869,
    19: 30.144, 20: 31.410,
}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wheel_frequencies(sequence, valid_numbers):
    """Theoretical probability of each number from the physical wheel layout."""
    counts = Counter(sequence)
    length = len(sequence) or 1
    return {n: counts.get(n, 0) / length for n in valid_numbers}


def _max_streak(flags):
    """Longest run of truthy values."""
    best = cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _current_streak(flags):
    cur = 0
    for f in reversed(flags):
        if f:
            cur += 1
        else:
            break
    return cur


def _markov_walkforward(actuals, sequence, valid_numbers):
    """Walk-forward test of the real Markov engine's top pick vs the naive
    'always most-frequent number' baseline. Returns rates + two-proportion z.
    Lazily imports the engine (pure-Python, no heavy deps)."""
    n = len(actuals)
    result = {
        "rounds": 0, "markov_top1_rate": None, "baseline_top1_rate": None,
        "z": None, "verdict": "insufficient",
        "note": "Butuh >= 40 hasil untuk uji walk-forward.",
    }
    if n < 40:
        return result
    try:
        from predictors.markov_engine import MarkovEngine
        engine = MarkovEngine()
        prior = engine.prior
    except Exception:
        # Fallback: frequency prior from the wheel layout.
        wf = wheel_frequencies(sequence, valid_numbers)
        prior = wf
        engine = None
    base_top1 = max(prior, key=prior.get)

    mk_hits = base_hits = rounds = 0
    for i in range(6, n - 1):
        actual = actuals[i + 1]
        if engine is not None:
            preds = engine.predict_next(actuals[: i + 1])
            if preds and preds[0]["number"] == actual:
                mk_hits += 1
        if base_top1 == actual:
            base_hits += 1
        rounds += 1
    if rounds < 10:
        return result

    mk_rate = mk_hits / rounds
    base_rate = base_hits / rounds
    pooled = (mk_hits + base_hits) / (2 * rounds)
    denom = math.sqrt(max(1e-9, pooled * (1 - pooled) * (2 / rounds)))
    z = (mk_rate - base_rate) / denom if denom > 0 else 0.0
    verdict = "edge" if (z > 1.96 and mk_rate > base_rate) else "no_edge"
    return {
        "rounds": rounds,
        "markov_top1_rate": mk_rate,
        "baseline_top1_rate": base_rate,
        "baseline_number": base_top1,
        "z": z,
        "verdict": verdict,
        "note": (
            "Edge statistik terdeteksi (model > baseline)." if verdict == "edge"
            else "Belum ada edge: model tidak mengalahkan tebakan paling sering."
        ),
    }


def per_number_bet_stats(history):
    """Per-number betting outcomes from the full snapshot log (PROMPT 1).

    Returns {number: {bets, wins, total_staked, net_profit, hit_rate, roi}}.
    Net profit is attributed per number so it SUMS to realized profit: the
    winning number gains (payout - its stake); every staked number loses its
    stake. Needs the per-round "bets" snapshot (empty for legacy records).
    """
    try:
        from config.settings import calculate_reward
    except Exception:
        calculate_reward = lambda tb, n: (tb * (1 if n == 1 else n)) + tb
    stats = {}
    for rec in history:
        actual = rec.get("actual_number")
        for b in (rec.get("bets") or []):
            tb = b.get("token_bet", 0) or 0
            if tb <= 0:
                continue
            num = b.get("number")
            s = stats.setdefault(num, {"bets": 0, "wins": 0,
                                       "total_staked": 0, "net_profit": 0})
            s["bets"] += 1
            s["total_staked"] += tb
            if actual == num:
                s["wins"] += 1
                s["net_profit"] += calculate_reward(tb, num) - tb
            else:
                s["net_profit"] -= tb
    for s in stats.values():
        s["hit_rate"] = (s["wins"] / s["bets"] * 100) if s["bets"] else 0.0
        s["roi"] = (s["net_profit"] / s["total_staked"] * 100) if s["total_staked"] else 0.0
    return stats


def engine_backtest_summary(actuals, sequence, valid_numbers, warmup=40):
    """Walk-forward backtest of the cheap engines (PROMPT 2).

    Guarded + lazy so the audit never breaks if predictors can't be imported
    (e.g. TensorFlow missing). Returns a list of metric rows, or [].
    """
    try:
        from core.backtest import WalkForwardBacktester
        bt = WalkForwardBacktester(actuals, valid_numbers, sequence)
        engines = {}
        try:
            from predictors.markov_engine import MarkovEngine
            engines["Markov"] = MarkovEngine()
        except Exception:
            pass
        try:
            from predictors.higher_order_markov import HigherOrderMarkovEngine
            engines["Markov-HO"] = HigherOrderMarkovEngine(max_order=4)
        except Exception:
            pass
        try:
            from predictors.bayesian_optimal import BayesianOptimalEngine
            engines["AI-Optimal"] = BayesianOptimalEngine()
        except Exception:
            pass
        try:
            from predictors.legacy.heuristic_engine import HeuristicEngine
            engines["Heuristic"] = HeuristicEngine()
        except Exception:
            pass
        if not engines:
            return []
        return bt.compare_all_engines(engines, warmup=warmup)
    except Exception:
        return []


def engine_bet_distribution(history):
    """Aggregate performance grouped by the engine that made each call."""
    dist = {}
    for rec in history:
        eng = rec.get("engine_used") or "unknown"
        g = dist.setdefault(eng, {"rounds": 0, "bet_rounds": 0,
                                  "total_staked": 0, "net_profit": 0, "wins": 0})
        g["rounds"] += 1
        staked = sum((b.get("token_bet", 0) or 0) for b in (rec.get("bets") or []))
        g["total_staked"] += staked
        if staked > 0:
            g["bet_rounds"] += 1
        g["net_profit"] += rec.get("profit_change", 0) or 0
        if rec.get("is_win"):
            g["wins"] += 1
    return dist


def session_analysis(data, sequence, valid_numbers, gap_minutes=30, alpha=0.05):
    """Per-session stats + KS-based temporal-drift verdict (PROMPT 9).

    Splits history into sessions (idle gap >= gap_minutes) and checks whether
    the outcome distribution drifted between the early and late sessions.
    """
    from collections import Counter
    from core.sessions import (
        parse_ts, chi_square_gof, detect_session_drift,
    )
    history = data.get("history", []) if isinstance(data, dict) else []
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

    rows = []
    actuals_per_session = []
    for i, sess in enumerate(sessions):
        actuals = [r.get("actual_number") for r in sess if r.get("actual_number") is not None]
        actuals_per_session.append(actuals)
        n = len(actuals)
        cnt = Counter(actuals)
        rows.append({
            "session_id": i,
            "start": sess[0].get("timestamp") if sess else None,
            "n_spins": n,
            "dominant_num": cnt.most_common(1)[0][0] if cnt else None,
            "win_rate": (sum(1 for r in sess if r.get("is_win")) / n * 100.0) if n else 0.0,
            "profit": sum(r.get("profit_change", 0) for r in sess),
            "chi_square": chi_square_gof(actuals, valid_numbers, sequence),
        })
    drift = detect_session_drift(actuals_per_session, alpha=alpha)
    return {"n_sessions": len(sessions), "gap_minutes": gap_minutes,
            "sessions": rows, "drift": drift}


def harvest_analysis(data):
    """PROMPT 12: audit Variance Harvest rounds SEPARATELY from normal play.

    Reads records tagged mode=="harvest" and reports round count, profit,
    win-rate, big-multiplier hits, max drawdown, and a Sharpe-equivalent
    (mean per-round profit / population stddev). Returns None when the mode was
    never used, so the section is hidden.
    """
    import math
    history = (data or {}).get("history", []) or []
    rounds = [h for h in history if h.get("mode") == "harvest"]
    if not rounds:
        return None
    returns = [float(h.get("profit_change", 0) or 0) for h in rounds]
    n = len(returns)
    wins = sum(1 for h in rounds if h.get("is_win"))
    big_wins = sum(1 for h in rounds
                   if h.get("is_win") and int(h.get("actual_number", 0)) >= 10)
    profit = sum(returns)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in returns:
        cum += x
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    mean = profit / n
    var = sum((x - mean) ** 2 for x in returns) / n
    sd = math.sqrt(var)
    sharpe = 0.0 if sd == 0 else mean / sd
    return {
        "n_rounds": n,
        "profit": profit,
        "win_rate": (wins / n * 100.0) if n else 0.0,
        "wins": wins,
        "big_wins": big_wins,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_profit_per_round": mean,
    }


def bankroll_breakdown(data):
    """PROMPT 15: daily + per-session bankroll roll-up for the audit report."""
    from core.bankroll import daily_report, session_report, overall_summary
    return {
        "summary": overall_summary(data),
        "days": daily_report(data),
        "sessions": session_report(data),
    }


def compute_report(data, sequence, valid_numbers, app_version="dev"):
    """Build the full diagnostics dict from a tracker data blob."""
    history = list(data.get("history", []))
    n = len(history)
    actuals = [h.get("actual_number") for h in history]
    profits = [h.get("profit_change", 0) for h in history]
    wins = [bool(h.get("is_win")) for h in history]

    # --- Bankroll ---
    profit_total = sum(profits)
    current_capital = data.get("current_capital", 0)
    starting_capital = current_capital - profit_total
    roi = (profit_total / starting_capital) if starting_capital else None
    running = starting_capital
    peak = starting_capital
    max_dd = 0.0
    equity = []
    for p in profits:
        running += p
        equity.append(running)
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
    max_dd_pct = (max_dd / peak * 100) if peak else 0.0

    # --- Performance ---
    top_hits = sum(wins)
    top_win_rate = (top_hits / n * 100) if n else 0.0
    bet_idx = [i for i, p in enumerate(profits) if p != 0]
    n_bets = len(bet_idx)
    n_skips = n - n_bets
    bet_wins = sum(1 for i in bet_idx if wins[i])
    bet_win_rate = (bet_wins / n_bets * 100) if n_bets else 0.0
    realized_per_bet = _mean([profits[i] for i in bet_idx]) if n_bets else 0.0

    # --- Per-round profit stats ---
    round_stats = {
        "best": max(profits) if profits else 0,
        "worst": min(profits) if profits else 0,
        "avg": _mean(profits),
        "std": _std(profits),
    }

    # --- Streaks ---
    streaks = {
        "current_win": _current_streak(wins),
        "max_win": _max_streak(wins),
        "max_loss": _max_streak([not w for w in wins]),
    }

    # --- Per-number frequency + fairness ---
    obs = Counter(a for a in actuals if a is not None)
    wf = wheel_frequencies(sequence, valid_numbers)
    per_number = []
    chi2 = 0.0
    for num in valid_numbers:
        o = obs.get(num, 0)
        obs_pct = (o / n * 100) if n else 0.0
        theo_pct = wf.get(num, 0.0) * 100
        exp = wf.get(num, 0.0) * n
        if exp > 0:
            chi2 += (o - exp) ** 2 / exp
        per_number.append({
            "number": num,
            "observed": o,
            "observed_pct": obs_pct,
            "theoretical_pct": theo_pct,
            "deviation_pct": obs_pct - theo_pct,
        })
    df = max(1, len(valid_numbers) - 1)
    crit = _CHI2_CRIT_05.get(df)
    if n < 30 or crit is None:
        fairness_verdict = "insufficient"
    elif chi2 > crit:
        fairness_verdict = "biased"
    else:
        fairness_verdict = "fair"
    fairness = {
        "chi_square": chi2, "df": df, "critical_value_0_05": crit,
        "verdict": fairness_verdict, "sample_size": n,
    }

    # --- Lag-1 autocorrelation ---
    repeats = sum(1 for i in range(1, n) if actuals[i] == actuals[i - 1])
    rep_rate = (repeats / (n - 1) * 100) if n > 1 else 0.0
    # Expected repeat rate under independence = sum p_i^2 over observed dist.
    p_obs = {num: (obs.get(num, 0) / n) for num in valid_numbers} if n else {}
    expected_rep = sum(v ** 2 for v in p_obs.values()) * 100
    autocorrelation = {
        "lag1_repeat_pct": rep_rate,
        "expected_repeat_pct": expected_rep,
        "verdict": (
            "insufficient" if n < 40 else
            "sticky" if rep_rate > expected_rep + 5 else "random"
        ),
    }

    # --- Markov edge ---
    markov = _markov_walkforward([a for a in actuals if a is not None], sequence, valid_numbers)

    # --- Per-number bet stats + per-engine distribution (PROMPT 1) ---
    per_number_bets = per_number_bet_stats(history)
    engine_dist = engine_bet_distribution(history)

    # --- Per-engine walk-forward backtest (PROMPT 2, guarded/lazy) ---
    engine_backtest = engine_backtest_summary(
        [a for a in actuals if a is not None], sequence, valid_numbers
    )

    # --- Data-quality flags + recommendations ---
    timestamps = [h.get("timestamp") for h in history if h.get("timestamp")]
    flags = []
    if any(h.get("predicted_number") is None and p > 0 for h, p in zip(history, profits)):
        flags.append("Ada ronde menang tanpa predicted_number tercatat (cek pencatatan).")
    if not any(h.get("bets") for h in history):
        flags.append(
            "Belum ada snapshot taruhan penuh pada data ini (record lama). Mulai "
            "v1.8.0 angka taruhan, confidence, EV, dan support dicatat tiap ronde."
        )
    else:
        flags.append(
            "Snapshot taruhan penuh AKTIF (v1.8.0+): angka, token, confidence, EV, "
            "dan support tercatat tiap ronde -> audit per-angka di bagian 9."
        )

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "app_version": app_version,
            "total_rounds": n,
            "first_timestamp": timestamps[0] if timestamps else None,
            "last_timestamp": timestamps[-1] if timestamps else None,
        },
        "bankroll": {
            "starting_capital": starting_capital,
            "current_capital": current_capital,
            "realized_profit": profit_total,
            "roi_pct": (roi * 100) if roi is not None else None,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "equity_curve": equity,
        },
        "performance": {
            "top_pick_win_rate_pct": top_win_rate,
            "top_pick_hits": top_hits,
            "bet_rounds": n_bets,
            "skip_rounds": n_skips,
            "bet_win_rate_pct": bet_win_rate,
            "realized_profit_per_bet": realized_per_bet,
        },
        "round_stats": round_stats,
        "streaks": streaks,
        "per_number": per_number,
        "fairness": fairness,
        "autocorrelation": autocorrelation,
        "markov_edge": markov,
        "per_number_bets": per_number_bets,
        "engine_distribution": engine_dist,
        "engine_backtest": engine_backtest,
        "data_quality": {"flags": flags},
        "sessions": session_analysis(data, sequence, valid_numbers),
        "harvest": harvest_analysis(data),
        "bankroll_daily": bankroll_breakdown(data),
    }


def _pct(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}%"


def report_to_markdown(r):
    """Render the report dict as a human-readable Markdown document (ID)."""
    m = r["meta"]; b = r["bankroll"]; p = r["performance"]
    rs = r["round_stats"]; st = r["streaks"]; f = r["fairness"]
    ac = r["autocorrelation"]; mk = r["markov_edge"]
    L = []
    L.append("# Laporan Audit - Spin Wheel Predictor")
    L.append("")
    L.append(f"- Dibuat: {m['generated_at']}")
    L.append(f"- Versi app: {m['app_version']}")
    L.append(f"- Total putaran tercatat: {m['total_rounds']}")
    if m["first_timestamp"]:
        L.append(f"- Rentang data: {m['first_timestamp']} s/d {m['last_timestamp']}")
    L.append("")
    L.append("## 1. Bankroll")
    L.append(f"- Modal awal: {b['starting_capital']:.0f} token")
    L.append(f"- Modal sekarang: {b['current_capital']:.0f} token")
    L.append(f"- Profit terealisasi: {b['realized_profit']:+.0f} token")
    L.append(f"- ROI: {_pct(b['roi_pct'])}")
    L.append(f"- Max drawdown: {b['max_drawdown']:.0f} token ({b['max_drawdown_pct']:.1f}%)")
    L.append("")
    L.append("## 2. Performa")
    L.append(f"- Win-rate tebakan teratas: {_pct(p['top_pick_win_rate_pct'])} ({p['top_pick_hits']}/{m['total_rounds']})")
    L.append(f"- Ronde bertaruh: {p['bet_rounds']} | Ronde skip: {p['skip_rounds']}")
    L.append(f"- Win-rate saat bertaruh: {_pct(p['bet_win_rate_pct'])}")
    L.append(f"- Rata-rata profit per ronde bertaruh: {p['realized_profit_per_bet']:+.2f} token")
    L.append("")
    L.append("## 3. Statistik per ronde")
    L.append(f"- Terbaik: {rs['best']:+.0f} | Terburuk: {rs['worst']:+.0f} | Rata2: {rs['avg']:+.2f} | Std: {rs['std']:.2f}")
    L.append(f"- Streak menang sekarang: {st['current_win']} | Max menang: {st['max_win']} | Max kalah beruntun: {st['max_loss']}")
    L.append("")
    L.append("## 4. Distribusi angka vs roda (54 segmen)")
    L.append("| Angka | Muncul | Observasi % | Teori roda % | Deviasi |")
    L.append("|------:|-------:|------------:|-------------:|--------:|")
    for row in r["per_number"]:
        L.append(
            f"| {row['number']} | {row['observed']} | {row['observed_pct']:.1f}% | "
            f"{row['theoretical_pct']:.1f}% | {row['deviation_pct']:+.1f}% |"
        )
    L.append("")
    L.append("## 5. Uji kewajaran roda (chi-square)")
    L.append(f"- chi^2 = {f['chi_square']:.2f} | df = {f['df']} | kritis(0.05) = {f['critical_value_0_05']}")
    verdict_map = {"fair": "RODA WAJAR (acak) - tidak ada bias frekuensi signifikan.",
                   "biased": "TERDETEKSI BIAS frekuensi (chi^2 > kritis).",
                   "insufficient": "Belum cukup data (min 30 putaran)."}
    L.append(f"- Putusan: {verdict_map.get(f['verdict'], f['verdict'])}")
    L.append("")
    L.append("## 6. Autokorelasi (angka berturut sama)")
    L.append(f"- Rate ulang lag-1: {ac['lag1_repeat_pct']:.1f}% vs harapan acak {ac['expected_repeat_pct']:.1f}%")
    L.append(f"- Putusan: {ac['verdict']}")
    L.append("")
    L.append("## 7. Edge Markov (walk-forward)")
    if mk["markov_top1_rate"] is None:
        L.append(f"- {mk['note']}")
    else:
        L.append(f"- Akurasi Markov top-1: {mk['markov_top1_rate']*100:.1f}% vs baseline {mk['baseline_top1_rate']*100:.1f}% atas {mk['rounds']} ronde")
        L.append(f"- z = {mk['z']:.2f} -> {mk['note']}")
    L.append("")
    bt_rows = r.get("engine_backtest") or []
    if bt_rows:
        L.append("### Backtest walk-forward semua engine (data nyata)")
        L.append("| Engine | n | Top-1 | Top-3 | Baseline | Lift | z | Verdict |")
        L.append("|:------|--:|------:|------:|---------:|-----:|--:|:--------|")
        for br in bt_rows:
            if br.get("top1_acc") is None:
                L.append(f"| {br['engine']} | {br['n_eval']} | - | - | - | - | - | {br['verdict']} |")
                continue
            L.append(
                f"| {br['engine']} | {br['n_eval']} | {br['top1_acc']*100:.1f}% | "
                f"{br['top3_acc']*100:.1f}% | {br['baseline_top1_acc']*100:.1f}% | "
                f"{br['lift']*100:+.1f}% | {br['z_score']:.2f} | {br['verdict']} |"
            )
        L.append("")
    L.append("## 8. Kualitas data & rekomendasi audit")
    for fl in r["data_quality"]["flags"]:
        L.append(f"- {fl}")
    L.append("")
    L.append("## 9. Taruhan per-angka (dari snapshot lengkap)")
    pnb = r.get("per_number_bets") or {}
    if not pnb:
        L.append("- Belum ada snapshot taruhan tercatat (logging penuh aktif mulai v1.8.0).")
    else:
        L.append("| Angka | #Bet | Menang | Hit rate | Token dipasang | Profit bersih | ROI |")
        L.append("|------:|-----:|-------:|---------:|---------------:|--------------:|----:|")
        for num in sorted(pnb.keys(), key=lambda x: (x is None, x)):
            s = pnb[num]
            L.append(
                f"| {num} | {s['bets']} | {s['wins']} | {s['hit_rate']:.1f}% | "
                f"{s['total_staked']:.0f} | {s['net_profit']:+.0f} | {s['roi']:+.1f}% |"
            )
    L.append("")
    ed = r.get("engine_distribution") or {}
    if ed:
        L.append("### Performa per-engine")
        L.append("| Engine | Ronde | Ronde bet | Token dipasang | Profit bersih | Menang |")
        L.append("|:------|------:|----------:|---------------:|--------------:|-------:|")
        for eng in sorted(ed.keys()):
            g = ed[eng]
            L.append(
                f"| {eng} | {g['rounds']} | {g['bet_rounds']} | {g['total_staked']:.0f} | "
                f"{g['net_profit']:+.0f} | {g['wins']} |"
            )
        L.append("")
    sa = r.get("sessions") or {}
    L.append("## 11. Analisis per-sesi")
    srows = sa.get("sessions") or []
    if not srows:
        L.append("- Belum ada data sesi.")
    else:
        L.append(f"- Terdeteksi {sa.get('n_sessions', 0)} sesi (jeda antar-sesi >= {sa.get('gap_minutes', 30)} menit).")
        L.append("")
        L.append("| Sesi | Mulai | #Spin | Angka dominan | Win rate | Profit | chi^2 |")
        L.append("|-----:|:------|------:|--------------:|---------:|-------:|------:|")
        for s in srows:
            chi = "-" if s["chi_square"] is None else f"{s['chi_square']:.1f}"
            L.append(
                f"| {s['session_id']} | {s['start']} | {s['n_spins']} | {s['dominant_num']} | "
                f"{s['win_rate']:.1f}% | {s['profit']:+.0f} | {chi} |"
            )
        L.append("")
        dr = sa.get("drift") or {}
        if dr.get("p_value") is None:
            L.append(f"- Putusan drift: data belum cukup ({dr.get('reason', 'butuh >=2 sesi')}).")
        elif dr.get("drift"):
            L.append(
                f"- Putusan: **TERDETEKSI DRIFT ANTAR-SESI** (KS D={dr['ks_d']:.3f}, "
                f"p={dr['p_value']:.4f} < {dr['alpha']}). Distribusi hasil berubah dari "
                f"sesi awal ke sesi akhir - jangan campur semua data jadi satu pool."
            )
        else:
            L.append(
                f"- Putusan: tidak ada drift signifikan (KS D={dr['ks_d']:.3f}, "
                f"p={dr['p_value']:.4f} >= {dr['alpha']}). Perilaku roda konsisten antar-sesi."
            )
    L.append("")

    hv = r.get("harvest")
    if hv:
        L.append("## 12. Variance Harvest (mode opt-in)")
        L.append(
            "> CATATAN JUJUR: mode ini SENGAJA pasang taruhan kecil di angka "
            "multiplier tinggi (lottery tickets). Secara jangka panjang tetap "
            "-EV; profit apa pun di sini adalah variance (keberuntungan), bukan edge."
        )
        L.append("")
        L.append("| Ronde harvest | Profit | Avg/ronde | Win rate | Big-win (mult>=10) | Max drawdown | Sharpe-eq |")
        L.append("|--------------:|-------:|----------:|---------:|-------------------:|-------------:|----------:|")
        L.append(
            f"| {hv['n_rounds']} | {hv['profit']:+.0f} | {hv['avg_profit_per_round']:+.2f} | "
            f"{hv['win_rate']:.1f}% | {hv['big_wins']} | {hv['max_drawdown']:.0f} | {hv['sharpe']:+.3f} |"
        )
        L.append("")
        if hv["profit"] > 0:
            L.append(
                "- Profit harvest saat ini POSITIF, tapi ingat: ini efek variance "
                "jangka pendek, bukan bukti sistem menang. Sharpe-eq mendekati 0 "
                "artinya imbal-hasil tidak berbeda nyata dari nol."
            )
        else:
            L.append(
                "- Profit harvest saat ini negatif - inilah biaya \"tiket lotre\" "
                "yang diharapkan. Mode ini untuk potensi variance, bukan profit konsisten."
            )
        L.append("")

    bd = r.get("bankroll_daily") or {}
    bsum = bd.get("summary") or {}
    bdays = bd.get("days") or []
    L.append("## 13. Bankroll harian & per-sesi")
    if not bdays:
        L.append("- Belum ada data ber-timestamp untuk dirinci per hari.")
    else:
        L.append(
            f"- {bsum.get('n_days', 0)} hari aktif | "
            f"{bsum.get('profit_days', 0)} hari hijau / {bsum.get('loss_days', 0)} merah | "
            f"profit total {bsum.get('total_profit', 0):+.0f} token"
        )
        bd_best = bsum.get("best_day") or {}
        bd_worst = bsum.get("worst_day") or {}
        L.append(
            f"- Hari terbaik {bd_best.get('date', '-')} ({bd_best.get('profit', 0):+.0f}) | "
            f"terburuk {bd_worst.get('date', '-')} ({bd_worst.get('profit', 0):+.0f}) | "
            f"rentet rugi terpanjang {bsum.get('longest_losing_streak', 0)} hari"
        )
        L.append("")
        L.append("| Tanggal | Ronde | Win rate | Profit | Kumulatif |")
        L.append("|:--------|------:|---------:|-------:|----------:|")
        for d in bdays:
            L.append(
                f"| {d['date']} | {d['rounds']} | {d['win_rate']:.0f}% | "
                f"{d['profit']:+.0f} | {d['cum_profit']:+.0f} |"
            )
        L.append("")
        L.append(
            "> CATATAN JUJUR: hari hijau bukan bukti edge. Di roda adil, P/L "
            "harian didominasi variance; gunakan rincian ini untuk audit "
            "drawdown dan konsentrasi profit, bukan sebagai sinyal menang."
        )
        L.append("")
    return "\n".join(L)


def export_audit_bundle(data, md_path, sequence, valid_numbers, app_version="dev"):
    """Write <name>.md + <name>.json + <name>_raw.csv next to md_path.
    Returns the list of files written."""
    report = compute_report(data, sequence, valid_numbers, app_version=app_version)
    base, _ = os.path.splitext(md_path)
    md_file = base + ".md"
    json_file = base + ".json"
    csv_file = base + "_raw.csv"

    with open(md_file, "w", encoding="utf-8") as fh:
        fh.write(report_to_markdown(report))
    with open(json_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    history = data.get("history", [])
    cols = ["timestamp", "actual_number", "predicted_number", "profit_change", "is_win"]
    import csv as _csv
    with open(csv_file, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in history:
            w.writerow({c: row.get(c) for c in cols})

    return [md_file, json_file, csv_file]
