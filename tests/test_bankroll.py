# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""PROMPT 15 tests: daily/per-session bankroll report + calendar heatmap."""
import os
import tempfile

from core import bankroll
from core import analytics_charts as ac

_checks = 0
_fail = 0


def check(name, cond):
    global _checks, _fail
    _checks += 1
    if cond:
        print(f"  PASS: {name}")
    else:
        _fail += 1
        print(f"  FAIL: {name}")
    assert cond, name  # pytest: surface failures as assertion errors


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


HISTORY = [
    {"timestamp": "2026-06-01T10:00:00", "actual_number": 5, "predicted_number": 5,
     "profit_change": 10, "is_win": True, "bets": [{"number": 5, "token_bet": 5}]},
    {"timestamp": "2026-06-01T10:02:00", "actual_number": 2, "predicted_number": 5,
     "profit_change": -5, "is_win": False, "bets": []},
    {"timestamp": "2026-06-01T11:00:00", "actual_number": 1, "predicted_number": 1,
     "profit_change": 3, "is_win": True, "bets": [{"number": 1, "token_bet": 3}]},
    {"timestamp": "2026-06-02T09:00:00", "actual_number": 8, "predicted_number": 2,
     "profit_change": -4, "is_win": False, "bets": [{"number": 2, "token_bet": 4}]},
    {"timestamp": "2026-06-02T09:05:00", "actual_number": 10, "predicted_number": 2,
     "profit_change": -6, "is_win": False, "bets": [{"number": 2, "token_bet": 6}]},
    {"timestamp": "2026-06-03T20:00:00", "actual_number": 20, "predicted_number": 20,
     "profit_change": 20, "is_win": True, "bets": [{"number": 20, "token_bet": 2}]},
]
DATA = {"history": HISTORY, "current_capital": 1018}


print("=== daily_report ===")
days = bankroll.daily_report(DATA)
check("3 hari", len(days) == 3)
check("urut tanggal", [d["date"] for d in days] ==
      ["2026-06-01", "2026-06-02", "2026-06-03"])
d0, d1, d2 = days
check("hari1 rounds=3", d0["rounds"] == 3)
check("hari1 wins=2", d0["wins"] == 2)
check("hari1 win_rate~66.7", approx(d0["win_rate"], 2 / 3 * 100))
check("hari1 profit=8", approx(d0["profit"], 8))
check("hari1 cum=8", approx(d0["cum_profit"], 8))
check("hari1 best=10", approx(d0["best"], 10))
check("hari1 worst=-5", approx(d0["worst"], -5))
check("hari1 bet_rounds=3", d0["bet_rounds"] == 3)
check("hari2 profit=-10", approx(d1["profit"], -10))
check("hari2 cum=-2", approx(d1["cum_profit"], -2))
check("hari2 win_rate=0", approx(d1["win_rate"], 0.0))
check("hari3 profit=20", approx(d2["profit"], 20))
check("hari3 cum=18", approx(d2["cum_profit"], 18))

print("=== session_report ===")
sess = bankroll.session_report(DATA, gap_minutes=30)
check("4 sesi (gap 30m)", len(sess) == 4)
check("sesi0 profit=5", approx(sess[0]["profit"], 5))
check("sesi0 rounds=2", sess[0]["rounds"] == 2)
check("sesi1 profit=3", approx(sess[1]["profit"], 3))
check("sesi2 profit=-10", approx(sess[2]["profit"], -10))
check("sesi3 profit=20", approx(sess[3]["profit"], 20))
check("session_id berurut", [s["session_id"] for s in sess] == [0, 1, 2, 3])
# Wider gap merges all of day-1 into one session.
sess_big = bankroll.session_report(DATA, gap_minutes=120)
check("gap 120m -> sesi0 gabung 3 ronde", sess_big[0]["rounds"] == 3)

print("=== overall_summary ===")
s = bankroll.overall_summary(DATA)
check("n_days=3", s["n_days"] == 3)
check("total_profit=18", approx(s["total_profit"], 18))
check("profit_days=2", s["profit_days"] == 2)
check("loss_days=1", s["loss_days"] == 1)
check("flat_days=0", s["flat_days"] == 0)
check("avg_daily=6", approx(s["avg_daily_profit"], 6.0))
check("best_day=hari3", s["best_day"]["date"] == "2026-06-03")
check("worst_day=hari2", s["worst_day"]["date"] == "2026-06-02")
check("win_rate=50", approx(s["win_rate"], 50.0))
check("max_daily_drawdown=10", approx(s["max_daily_drawdown"], 10))
check("longest_losing_streak=1", s["longest_losing_streak"] == 1)

print("=== empty history ===")
check("daily kosong", bankroll.daily_report({"history": []}) == [])
check("sesi kosong", bankroll.session_report([]) == [])
empt = bankroll.overall_summary({"history": []})
check("summary kosong n_days=0", empt["n_days"] == 0)
check("summary kosong best=None", empt["best_day"] is None)

print("=== format_report_text ===")
txt = bankroll.format_report_text(DATA)
check("teks ada RINGKASAN", "RINGKASAN BANKROLL" in txt)
check("teks ada PER HARI", "PER HARI" in txt)
check("teks ada PER SESI", "PER SESI" in txt)
check("teks ada catatan jujur", "CATATAN JUJUR" in txt)
check("teks kosong aman", "Belum ada" in bankroll.format_report_text({"history": []}))

print("=== records as bare list (no dict) ===")
check("daily dari list", len(bankroll.daily_report(HISTORY)) == 3)

print("=== unparseable timestamp skipped ===")
bad = {"history": HISTORY + [{"timestamp": "", "profit_change": 99, "is_win": True}]}
check("timestamp kosong diabaikan di daily",
      len(bankroll.daily_report(bad)) == 3)

print("=== charts ===")
fig_cal = ac.build_calendar_heatmap_figure(DATA)
check("kalender punya axes", len(fig_cal.axes) >= 1)
check("kalender punya image", len(fig_cal.axes[0].images) >= 1)
fig_pnl = ac.build_daily_pnl_figure(DATA)
check("daily pnl punya bar", len(fig_pnl.axes[0].patches) >= 3)
figs = ac.build_bankroll_figures(DATA)
check("build_bankroll_figures 2 grafik", set(figs.keys()) ==
      {"daily_pnl", "calendar_heatmap"})
# Empty data still returns figures (no crash).
fe = ac.build_calendar_heatmap_figure({"history": []})
check("kalender kosong tetap Figure", fe is not None and len(fe.axes) >= 1)

print("=== export ===")
tmp = tempfile.mkdtemp()
png_files = ac.export_bankroll_charts(DATA, os.path.join(tmp, "bk.png"), fmt="png")
check("export png 2 file", len(png_files) == 2)
check("png files ada", all(os.path.exists(p) for p in png_files))
pdf_files = ac.export_bankroll_charts(DATA, os.path.join(tmp, "bk.pdf"), fmt="pdf")
check("export pdf 1 file", len(pdf_files) == 1)
check("pdf file ada", os.path.exists(pdf_files[0]))

print(f"\n{_checks - _fail}/{_checks} checks passed")
if _fail:
    raise SystemExit(f"{_fail} FAILED")
print("ALL GREEN")
