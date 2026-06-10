"""PROMPT 13 smoke test: analytics charts build + export without a GUI.

Runs headless (no tkinter). Verifies the four figures build from dummy data,
the data reducers are sane, empty data degrades gracefully, and PNG + PDF
exports produce non-empty files.
"""
import os
import random
import tempfile

from core import analytics_charts as ac

VALID = [1, 2, 5, 8, 10, 15, 20, 30, 40]
PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def make_history(n=120, seed=7):
    rng = random.Random(seed)
    engines = ["Ensemble", "AI-Optimal", "Markov"]
    hist = []
    for i in range(n):
        eng = engines[i % len(engines)]
        actual = rng.choice(VALID)
        # two bets per round with confidences
        picks = rng.sample(VALID, 2)
        bets = [{"number": p, "token_bet": 5,
                 "confidence": round(rng.uniform(0.05, 0.6), 3)} for p in picks]
        is_win = actual in picks
        profit = rng.choice([20, 50, -10, -10, -10]) if is_win else -10
        hist.append({
            "timestamp": f"2026-06-09T01:{i % 60:02d}:00",
            "actual_number": actual,
            "predicted_number": picks[0],
            "profit_change": profit,
            "is_win": is_win,
            "bets": bets,
            "engine_used": eng,
        })
    return hist


def main():
    hist = make_history()
    data = {"history": hist, "current_capital": 1000}

    # --- reducers ---
    rb = ac.reliability_bins(hist, n_bins=10)
    check("reliability_bins returns per-engine bins", len(rb) >= 1)
    check("reliability bins are (conf, hit, count) triples",
          all(len(t) == 3 for pts in rb.values() for t in pts))
    check("reliability hit-rate within [0,1]",
          all(0.0 <= t[1] <= 1.0 for pts in rb.values() for t in pts))

    dev = ac.per_number_deviation(hist, VALID)
    check("deviation has every valid number per engine",
          all(set(row.keys()) == set(VALID) for row in dev.values()))
    check("deviation = observed - predicted",
          all(abs(t[2] - (t[1] - t[0])) < 1e-9
              for row in dev.values() for t in row.values()))

    eq, peaks, dd = ac.equity_series(hist)
    check("equity series length matches history", len(eq) == len(hist))
    check("running peak monotonic non-decreasing",
          all(peaks[i] >= peaks[i - 1] for i in range(1, len(peaks))))
    check("drawdown never negative", all(d >= -1e-9 for d in dd))
    check("drawdown = peak - equity",
          all(abs(dd[i] - (peaks[i] - eq[i])) < 1e-9 for i in range(len(eq))))

    rw = ac.rolling_winrate(hist, window=50)
    check("rolling winrate length matches", len(rw) == len(hist))
    check("rolling winrate within [0,100]", all(0.0 <= v <= 100.0 for v in rw))

    # --- figures ---
    figs = ac.build_all_figures(data, VALID, window=50, baseline=100.0 / 9)
    check("build_all_figures returns 4 charts", len(figs) == 4)
    check("all four expected names present",
          set(figs.keys()) == {"reliability", "deviation_heatmap",
                               "equity_curve", "rolling_winrate"})
    check("each figure has >=1 axes", all(len(f.axes) >= 1 for f in figs.values()))

    # --- empty data graceful ---
    empty = ac.build_all_figures({"history": []}, VALID)
    check("empty data still builds 4 figures", len(empty) == 4)

    # --- export PNG ---
    tmp = tempfile.mkdtemp()
    base = os.path.join(tmp, "audit_charts")
    pngs = ac.export_charts(figs, base, fmt="png")
    check("PNG export wrote 4 files", len(pngs) == 4)
    check("all PNG files exist and non-empty",
          all(os.path.exists(p) and os.path.getsize(p) > 0 for p in pngs))

    # --- export PDF (single combined file) ---
    pdfs = ac.export_charts(figs, base, fmt="pdf")
    check("PDF export wrote 1 file", len(pdfs) == 1)
    check("PDF file exists and non-empty",
          os.path.exists(pdfs[0]) and os.path.getsize(pdfs[0]) > 0)

    # --- convenience wrapper ---
    w = ac.export_audit_charts(data, base + "_w", VALID, fmt="png")
    check("export_audit_charts wrote 4 PNGs", len(w) == 4)

    print()
    if FAIL == 0:
        print(f"ALL PASSED ({PASS} checks)")
    else:
        print(f"{FAIL} FAILED, {PASS} passed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
