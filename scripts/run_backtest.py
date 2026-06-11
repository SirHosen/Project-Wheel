# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""run_backtest.py - Standalone walk-forward backtest on real data (PROMPT 2).

Usage:
    python scripts/run_backtest.py --history data/history.json --csv samples/1.csv samples/2.csv

Loads actual spin results from a tracker history.json and/or CSV files, then
grades every cheap (pure-Python) engine walk-forward and prints a comparison
table + writes backtest_results.csv.

Note: LSTM / Ensemble are excluded here because a fair walk-forward would
require retraining the network at every step (slow) and TensorFlow may be
absent. Use the in-app audit for those once enough data is logged.
"""
import argparse
import csv as _csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from core.backtest import WalkForwardBacktester


def _load_history_json(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return [h.get("actual_number") for h in data.get("history", [])]
    except Exception:
        return []


def _coerce(cell, valid):
    try:
        v = int(float((cell or "").strip()))
    except (ValueError, TypeError):
        return None
    return v if v in valid else None


def _load_csv(path, valid):
    """Read ONLY the spin results. Prefers the 'actual_number' column so we never
    double-count predicted_number / index columns."""
    out = []
    if not os.path.exists(path):
        print(f"[skip] csv not found: {path}")
        return out
    with open(path, newline="") as f:
        first = f.readline()
        f.seek(0)
        if "actual_number" in first:
            for row in _csv.DictReader(f):
                v = _coerce(row.get("actual_number"), valid)
                if v is not None:
                    out.append(v)
        else:
            # Headerless: take the first valid numeric column per row only.
            for row in _csv.reader(f):
                for cell in row:
                    v = _coerce(cell, valid)
                    if v is not None:
                        out.append(v)
                        break
    return out


def _build_engines():
    engines = {}
    try:
        from predictors.markov_engine import MarkovEngine
        engines["Markov"] = MarkovEngine()
    except Exception as e:
        print(f"[skip] Markov: {e}")
    try:
        from predictors.higher_order_markov import HigherOrderMarkovEngine
        engines["Markov-HO"] = HigherOrderMarkovEngine(max_order=4)
    except Exception as e:
        print(f"[skip] Markov-HO: {e}")
    try:
        from predictors.bayesian_optimal import BayesianOptimalEngine
        engines["AI-Optimal"] = BayesianOptimalEngine()
    except Exception as e:
        print(f"[skip] Bayesian: {e}")
    try:
        from predictors.legacy.heuristic_engine import HeuristicEngine
        engines["Heuristic"] = HeuristicEngine()
    except Exception as e:
        print(f"[skip] Heuristic: {e}")
    return engines


def _fmt_pct(x):
    return "-" if x is None else f"{x*100:5.1f}%"


def _fmt_num(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/history.json")
    ap.add_argument("--csv", nargs="*", default=[])
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--out", default="backtest_results.csv")
    args = ap.parse_args()

    valid = set(settings.VALID_NUMBERS)
    actuals = _load_history_json(args.history)
    for c in args.csv:
        actuals += _load_csv(c, valid)
    actuals = [a for a in actuals if a in valid]
    print(f"Total spin termuat: {len(actuals)} (warmup={args.warmup})\n")
    if len(actuals) < args.warmup + 10:
        print("Data kurang untuk backtest.")
        return

    bt = WalkForwardBacktester(actuals, settings.VALID_NUMBERS, settings.SPINWHEEL_SEQUENCE)
    rows = bt.compare_all_engines(_build_engines(), warmup=args.warmup)

    cols = [("engine", 11), ("n_eval", 7), ("top1", 7), ("top3", 7),
            ("baseline", 9), ("lift", 7), ("z", 7), ("brier", 7),
            ("logloss", 8), ("profit/u", 9), ("sharpe", 7), ("verdict", 10)]
    print("".join(name.ljust(w) for name, w in cols))
    print("-" * sum(w for _, w in cols))
    for r in rows:
        line = [
            str(r.get("engine", "")).ljust(11),
            str(r["n_eval"]).ljust(7),
            _fmt_pct(r["top1_acc"]).ljust(7),
            _fmt_pct(r["top3_acc"]).ljust(7),
            _fmt_pct(r["baseline_top1_acc"]).ljust(9),
            (_fmt_pct(r["lift"]) if r["lift"] is not None else "-").ljust(7),
            _fmt_num(r["z_score"], 2).ljust(7),
            _fmt_num(r["brier_score"]).ljust(7),
            _fmt_num(r["log_loss"]).ljust(8),
            (str(r["simulated_profit_unit_bet"]) if r["simulated_profit_unit_bet"] is not None else "-").ljust(9),
            _fmt_num(r["sharpe"], 2).ljust(7),
            str(r["verdict"]).ljust(10),
        ]
        print("".join(line))

    fields = ["engine", "n_eval", "top1_acc", "top3_acc", "baseline_top1_acc",
              "lift", "z_score", "brier_score", "log_loss",
              "simulated_profit_unit_bet", "sharpe", "verdict"]
    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDitulis: {args.out}")


if __name__ == "__main__":
    main()
