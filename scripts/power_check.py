#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Print a POWER table: for a hypothetical bias, how many spins until I could
detect it? Helps set realistic expectations before you start grinding data.

    python scripts/power_check.py --number 40 --boost 0.30
    python scripts/power_check.py --number 8 --boost 0.20 --trials 500
"""
import argparse

from core import power


def main(argv=None):
    p = argparse.ArgumentParser(description="Bias detection power table")
    p.add_argument("--number", type=int, default=40, help="Which number is (hypothetically) biased.")
    p.add_argument("--boost", type=float, default=0.30, help="Its true probability (e.g. 0.30).")
    p.add_argument("--trials", type=int, default=300, help="Monte-Carlo trials per sample size.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ns = [25, 50, 75, 100, 150, 200, 300, 400, 600]
    fair = 1.0 / 9
    print(f"[power] number={args.number} biased to p={args.boost:.3f} "
          f"(fair ~= {fair:.3f}); trials={args.trials}")
    print(f"{'n':>5}  {'bias_power':>10}  {'bet_power':>9}  {'bet_on_target':>13}")
    print("-" * 44)
    for n in ns:
        r = power.simulate_power(args.number, args.boost, n,
                                 trials=args.trials, seed=args.seed)
        print(f"{n:>5}  {r['bias_power']:>10.2f}  {r['bet_power']:>9.2f}  "
              f"{r['bet_on_target_power']:>13.2f}")
    n80, p80 = power.sample_size_for_power(args.number, args.boost,
                                           target_power=0.8, trials=args.trials,
                                           seed=args.seed)
    if n80:
        print(f"[power] ~80% chance to flag the bias by n={n80} (power={p80:.2f})")
    else:
        print(f"[power] even the largest tested n did not reach 80% (best={p80:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
