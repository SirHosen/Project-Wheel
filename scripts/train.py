# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Train the LSTM + run an HONEST walk-forward backtest.

Data sources (pick one):
    python scripts/train.py                       # use your logged results (runtime/observations.csv)
    python scripts/train.py --demo fair --n 1500   # practice on a FAIR wheel (model should NOT beat baseline)
    python scripts/train.py --demo biased --n 1500 # a biased wheel (bias tracker should flag an edge)
    python scripts/train.py --demo markov --n 1500 # a sequential pattern (LSTM SHOULD beat baseline)

The point of --demo is to SEE the difference between data the AI can learn from
and data it cannot. A real wheel behaves like 'fair': no sequential edge.

The physics + bias checks run even without PyTorch; only the LSTM parts need it.
"""
import argparse


def _load_demo(kind, n, seed):
    from ai import dataset
    if kind == "fair":
        return dataset.synthetic_fair(n, seed=seed)
    if kind == "biased":
        return dataset.synthetic_biased(n, seed=seed)
    if kind == "markov":
        return dataset.synthetic_markov(n, seed=seed)
    raise ValueError(f"unknown --demo {kind!r} (fair/biased/markov)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train LSTM + honest backtest")
    p.add_argument("--demo", choices=["fair", "biased", "markov"], default=None,
                   help="Use synthetic practice data instead of the results log.")
    p.add_argument("--n", type=int, default=1500, help="Number of synthetic spins for --demo.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    p.add_argument("--no-save", action="store_true", help="Do not save the trained model.")
    p.add_argument("--backtest-only", action="store_true", help="Skip final save-train; only backtest.")
    args = p.parse_args(argv)

    from ai import lstm
    from ai.device import status_line as device_status
    from core.bias_tracker import OnlineBiasTracker
    from core.physics import predictability_report

    # Show the compute backend up front (GREEN=GPU / RED=CPU in the panel).
    print(f"[device] {device_status()}")

    # --- load the result history ------------------------------------------
    if args.demo:
        numbers = _load_demo(args.demo, args.n, args.seed)
        print(f"[train] demo='{args.demo}'  spins={len(numbers)}")
    else:
        from app.observation_log import load_numbers
        numbers = load_numbers()
        print(f"[train] loaded {len(numbers)} results from the observations log")
    if len(numbers) < 100:
        print("[train] WARNING: < 100 results -> results will be noisy. "
              "Collect more with auto_watch, or use --demo to practice.")

    # --- 1) physics reality check (no TF needed) --------------------------
    print(f"[physics] {predictability_report()['verdict']}")

    # --- 2) Bayesian bias check (no TF needed) ----------------------------
    bt = OnlineBiasTracker()
    bt.observe_many(numbers)
    print(f"[bias] {bt.summary()['recommendation']}  |  {bt.bias_test()}")
    # Fully Bayesian EV: report the FULL posterior uncertainty (not just a bound)
    # for the most-observed number, incl. P(EV>0).
    try:
        post = bt.posterior()
        ev = bt.ev_samples()
        top_n = max(ev, key=lambda k: post[k]["mean"])
        e = ev[top_n]
        print(f"[bias] Bayesian EV of {top_n}: mean={e['ev_mean']:+.3f} "
              f"[{e['ev_lo']:+.3f}, {e['ev_hi']:+.3f}]  "
              f"P(EV>0)={e['prob_positive']:.2f}")
    except Exception as ev_err:  # numpy missing etc. -- never block the run
        print(f"[bias] Bayesian EV sampling skipped: {ev_err}")

    # --- 2b) strategy comparison + probabilistic calibration --------------
    # Runs WITHOUT PyTorch (baselines + frequency-model calibration); the LSTM
    # rows/metrics are added automatically when PyTorch is installed.
    try:
        table = lstm.walk_forward_table(numbers)
        print("[backtest] strategy comparison (walk-forward):")
        print(lstm.format_backtest_table(table))
        print(lstm.format_probabilistic_report(lstm.probabilistic_report(numbers)))
    except ValueError as e:
        print(f"[backtest] comparison skipped: {e}")

    # --- 3) + 4) the AI parts (need PyTorch) ------------------------------
    if not lstm.available():
        print("[train] " + lstm.status_line())
        print("[train] Skipping LSTM train/backtest. Install with: pip install -r requirements-ai.txt")
        return 0  # the honest non-AI checks above still ran

    if not args.backtest_only:
        try:
            _, hist = lstm.train(numbers,
                                 epochs=(args.epochs or lstm.EPOCHS),
                                 verbose=2,
                                 save_path=None if args.no_save else lstm.MODEL_PATH)
            vals = (hist.get("val_accuracy") if hist else None) or hist.get("accuracy") or [0]
            best = max(vals)
            print(f"[train] done. best val_accuracy={best:.3f}")
            if not args.no_save:
                print(f"[train] model saved -> {lstm.MODEL_PATH}")
        except ValueError as e:
            print(f"[train] skipped final train: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
