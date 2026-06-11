import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""PROMPT 14 test: manual % as a Dirichlet prior (core/priors.py).

Verifies the weight formula, its limiting behaviours, convex blending,
backward-compatibility with the old fixed 0.35 weight at ~50 spins, and the
no-op path when the user supplied no manual input.
"""
from core import priors as P

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main():
    w = P.dirichlet_prior_weight

    # --- limiting behaviours ---
    check("strength 0 -> weight 0 (pure engine)", w(0, 50) == 0.0)
    check("data 0 -> weight 1 (pure manual / cold start)", w(27, 0) == 1.0)
    check("strength == data -> weight 0.5", approx(w(27, 27), 0.5))
    check("both 0 -> weight 0 (safe)", w(0, 0) == 0.0)

    # --- monotonicity ---
    check("weight increases with strength", w(10, 50) < w(40, 50))
    check("weight decreases as data grows",
          w(27, 10) > w(27, 50) > w(27, 200))

    # --- backward-compat: default 27 at ~50 spins ~= old 0.35 ---
    check("default strength 27 @ 50 spins ~= 0.35", approx(w(27, 50), 27 / 77, 1e-9)
          and abs(w(27, 50) - 0.35) < 0.012)

    # --- convex blend stays between engine and manual ---
    b = P.blend_confidence
    lo, hi = b(0.2, 0.8, 0.5), 0.5
    check("blend midpoint correct", approx(b(0.2, 0.8, 0.5), 0.5))
    check("blend at w=0 = engine", approx(b(0.2, 0.8, 0.0), 0.2))
    check("blend at w=1 = manual", approx(b(0.2, 0.8, 1.0), 0.8))
    check("blend clamps weight >1", approx(b(0.2, 0.8, 5.0), 0.8))
    check("blend between bounds",
          all(0.2 <= b(0.2, 0.8, ww) <= 0.8 for ww in (0.0, 0.25, 0.5, 0.75, 1.0)))

    # --- has_manual_input ---
    check("has_manual_input false on all-zero",
          not P.has_manual_input({1: 0.0, 2: 0.0}))
    check("has_manual_input false on empty/None",
          (not P.has_manual_input({})) and (not P.has_manual_input(None)))
    check("has_manual_input true when any positive",
          P.has_manual_input({1: 0.0, 2: 30.0}))

    # --- apply_manual_prior mutates correctly ---
    preds = [{"number": 1, "confidence": 0.50},
             {"number": 2, "confidence": 0.30},
             {"number": 5, "confidence": 0.20}]
    manual = {1: 0.0, 2: 0.0, 5: 100.0}  # 100% on number 5
    out, weight = P.apply_manual_prior(preds, manual, strength=27, data_count=27)
    check("apply returns weight 0.5 at strength==data", approx(weight, 0.5))
    # number 5: 0.5*0.20 + 0.5*1.0 = 0.60
    c5 = next(p["confidence"] for p in out if p["number"] == 5)
    check("number 5 confidence blended to 0.60", approx(c5, 0.60))
    # number 1: 0.5*0.50 + 0.5*0.0 = 0.25
    c1 = next(p["confidence"] for p in out if p["number"] == 1)
    check("number 1 confidence blended to 0.25", approx(c1, 0.25))

    # --- no manual input -> unchanged ---
    preds2 = [{"number": 1, "confidence": 0.5}]
    out2, weight2 = P.apply_manual_prior(preds2, {1: 0.0}, strength=27, data_count=10)
    check("no manual input leaves confidence unchanged",
          approx(out2[0]["confidence"], 0.5))

    # --- clamp_strength ---
    check("clamp below min", P.clamp_strength(1, 5, 100, 27) == 5)
    check("clamp above max", P.clamp_strength(500, 5, 100, 27) == 100)
    check("clamp in range passthrough", P.clamp_strength(42, 5, 100, 27) == 42)
    check("clamp bad input -> default", P.clamp_strength("x", 5, 100, 27) == 27)

    print()
    if FAIL == 0:
        print(f"ALL PASSED ({PASS} checks)")
    else:
        print(f"{FAIL} FAILED, {PASS} passed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
