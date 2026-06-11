# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""PROMPT 10: verify Heuristic is retired to predictors/legacy/.

- canonical engine lives in predictors.legacy.heuristic_engine and still works
- old import path still works but emits a DeprecationWarning (back-compat shim)
- main engine dropdown no longer lists "Heuristic"
- the main window exposes the hidden Lab Mode handler
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_failures = []


def check(name, cond):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")
        _failures.append(name)


def test_legacy_canonical():
    from predictors.legacy.heuristic_engine import HeuristicEngine
    eng = HeuristicEngine()
    preds = eng.predict_next([1, 2, 5, 1, 2])
    check("legacy engine returns one row per valid number",
          len(preds) == len(eng.valid_numbers))
    s = sum(p["confidence"] for p in preds)
    check("legacy engine confidences ~normalize to 1", abs(s - 1.0) < 1e-6)
    # empty history -> uniform
    upreds = eng.predict_next([])
    confs = {round(p["confidence"], 9) for p in upreds}
    check("empty history -> uniform distribution", len(confs) == 1)


def test_legacy_file_location():
    import predictors.legacy.heuristic_engine as mod
    path = os.path.abspath(mod.__file__)
    check("canonical module physically lives under predictors/legacy/",
          os.path.join("predictors", "legacy") in path)


def test_deprecation_shim():
    # the old import path must still resolve (back-compat) AND warn
    for m in list(sys.modules):
        if m == "predictors.heuristic_engine":
            del sys.modules[m]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from predictors.heuristic_engine import HeuristicEngine as ShimEngine
        from predictors.legacy.heuristic_engine import HeuristicEngine as RealEngine
        check("shim re-exports the same class as legacy", ShimEngine is RealEngine)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    check("old import path emits DeprecationWarning", len(dep) >= 1)


def test_dropdown_excludes_heuristic():
    with open(os.path.join("gui", "views", "main_window.py"), encoding="utf-8") as f:
        src = f.read()
    check("dropdown values line has no 'Heuristic'",
          'values=["AI-Optimal", "Ensemble", "Markov", "TF-LSTM"]' in src)
    check("main_window defines hidden Lab Mode handler",
          "_open_lab_mode" in src and "Control-Shift-L" in src)


def test_continuous_engine_has_no_heuristic():
    with open(os.path.join("core", "continuous_engine.py"), encoding="utf-8") as f:
        src = f.read().lower()
    check("continuous (ensemble) engine never references heuristic",
          "heuristic" not in src)


if __name__ == "__main__":
    print("== PROMPT 10: retire Heuristic to legacy ==")
    test_legacy_canonical()
    test_legacy_file_location()
    test_deprecation_shim()
    test_dropdown_excludes_heuristic()
    test_continuous_engine_has_no_heuristic()
    if _failures:
        print(f"\nFAILED ({len(_failures)}): {_failures}")
        sys.exit(1)
    print("\nALL PASSED")
