# -*- coding: utf-8 -*-
"""Test PROMPT 6: BMA ensemble weights + optional stacking."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from predictors.markov_engine import MarkovEngine


def _fresh_engine():
    path = os.path.join(tempfile.gettempdir(), "_bma_state.json")
    if os.path.exists(path):
        os.remove(path)
    settings.LEARNING_STATE_PATH = path
    from core.continuous_engine import ContinuousLearningEngine
    return ContinuousLearningEngine(markov_engine=MarkovEngine())


def test_cold_start_anchored_by_physics_bayes():
    eng = _fresh_engine()
    w = eng.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # trainable models gated out at spin 0
    assert w["markov"] == 0.0 and w["lstm"] == 0.0
    # physics & bayes split evenly (equal log-evidence = 0)
    assert abs(w["physics"] - 0.5) < 1e-9 and abs(w["bayes"] - 0.5) < 1e-9
    print("OK cold start -> physics/bayes anchor (0.5/0.5), trainable gated")


def test_bma_rewards_better_model():
    eng = _fresh_engine()
    # Reality is heavily number 2 (physics says only 24%). Bayes adapts; physics
    # cannot -> bayes should accrue more predictive evidence -> more weight.
    stream = [2, 2, 2, 1, 2, 2, 5, 2, 2, 1] * 9
    hist = []
    for x in stream:
        eng.observe(x, list(hist))
        hist.append(x)
    w = eng.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["bayes"] > w["physics"], w
    assert eng.log_evidence["bayes"] > eng.log_evidence["physics"]
    print(f"OK BMA rewards bayes over physics (w_bayes={w['bayes']:.2f} > w_phys={w['physics']:.2f})")


def test_stacking_weights_valid():
    eng = _fresh_engine()
    stream = [2, 2, 1, 2, 5, 2, 2, 1, 2, 8] * 6
    hist = []
    for x in stream:
        eng.observe(x, list(hist))
        hist.append(x)
    stk = eng.stacking_weights()
    assert stk is not None
    assert abs(sum(stk.values()) - 1.0) < 1e-6
    assert stk["lstm"] == 0.0  # lstm never present -> excluded from stacking
    shown = {k: round(v, 2) for k, v in stk.items()}
    print("OK stacking weights valid:", shown)


def test_use_stacking_blend_runs():
    eng = _fresh_engine()
    eng.use_stacking = True
    stream = [2, 1, 2, 5, 2, 1, 2, 2, 1, 2] * 6
    hist = []
    for x in stream:
        eng.observe(x, list(hist))
        hist.append(x)
    w = eng.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6
    print("OK stacking blend mode produces valid weights")


def test_predict_and_status_intact():
    eng = _fresh_engine()
    for x in [1, 2, 5, 2, 1, 2]:
        eng.observe(x, [])
    preds = eng.predict_next([1, 2, 5])
    assert abs(sum(p["confidence"] for p in preds) - 1.0) < 1e-6
    st = eng.learning_status()
    assert st["method"] in ("BMA", "BMA+stacking")
    assert "log_evidence" in st
    print("OK predict_next + learning_status intact")


if __name__ == "__main__":
    test_cold_start_anchored_by_physics_bayes()
    test_bma_rewards_better_model()
    test_stacking_weights_valid()
    test_use_stacking_blend_runs()
    test_predict_and_status_intact()
    print("\nALL PROMPT 6 TESTS PASSED")
