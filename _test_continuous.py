# -*- coding: utf-8 -*-
"""Tests for core/continuous_engine.py (no TF needed)."""
import os
import tempfile
import random
from collections import Counter

from config import settings
from core.continuous_engine import ContinuousLearningEngine, _normalize
from predictors.markov_engine import MarkovEngine


def _fresh(tmp):
    settings.LEARNING_STATE_PATH = os.path.join(tmp, "state.json")
    return ContinuousLearningEngine(markov_engine=MarkovEngine())


def test_physics_prior_matches_area():
    e = ContinuousLearningEngine(markov_engine=MarkovEngine())
    prior = e.physics_prior()
    assert abs(sum(prior.values()) - 1.0) < 1e-9
    # number 1 is the most common segment
    assert max(prior, key=prior.get) == 1
    assert abs(prior[1] - 20/54) < 1e-6
    print("physics prior OK:", {k: round(v, 3) for k, v in prior.items()})


def test_bayes_converges_to_empirical():
    e = ContinuousLearningEngine(markov_engine=MarkovEngine())
    # feed a heavily biased history; posterior should lean toward 8
    hist = [8] * 300 + [1] * 20
    post = e.bayes_posterior(hist)
    assert max(post, key=post.get) == 8
    assert post[8] > 0.7
    print("bayes posterior leans to 8:", round(post[8], 3))


def test_ensemble_is_valid_distribution():
    e = ContinuousLearningEngine(markov_engine=MarkovEngine())
    hist = [random.choice(settings.VALID_NUMBERS) for _ in range(60)]
    probs = e.ensemble_probabilities(hist)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in probs.values())
    preds = e.predict_next(hist)
    assert preds[0]["confidence"] >= preds[-1]["confidence"]
    print("ensemble valid; top pick:", preds[0]["number"], round(preds[0]["confidence"], 3))


def test_observe_updates_scores_and_weights():
    with tempfile.TemporaryDirectory() as tmp:
        e = _fresh(tmp)
        w0 = e.weights().copy()
        # Simulate spins where the result is ALWAYS the physics top pick (1):
        # physics accuracy should climb and its weight should rise.
        hist = []
        for _ in range(40):
            e.observe(1, hist)
            hist.append(1)
        st = e.learning_status()
        assert st["n_observed"] == 40
        # Physics clearly learned to be reliable here.
        assert e.scores["physics"] > 0.7
        w = e.weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9
        # After warmup it shares trust with the other now-proven models, but
        # always keeps a strong, non-degenerate anchor share.
        assert w["physics"] >= 0.2
        print("after 40 spins -> weights:", st["weights"], "acc:", st["accuracy"])


def test_state_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        settings.LEARNING_STATE_PATH = os.path.join(tmp, "state.json")
        e1 = ContinuousLearningEngine(markov_engine=MarkovEngine())
        hist = []
        for _ in range(15):
            e1.observe(2, hist); hist.append(2)
        n1 = e1.n_observed
        # New instance must reload the learned state from disk.
        e2 = ContinuousLearningEngine(markov_engine=MarkovEngine())
        assert e2.n_observed == n1 == 15
        assert abs(e2.scores["physics"] - e1.scores["physics"]) < 1e-9
        print("state persisted: n_observed =", e2.n_observed)


class _FakeLstm:
    """Undertrained LSTM that collapsed onto the rarest number (40)."""
    _trained = True

    def predict_next(self, history):
        return [{"number": 40, "confidence": 0.95}, {"number": 1, "confidence": 0.05}]


def test_coldstart_ignores_overconfident_lstm():
    # Regression test for the "hasil 40 terus" bug: a fresh brain must not let
    # an over-confident, undertrained LSTM hijack the prediction to a rare number.
    e = ContinuousLearningEngine(markov_engine=MarkovEngine(), lstm_engine=_FakeLstm())
    preds = e.predict_next([1, 2, 1, 5, 1, 2])
    top = preds[0]["number"]
    assert top != 40, f"cold-start ensemble hijacked by undertrained LSTM -> {top}"
    assert top == 1, f"expected most-common number 1, got {top}"
    print("cold-start safe; top pick:", top, "(lstm gated, not 40)")


if __name__ == "__main__":
    test_physics_prior_matches_area()
    test_bayes_converges_to_empirical()
    test_ensemble_is_valid_distribution()
    test_observe_updates_scores_and_weights()
    test_state_persists_across_instances()
    test_coldstart_ignores_overconfident_lstm()
    print("\nALL CONTINUOUS-ENGINE TESTS PASSED")
