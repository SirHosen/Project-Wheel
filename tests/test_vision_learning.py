# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Tests untuk vision learning loop (kamera -> evidence -> prior)."""
import json
import os
import tempfile

from config import settings
from core.wheel_bias import chi_square_gof, design_distribution, gammq
from vision.observation_log import load_observations, log_observation
from predictors.bayesian_optimal import BayesianOptimalEngine


def test_gammq_chi_square_survival():
    # chi2 = 3.841 pada df=1 -> p ~ 0.05 (nilai kritis 95% klasik)
    p = gammq(0.5, 3.841 / 2.0)
    assert abs(p - 0.05) < 0.005, p
    # Q(a, 0) == 1
    assert gammq(2.0, 0.0) == 1.0


def test_chi_square_fair_vs_biased():
    valid = settings.VALID_NUMBERS
    design = design_distribution(settings.SPINWHEEL_SEQUENCE, valid)
    n = 540
    expected = {k: n * design[k] for k in valid}
    # observed = desain-scaled -> chi2 ~ 0, p tinggi
    fair_obs = {k: int(round(expected[k])) for k in valid}
    _, _, p_fair = chi_square_gof(fair_obs, expected)
    assert p_fair > 0.5, p_fair
    # bias kuat: semua jatuh di angka 40 (porsi desain ~1.85%)
    biased_obs = {k: 0 for k in valid}
    biased_obs[40] = n
    _, _, p_b = chi_square_gof(biased_obs, expected)
    assert p_b < 0.001, p_b


def test_observation_log_roundtrip():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    os.remove(path)
    try:
        log_observation({"number": 5, "index": 12, "angle": 100.5, "confidence": 0.8, "stopped": True}, path=path)
        log_observation({"number": 1, "index": 0, "angle": 5.0, "confidence": 0.9, "stopped": False}, path=path)
        rows = load_observations(path=path)
        assert len(rows) == 2
        assert rows[0]["number"] == 5 and rows[0]["stopped"] is True
        assert rows[1]["number"] == 1 and rows[1]["stopped"] is False
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_engine_folds_observed_counts():
    base = BayesianOptimalEngine(wheel_prior_path="/nonexistent/wheel_prior.json")
    base_preds = {p["number"]: p["confidence"] for p in base.predict_next([])}

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"counts": {"40": 200}, "n_obs": 200}, f)
        learned = BayesianOptimalEngine(wheel_prior_path=path)
        learned_preds = {p["number"]: p["confidence"] for p in learned.predict_next([])}
        # mengamati angka 40 banyak kali harus menaikkan posterior mean-nya signifikan
        assert learned_preds[40] > base_preds[40] + 0.2, (base_preds[40], learned_preds[40])
        # support harus mencerminkan spin observasi tambahan
        assert learned.predict_next([])[0]["support"] >= 200
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_engine_graceful_without_file():
    eng = BayesianOptimalEngine(wheel_prior_path="/definitely/not/here.json")
    assert sum(eng.observed_counts.values()) == 0
    preds = eng.predict_next([])
    assert preds and preds[0]["support"] == 0


def main():
    test_gammq_chi_square_survival()
    test_chi_square_fair_vs_biased()
    test_observation_log_roundtrip()
    test_engine_folds_observed_counts()
    test_engine_graceful_without_file()
    print("ALL VISION LEARNING TESTS PASSED")


if __name__ == "__main__":
    main()
