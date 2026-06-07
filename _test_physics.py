# -*- coding: utf-8 -*-
"""Tests for core/physics_wheel.py (numpy path, runs anywhere)."""
import math
import numpy as np
from core.physics_wheel import WheelPhysics, TWO_PI


def test_inertia_and_torque():
    w = WheelPhysics(radius_m=0.8, mass_kg=25.0, decel_rad_s2=0.6)
    assert abs(w.inertia - 0.5 * 25.0 * 0.64) < 1e-9
    assert abs(w.friction_torque - w.inertia * 0.6) < 1e-9
    print(f"inertia={w.inertia:.3f} kg*m^2, friction_torque={w.friction_torque:.3f} N*m")


def test_spin_determinism_and_kinematics():
    w = WheelPhysics()
    r = w.spin(theta0=0.0, omega0=12.0)
    # total rotation = omega^2/(2 alpha) = 144/1.2 = 120 rad
    assert abs(r["total_rotation_rad"] - 120.0) < 1e-6
    assert abs(r["t_stop_s"] - 20.0) < 1e-6
    # same inputs -> same output (deterministic)
    r2 = w.spin(theta0=0.0, omega0=12.0)
    assert r["number"] == r2["number"] and r["segment_index"] == r2["segment_index"]
    print(f"spin: {r['revolutions']:.2f} rev, stops in {r['t_stop_s']:.1f}s, lands {r['number']}")


def test_vec_matches_scalar():
    w = WheelPhysics()
    rng = np.random.default_rng(0)
    th = rng.uniform(0, TWO_PI, 500)
    om = np.abs(rng.normal(12, 3, 500))
    vec = w.spin_vec(th, om)
    for i in range(0, 500, 53):
        assert int(vec[i]) == w.spin(th[i], om[i])["number"]
    print("vectorized spin_vec matches scalar spin on samples")


def test_monte_carlo_reproduces_area_fractions():
    """Core honesty check: with random initial conditions, the outcome
    distribution must converge to the wheel's AREA fractions (count/54).
    This proves the wheel just pays out by segment area — nothing more."""
    w = WheelPhysics()
    res = w.monte_carlo(n=400_000, omega_mean=12.0, omega_std=3.0, seed=42)
    theo = w.area_fractions()
    max_err = 0.0
    for num in w.valid_numbers:
        err = abs(res["freq"][num] - theo[num])
        max_err = max(max_err, err)
    print(f"Monte-Carlo vs area-fraction max abs error: {max_err:.4f}")
    assert max_err < 0.01, f"distribution should match area fractions, got err {max_err}"


def test_predictability_ceiling():
    """The precision needed to call a segment should be far tighter than any
    realistic video measurement — i.e. the wheel is NOT predictable from video."""
    w = WheelPhysics()
    rep = w.predictability_report(omega_mean=12.0, omega_std=3.0, video_fps=30)
    print(f"need {rep['relative_precision_needed_pct']:.3f}% precision on omega0; "
          f"video gives ~{rep['optimistic_video_precision_pct']:.2f}%; "
          f"-> {rep['segments_of_uncertainty_from_video']:.0f} segments of uncertainty")
    assert rep["predictable_from_video"] is False
    assert rep["relative_precision_needed_pct"] < 0.2  # sub-0.2% precision needed


if __name__ == "__main__":
    test_inertia_and_torque()
    test_spin_determinism_and_kinematics()
    test_vec_matches_scalar()
    test_monte_carlo_reproduces_area_fractions()
    test_predictability_ceiling()
    print("\nALL PHYSICS TESTS PASSED")
