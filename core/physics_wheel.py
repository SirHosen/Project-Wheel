# -*- coding: utf-8 -*-
"""
core/physics_wheel.py — Rigid-body rotational physics of the Spin Wheel.

WHAT THIS IS (honest scope):
  A real rotational-dynamics model of a vertical money-wheel that is spun by
  hand and slows down under friction until it stops. It lets us:
    1. Simulate a single spin deterministically from initial conditions
       (release angle theta0, release angular velocity omega0).
    2. Run massive Monte-Carlo batches (CPU/numpy here, GPU/TensorFlow in
       physics_lab.py) over realistic initial-condition distributions.
    3. Quantify the *predictability ceiling*: how precisely you would have to
       MEASURE omega0 / theta0 to call the landing segment in advance.

WHAT THIS IS NOT:
  It is NOT a way to predict live spins from past results. The landing segment
  is a deterministic function of the *initial conditions of that spin*, which
  the spinner sets anew (and essentially at random) every time. There is no
  causal link from the fixed number sequence to the spin force, so nothing in
  the result history can forecast the next outcome. See predictability_report().

Physics model
-------------
Uniform disk, radius R, mass m:  I = 1/2 * m * R^2
Friction is modelled as a constant decelerating torque tau_f, giving constant
angular deceleration alpha = tau_f / I (Coulomb-dominated). An optional viscous
term (proportional to omega) can be added for realism.

Constant-deceleration kinematics:
  omega(t) = omega0 - alpha * t
  stops at  t_stop = omega0 / alpha
  total rotation  Theta = omega0^2 / (2 * alpha)
  final angle  theta_f = (theta0 + Theta) mod 2*pi
  segment = floor(theta_f / seg_angle),  seg_angle = 2*pi / n_segments
"""

import math
import numpy as np

try:
    from config import settings
    _DEFAULT_SEQUENCE = settings.SPINWHEEL_SEQUENCE
    _DEFAULT_VALID = settings.VALID_NUMBERS
except Exception:  # allow standalone use / testing without the package context
    _DEFAULT_SEQUENCE = [1, 5, 2, 10, 1, 2, 20, 1, 8, 2, 1, 5, 1, 10, 2, 1, 5, 2,
                         1, 40, 2, 1, 8, 1, 5, 1, 15, 2, 1, 10, 1, 5, 1, 20, 2, 1,
                         8, 2, 1, 2, 10, 1, 2, 5, 1, 2, 30, 1, 8, 1, 5, 1, 2, 15]
    _DEFAULT_VALID = [1, 2, 5, 8, 10, 15, 20, 30, 40]

TWO_PI = 2.0 * math.pi


class WheelPhysics:
    """Rotational-dynamics model of the physical spin wheel."""

    def __init__(self,
                 radius_m: float = 0.80,      # ~1.6 m diameter ("adult-woman-sized")
                 mass_kg: float = 25.0,        # large wooden/acrylic wheel estimate
                 decel_rad_s2: float = 0.60,   # constant angular deceleration (Coulomb)
                 viscous_coeff: float = 0.0,   # optional viscous drag (per-omega), 0 = off
                 sequence: list = None,
                 valid_numbers: list = None):
        self.radius = float(radius_m)
        self.mass = float(mass_kg)
        self.decel = float(decel_rad_s2)
        self.viscous = float(viscous_coeff)
        self.sequence = list(sequence if sequence is not None else _DEFAULT_SEQUENCE)
        self.valid_numbers = list(valid_numbers if valid_numbers is not None else _DEFAULT_VALID)
        self.n_segments = len(self.sequence)
        self.seg_angle = TWO_PI / self.n_segments
        self._seg_arr = np.array(self.sequence, dtype=np.int64)

    # ---- derived physical quantities -------------------------------------
    @property
    def inertia(self) -> float:
        """Moment of inertia of a uniform disk: I = 1/2 m R^2 (kg*m^2)."""
        return 0.5 * self.mass * self.radius ** 2

    @property
    def friction_torque(self) -> float:
        """Constant friction torque implied by the deceleration: tau = I*alpha (N*m)."""
        return self.inertia * self.decel

    def segment_for_angle(self, angle: float) -> int:
        idx = int(math.floor((angle % TWO_PI) / self.seg_angle)) % self.n_segments
        return idx

    # ---- single deterministic spin ---------------------------------------
    def spin(self, theta0: float, omega0: float) -> dict:
        """Simulate one spin from release angle theta0 and angular velocity omega0."""
        if omega0 <= 0:
            total = 0.0
            t_stop = 0.0
        elif self.viscous > 0:
            # viscous + Coulomb: integrate omega until it hits zero (closed form
            # is messy with both terms, so use a stable analytic Coulomb base and
            # treat viscous as an effective extra deceleration at mean omega).
            eff_alpha = self.decel + self.viscous * (omega0 / 2.0)
            t_stop = omega0 / eff_alpha
            total = omega0 ** 2 / (2.0 * eff_alpha)
        else:
            t_stop = omega0 / self.decel
            total = omega0 ** 2 / (2.0 * self.decel)
        final_angle = (theta0 + total) % TWO_PI
        idx = self.segment_for_angle(final_angle)
        return {
            "theta0": theta0,
            "omega0": omega0,
            "total_rotation_rad": total,
            "revolutions": total / TWO_PI,
            "t_stop_s": t_stop,
            "final_angle_rad": final_angle,
            "segment_index": idx,
            "number": int(self._seg_arr[idx]),
        }

    # ---- vectorized spins (numpy) ----------------------------------------
    def spin_vec(self, theta0: np.ndarray, omega0: np.ndarray) -> np.ndarray:
        """Vectorized landing-number for arrays of initial conditions (numpy)."""
        theta0 = np.asarray(theta0, dtype=np.float64)
        omega0 = np.asarray(omega0, dtype=np.float64)
        omega0 = np.clip(omega0, 0.0, None)
        if self.viscous > 0:
            eff_alpha = self.decel + self.viscous * (omega0 / 2.0)
        else:
            eff_alpha = self.decel
        total = omega0 ** 2 / (2.0 * eff_alpha)
        final_angle = np.mod(theta0 + total, TWO_PI)
        idx = np.floor(final_angle / self.seg_angle).astype(np.int64) % self.n_segments
        return self._seg_arr[idx]

    # ---- Monte-Carlo over realistic spins (numpy) ------------------------
    def monte_carlo(self, n: int = 1_000_000,
                    omega_mean: float = 12.0, omega_std: float = 3.0,
                    seed: int = None) -> dict:
        """Simulate n random spins; return outcome distribution over numbers.

        theta0 ~ Uniform(0, 2pi)  (release angle unknown / uncontrolled)
        omega0 ~ Normal(omega_mean, omega_std) truncated > 0  (human spin force)
        """
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(0.0, TWO_PI, size=n)
        omega0 = np.abs(rng.normal(omega_mean, omega_std, size=n))
        numbers = self.spin_vec(theta0, omega0)
        unique, counts = np.unique(numbers, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        freq = {int(v): dist.get(int(v), 0) / n for v in self.valid_numbers}
        return {"n": n, "counts": dist, "freq": freq}

    def area_fractions(self) -> dict:
        """Theoretical landing probability = fraction of wheel area per number."""
        return {int(v): self.sequence.count(v) / self.n_segments
                for v in self.valid_numbers}

    # ---- predictability ceiling ------------------------------------------
    def sensitivity(self, omega0: float) -> dict:
        """How sensitive the landing segment is to a tiny change in omega0.

        dTheta/domega0 = omega0 / alpha  (rad of extra rotation per rad/s).
        One segment spans seg_angle rad, so the omega0 change that shifts the
        result by exactly one segment is:  d_omega = seg_angle * alpha / omega0.
        """
        alpha = self.decel
        dTheta_domega = omega0 / alpha
        d_omega_per_seg = self.seg_angle / dTheta_domega
        return {
            "omega0": omega0,
            "dTheta_domega_rad_per_rad_s": dTheta_domega,
            "d_omega_per_segment_rad_s": d_omega_per_seg,
            "relative_precision_needed": d_omega_per_seg / omega0,
        }

    def predictability_report(self, omega_mean: float = 12.0, omega_std: float = 3.0,
                              video_fps: int = 30) -> dict:
        """Compare the precision needed to predict vs. what video measurement gives."""
        s = self.sensitivity(omega_mean)
        need = s["relative_precision_needed"]
        # Realistic omega0 measurement error from video: across one frame the
        # wheel turns omega0/fps rad; estimating omega0 by finite differences on
        # a compressed stream realistically carries a few-percent error. We use a
        # conservative ~2% floor as an optimistic best case.
        video_rel_err = max(0.02, (self.seg_angle * video_fps) / (omega_mean))
        # Natural spread of omega0 across spins relative to its mean:
        natural_rel_spread = omega_std / omega_mean
        return {
            "relative_precision_needed_pct": 100 * need,
            "optimistic_video_precision_pct": 100 * video_rel_err,
            "natural_spin_spread_pct": 100 * natural_rel_spread,
            "predictable_from_video": video_rel_err <= need,
            "segments_of_uncertainty_from_video": video_rel_err / need,
        }


def default_wheel() -> "WheelPhysics":
    return WheelPhysics()
