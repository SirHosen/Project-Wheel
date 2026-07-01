# -*- coding: utf-8 -*-
"""A back-of-envelope predictability check.

A heavy wheel started by hand has tiny spin-to-spin differences in launch speed,
and constant deceleration amplifies those differences into a large uncertainty
about where it stops. This module quantifies that to make one honest point: you
cannot predict the next result.
"""
import math

from config import DECEL, OMEGA_MEAN, OMEGA_STD, WHEEL_SEQUENCE


def predictability_report(omega_mean=OMEGA_MEAN, omega_std=OMEGA_STD,
                          decel=DECEL, n_segments=None):
    """Return how many segments of uncertainty the launch-speed noise creates.

    Stop angle from launch speed w under constant decel: theta = w^2 / (2*decel).
    Sensitivity of the stop angle to launch speed: d(theta)/dw = w / decel.
    So a launch-speed spread of omega_std maps to (w/decel)*omega_std radians of
    stop-angle spread -- which we express in wheel turns and segments.
    """
    n_segments = n_segments or len(WHEEL_SEQUENCE)
    sensitivity = omega_mean / decel                 # rad per (rad/s)
    theta_spread = sensitivity * omega_std           # rad of uncertainty
    turns = theta_spread / (2.0 * math.pi)
    segments_spread = turns * n_segments
    predictable = segments_spread < 1.0
    if predictable:
        verdict = (f"borderline: launch-speed noise spans only "
                   f"{segments_spread:.2f} segments")
    else:
        verdict = (f"NOT predictable (launch-speed noise spans "
                   f"{segments_spread:.1f} segments / {turns:.1f} turns)")
    return {"sensitivity_rad_per_rad_s": sensitivity,
            "theta_spread_rad": theta_spread, "turns": turns,
            "segments_spread": segments_spread, "predictable": predictable,
            "verdict": verdict}
