# -*- coding: utf-8 -*-
"""Central configuration: wheel layout, payouts, and every tunable knob."""
import os

APP_VERSION = "2.3.1"

# --- Wheel layout ---------------------------------------------------------
# The 54 physical segments, in wheel order.
WHEEL_SEQUENCE = [1, 5, 2, 10, 1, 2, 20, 1, 8, 2, 1, 5, 1, 10, 2, 1, 5, 2, 1,
                  40, 2, 1, 8, 1, 5, 1, 15, 2, 1, 10, 1, 5, 1, 20, 2, 1, 8, 2,
                  1, 2, 10, 1, 2, 5, 1, 2, 30, 1, 8, 1, 5, 1, 2, 15]
# Distinct payable numbers.
VALID_NUMBERS = [1, 2, 5, 8, 10, 15, 20, 30, 40]


def payout_multiplier(n):
    """A bet on number `n` pays `n` to 1, except 1 which pays 1 to 1."""
    return 1 if n == 1 else int(n)


# --- Physics (rough 'is it predictable?' sanity check) --------------------
RADIUS_M = 0.80       # wheel radius (m)
MASS_KG = 25.0        # wheel mass (kg)
DECEL = 0.60          # angular deceleration (rad/s^2)
OMEGA_MEAN = 12.0     # typical launch angular speed (rad/s)
OMEGA_STD = 3.0       # spin-to-spin variation in launch speed (rad/s)

# --- Bayesian bias tracker ------------------------------------------------
PRIOR_STRENGTH = len(WHEEL_SEQUENCE)   # 54 pseudo-spins of "the wheel is fair"
CI_Z = 1.96                            # 95% confidence band
EV_MARGIN = 0.05                       # require EV_lo > this to call it an edge
MIN_OBS = 25                           # never claim bias/edge below this many spins
# Multiple-testing correction for the per-number edge search. We test all 9
# numbers at once, so an uncorrected 5% test would false-positive far too often.
# The family-wide alpha is split across numbers (Sidak by default; Bonferroni
# also available; "none" disables the correction).
EDGE_FAMILY_ALPHA = 0.05
MULTIPLE_TEST_CORRECTION = "sidak"     # "sidak" | "bonferroni" | "none"

# --- LSTM (AI training playground) ----------------------------------------
SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 16
LSTM_UNITS = 64
DROPOUT = 0.2
EPOCHS = 60
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.15
EARLY_STOP_PATIENCE = 8

# --- Runtime paths --------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(ROOT_DIR, "runtime")
MODEL_PATH = os.path.join(RUNTIME_DIR, "lstm_spinwheel.pt")
OBSERVATIONS_PATH = os.path.join(RUNTIME_DIR, "observations.csv")

# --- Screen capture -------------------------------------------------------
CAPTURE_FPS = 15
RESULT_MARGIN = 26.0
STABLE_FRAMES = 3

# --- Result reader calibration -------------------------------------------
# Positions are FRACTIONS of the captured frame (0..1), so they adapt to any
# resolution. If detection misreads numbers, run:
#     python scripts\auto_watch.py --snapshot --monitor 1
# open runtime/calibration.png, and nudge these until each green box sits on
# the matching number cell in the game's result row.
RESULT_LANDSCAPE = {
    "fy": 0.8573,        # vertical center of the result row
    "fx_start": 0.2730,  # x-center of the first number (1)
    "fx_end": 0.7416,    # x-center of the last number (40)
    "bw": 0.0145,        # sample box width
    "bh": 0.0285,        # sample box height
}
RESULT_PORTRAIT = {
    "rows": [0.5708, 0.6750, 0.7833],
    "cols": [0.4010, 0.5000, 0.5990],
    "bw": 0.0271,
    "bh": 0.0271,
}
# A detection must beat the 2nd-brightest cell by at least this much (guards
# against a glow bleeding into a neighbouring number and causing a misread).
RESULT_MIN_SEPARATION = 8.0

# --- UI (tiny live panel) -------------------------------------------------
UI_COLORS = {
    "background": "#1e1e2e",
    "text": "#cdd6f4",
    "text_secondary": "#9399b2",
    "primary": "#89b4fa",
    "gpu": "#a6e3a1",      # green  = AI running on GPU
    "cpu": "#f38ba8",      # red    = AI running on CPU (or no TensorFlow)
    "button": "#313244",   # button background
    "surface": "#313244",  # separators / bar troughs
}
