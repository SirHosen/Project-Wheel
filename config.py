# -*- coding: utf-8 -*-
"""Central configuration: wheel layout, payouts, and every tunable knob."""
import os

APP_VERSION = "2.0.0"

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
MODEL_PATH = os.path.join(RUNTIME_DIR, "lstm_spinwheel.keras")
OBSERVATIONS_PATH = os.path.join(RUNTIME_DIR, "observations.csv")

# --- Screen capture -------------------------------------------------------
CAPTURE_FPS = 15
RESULT_MARGIN = 26.0
STABLE_FRAMES = 3

# --- UI (tiny live panel) -------------------------------------------------
UI_COLORS = {
    "background": "#1e1e2e",
    "text": "#cdd6f4",
    "text_secondary": "#9399b2",
    "primary": "#89b4fa",
}
