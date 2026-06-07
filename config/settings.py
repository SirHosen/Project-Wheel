# -*- coding: utf-8 -*-
"""
config/settings.py — Application configuration, constants, and UI themes.
"""

# Bumped whenever the prediction/betting/reporting logic changes. Stamped onto
# every exported audit report so upgrades and audits are traceable.
APP_VERSION = "1.6.1"

# ---------------------------------------------------------
# Wheel Configuration
# ---------------------------------------------------------

# The sequence of numbers on the spin wheel physically
SPINWHEEL_SEQUENCE = [
    1, 5, 2, 10, 1, 2, 20, 1, 8, 2, 1, 5, 1, 10, 2, 1, 5, 2, 1, 40,
    2, 1, 8, 1, 5, 1, 15, 2, 1, 10, 1, 5, 1, 20, 2, 1, 8, 2, 1, 2,
    10, 1, 2, 5, 1, 2, 30, 1, 8, 1, 5, 1, 2, 15
]

# Valid numbers that can be bet on
VALID_NUMBERS = [1, 2, 5, 8, 10, 15, 20, 30, 40]

def calculate_reward(token_bet: int, winning_number: int) -> int:
    """Calculates the total payout including the original bet."""
    if winning_number == 1:
        return (token_bet * 1) + token_bet
    return (token_bet * winning_number) + token_bet

# ---------------------------------------------------------
# UI Theme Constants
# ---------------------------------------------------------

UI_COLORS = {
    "background": "#0D0D0D",      # PRIMARY_BG
    "panel": "#1A1A2E",           # SECONDARY_BG
    "card": "#16213E",            # CARD_BG
    "text": "#FFFFFF",            # TEXT_PRIMARY
    "text_secondary": "#8892B0",  # TEXT_SECONDARY
    "primary": "#39FF14",         # ACCENT_GREEN
    "secondary": "#FFD700",       # ACCENT_GOLD
    "error": "#FF4444",           # ACCENT_RED
    "info": "#00D4FF",            # ACCENT_BLUE
}

FONTS = {
    "header": ("Inter", 24, "bold"),
    "subheader": ("Inter", 16, "bold"),
    "body": ("Inter", 14),
    "small": ("Inter", 12),
}

# ---------------------------------------------------------
# Predictor Settings
# ---------------------------------------------------------

# Default allocation risk per spin (percentage of total capital)
DEFAULT_RISK_PCT = 0.30 

# TensorFlow settings
LSTM_SEQUENCE_LENGTH = 5
TF_EPOCHS_INCREMENTAL = 1
TF_EPOCHS_BULK = 10

# ---- Deep TF-LSTM model (tuned to exploit a real GPU, e.g. RTX 3080) ----
LSTM_EMBEDDING_DIM = 16            # size of the learned per-number embedding
LSTM_UNITS = [128, 64]             # stacked LSTM layer sizes (deeper on GPU)
LSTM_DENSE_UNITS = 64              # hidden dense layer before the softmax
LSTM_DROPOUT = 0.2                 # regularization
LSTM_BATCH_SIZE = 64               # GPU-friendly batch size
LSTM_BULK_EPOCHS = 60              # heavy one-time training (fast on a 3080)
LSTM_VALIDATION_SPLIT = 0.15       # held-out split for honest validation
LSTM_EARLY_STOP_PATIENCE = 8       # stop when val accuracy plateaus
LSTM_USE_MIXED_PRECISION = True    # float16 Tensor Cores (Ampere/RTX 3080)
LSTM_MODEL_PATH = "models/lstm_spinwheel.keras"  # saved/reused trained model

# ---------------------------------------------------------
# Physics model (core/physics_wheel.py, physics_lab.py)
# Rigid-body rotational dynamics of the physical wheel. Used to SIMULATE and
# EXPLAIN the wheel (GPU Monte-Carlo), not to predict live spins from history.
# ---------------------------------------------------------
WHEEL_RADIUS_M = 0.80        # ~1.6 m diameter ("adult-woman-sized", per video)
WHEEL_MASS_KG = 25.0         # estimated mass of the wheel disk
WHEEL_DECEL_RAD_S2 = 0.60    # constant angular deceleration from friction
WHEEL_SPIN_OMEGA_MEAN = 12.0 # typical release angular velocity (rad/s)
WHEEL_SPIN_OMEGA_STD = 3.0   # human spin-to-spin variability (rad/s)

# ---------------------------------------------------------
# Continuous-learning ensemble (core/continuous_engine.py)
# Persisted learning state so the brain keeps improving across launches.
# ---------------------------------------------------------
LEARNING_STATE_PATH = "models/continuous_state.json"
ENSEMBLE_EMA_LR = 0.08       # how fast model trust adapts per spin
ENSEMBLE_TEMPERATURE = 0.15  # softmax sharpness for blend weights
ENSEMBLE_WARMUP_SPINS = 30   # spins before markov/lstm earn full trust weight
