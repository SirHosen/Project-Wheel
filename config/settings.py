# -*- coding: utf-8 -*-
"""
config/settings.py — Application configuration, constants, and UI themes.
"""

# Bumped whenever the prediction/betting/reporting logic changes. Stamped onto
# every exported audit report so upgrades and audits are traceable.
APP_VERSION = "1.30.0"

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
LSTM_SEQUENCE_LENGTH = 10            # longer context window (was 5; PROMPT 11)
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
# ---- PROMPT 11: attention + feature engineering + augmentation ----
LSTM_USE_ATTENTION = True          # self-attention (MultiHeadAttention) between LSTMs
LSTM_ATTENTION_HEADS = 4           # number of attention heads
LSTM_ATTENTION_DIM = 32            # per-head key/query dimension
LSTM_USE_FEATURES = True           # feed engineered features beside the embedding
LSTM_AUGMENT = True                # drop-random-prefix augmentation during bulk train
LSTM_AUGMENT_PROB = 0.5            # fraction of windows augmented
LSTM_AUGMENT_MAX_FRAC = 0.5        # max prefix fraction dropped

# ---- PROMPT 12: Variance Harvest mode (OPT-IN, default OFF) ----
# HONEST: every number is long-run -EV; this mode buys tiny lottery tickets on
# high-multiplier numbers for high-variance upside. Expect slow bleed long-run.
HARVEST_MODE_DEFAULT = False              # OFF unless the user explicitly enables it
HARVEST_MIN_CONFIDENCE = 0.05             # 5% chance threshold (bypasses EV gate)
HARVEST_TARGET_MULTIPLIERS = [5, 8, 10, 15, 20, 30, 40]  # skip #1/#2 (payout too small)
HARVEST_TOKEN_PCT = 0.02                  # fixed tiny stake = 2% of capital per pick
HARVEST_MAX_PICKS = 2                     # at most 2 lottery tickets per round
HARVEST_SKIP_RATE = 0.20                  # randomly skip 20% of rounds to save capital

# --- PROMPT 14: manual % as a Dirichlet prior --------------------------------
# The manual distribution is worth MANUAL_PRIOR_STRENGTH pseudo-observations.
# Blend weight vs the engine = strength / (strength + data_count), so a hunch
# fades automatically as real spins accumulate. Default 27 reproduces the old
# 0.35 weight at ~50 spins, but now the weight MOVES with the data.
MANUAL_PRIOR_STRENGTH_DEFAULT = 27
MANUAL_PRIOR_STRENGTH_MIN = 5
MANUAL_PRIOR_STRENGTH_MAX = 100

# --- PROMPT 16: Anti-tilt guard (TiltDetector) ---
# Detects emotional loss-chasing: N losing bet-rounds inside a short time
# window WITH escalating stakes -> enforce a mandatory cooldown before the next
# bet. Protects bankroll/discipline; it does NOT make a fair wheel beatable.
TILT_DETECT_ENABLED = True
TILT_N_LOSSES = 3            # losing rounds in a row to trip the guard
TILT_WINDOW_MINUTES = 5.0    # they must happen within this many minutes
TILT_COOLDOWN_SECONDS = 60   # enforced breather after a trip
TILT_REQUIRE_RISING = True   # only trip when stakes are escalating
TILT_STRICT_RISING = True    # True = every step must strictly increase

# --- PROMPT 17: Multi-engine consensus filter ---
# Only keep a stake on a number when at least CONSENSUS_MIN_AGREE of the
# independent models (physics / bayes / markov / lstm) rank it in their top-N.
# A risk filter against single-model overconfidence -- NOT an edge generator.
# Fails OPEN: if fewer than MIN_AGREE engines are available, nothing is blocked.
CONSENSUS_FILTER_ENABLED = True
CONSENSUS_MIN_AGREE = 2       # how many engines must agree to allow a bet
CONSENSUS_TOP_N = 3           # each engine votes for its top-N numbers
CONSENSUS_MIN_PROB = 0.0      # ignore a vote below this probability

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

# ---------------------------------------------------------
# Bayesian-Optimal engine (predictors/bayesian_optimal.py)
# The provably-optimal predictor for an i.i.d. wheel: a Dirichlet-Multinomial
# posterior + statistically-gated EV/edge detector. Only stakes when a number's
# LOWER credible bound is robustly +EV (real bias), else recommends SKIP.
# ---------------------------------------------------------
BAYES_OPT_PRIOR_STRENGTH = len(SPINWHEEL_SEQUENCE)  # 54 pseudo-obs from layout
BAYES_OPT_CI_Z = 1.96        # credible-interval width (~95%)
BAYES_OPT_EV_MARGIN = 0.05   # conservative EV must clear this to bet
BAYES_OPT_MIN_OBS = 25       # min real spins before any +EV bet is allowed

# --- Ensemble BMA (PROMPT 6, v1.16.0) ---
# Bobot ensemble = Bayesian Model Averaging atas log-evidence prediktif terdiskon.
ENSEMBLE_BMA_DISCOUNT = 0.98      # forgetting factor (1.0 = ingat semua, <1 = adaptif)
ENSEMBLE_BMA_TEMPERATURE = 1.0    # >1 melembutkan, <1 menajamkan kolaps ke model terbaik
ENSEMBLE_USE_STACKING = False     # True -> blend 50/50 BMA + stacking EM (opsional)

# PROMPT 9: session detection / temporal-drift.
SESSION_GAP_MINUTES = 30        # idle gap that starts a new session
SESSION_DRIFT_ALPHA = 0.05      # KS 2-sample significance for drift verdict
RECENCY_HALF_LIFE = 50          # spins per halving for optional recency weighting
