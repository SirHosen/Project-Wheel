# -*- coding: utf-8 -*-
"""
config/settings.py — Application configuration, constants, and UI themes.
"""

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
