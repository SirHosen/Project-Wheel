# -*- coding: utf-8 -*-
"""
core/wheel_math.py — Mathematical logic, probabilities, Expected Value (EV),
and statistical proofs for the Spin Wheel.
"""

import numpy as np

class WheelMath:
    """Core mathematical engine for the Spin Wheel."""
    
    def __init__(self, sequence: list, valid_numbers: list):
        self.sequence = sequence
        self.valid_numbers = valid_numbers
        self.n_segments = len(sequence)
        
        # Calculate true probabilities from the sequence
        self.counts = {num: sequence.count(num) for num in valid_numbers}
        self.true_probs = {num: count / self.n_segments for num, count in self.counts.items()}
        
    def calculate_ev(self, number: int) -> float:
        """
        Calculates the Expected Value (EV) per token bet for a specific number.
        Formula: (Probability * Payout) - (1 - Probability)
        Note: If number is 1, payout multiplier is 1 (net profit is 1x).
        """
        prob = self.true_probs.get(number, 0.0)
        payout_multiplier = 1 if number == 1 else number
        
        # EV = P(Win) * Payout - P(Loss) * Bet(1)
        ev = (prob * payout_multiplier) - (1 - prob)
        return ev
        
    def get_all_evs(self) -> dict:
        """Returns the expected value for all valid numbers."""
        return {num: self.calculate_ev(num) for num in self.valid_numbers}

    @staticmethod
    def wilson_interval(count: int, total: int, z: float = 1.96) -> tuple:
        """
        Wilson confidence interval for a proportion.
        Returns (proportion, lower_bound, upper_bound).
        """
        if total == 0:
            return 0.0, 0.0, 0.0
        p = count / total
        denom = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denom
        margin = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
        return p, max(0.0, center - margin), min(1.0, center + margin)

class GamblersFallacySimulator:
    """[EXPERIMENTAL / NOT WIRED] Proves why heuristic strategies fail over time.

    Educational simulator; not used anywhere in the live app flow (audit V3
    dead-code). Kept for experiments (see experimental/README.md). NOTE: WheelMath
    in this same module IS used for EV calculations and must stay.
    """
    
    @staticmethod
    def simulate_random_spins(wheel_math: WheelMath, rounds: int = 1000, seed=None) -> list:
        rng = np.random.default_rng(seed)
        probs = [wheel_math.true_probs[num] for num in wheel_math.valid_numbers]
        # Simulate index choices based on probability
        choices = rng.choice(wheel_math.valid_numbers, size=rounds, p=probs)
        return choices.tolist()
