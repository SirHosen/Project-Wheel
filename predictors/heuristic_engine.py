# -*- coding: utf-8 -*-
from collections import Counter
from .base import BasePredictor
from config import settings

class HeuristicEngine(BasePredictor):
    """
    Predictor based on 'Overdue' and 'Proximity' heuristics.
    Note: Statistically invalid for fair wheels (Gambler's Fallacy).
    Included for entertainment and comparison purposes.
    """
    
    def __init__(self):
        self.sequence = settings.SPINWHEEL_SEQUENCE
        self.valid_numbers = settings.VALID_NUMBERS
        
        # Calculate expected frequency
        wheel_counts = Counter(self.sequence)
        self.expected_freq = {num: wheel_counts[num] / len(self.sequence) 
                              for num in self.valid_numbers}
                              
    def _normalize(self, d: dict) -> dict:
        total = sum(d.values())
        if total == 0:
            return {k: 1.0/len(d) for k in d.keys()}
        return {k: v / total for k, v in d.items()}
        
    def _find_positions(self, target: int) -> list:
        return [i for i, num in enumerate(self.sequence) if num == target]

    def predict_next(self, history: list) -> list:
        if not history:
            prob = 1.0 / len(self.valid_numbers)
            return [{"number": n, "confidence": prob} for n in self.valid_numbers]
            
        # 1. Overdue Analysis (Gambler's Fallacy)
        last_seen = {num: len(history) for num in self.valid_numbers}
        for i, num in enumerate(reversed(history)):
            if last_seen[num] == len(history):
                last_seen[num] = i
                
        skor_overdue = {num: last_seen[num] * self.expected_freq[num] 
                       for num in self.valid_numbers}
        skor_overdue_norm = self._normalize(skor_overdue)
        
        # 2. Proximity Analysis
        angka_terakhir = history[-1]
        posisi_terakhir = self._find_positions(angka_terakhir)
        skor_posisi = {num: 0.0 for num in self.valid_numbers}
        
        for pos in posisi_terakhir:
            for offset in range(1, 6):
                idx = (pos + offset) % len(self.sequence)
                num_kandidat = self.sequence[idx]
                skor_posisi[num_kandidat] += (6 - offset)
                
        skor_posisi_norm = self._normalize(skor_posisi)
        
        # 3. Combine scores
        skor_final = {}
        for num in self.valid_numbers:
            skor_final[num] = 0.5 * skor_overdue_norm.get(num, 0) + 0.5 * skor_posisi_norm.get(num, 0)
            
        skor_final_norm = self._normalize(skor_final)
        
        predictions = [{"number": num, "confidence": conf} 
                      for num, conf in skor_final_norm.items()]
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return predictions
