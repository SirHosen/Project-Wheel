# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class BasePredictor(ABC):
    """Abstract base class for all prediction engines."""
    
    @abstractmethod
    def predict_next(self, history: list) -> list:
        """
        Takes a list of historical numbers and returns a list of dictionaries:
        [ {"number": int, "confidence": float}, ... ]
        Sorted by highest confidence first.
        """
        pass
        
    def get_token_allocation(self, predictions: list, capital: int, risk_pct: float) -> list:
        """
        Distributes tokens based on confidence.
        Base implementation handles basic proportional distribution.
        """
        if not predictions:
            return []
            
        budget = capital * risk_pct
        top_preds = predictions[:3]
        total_confidence = sum(p["confidence"] for p in top_preds)
        
        allocations = []
        for p in top_preds:
            if total_confidence > 0:
                amount = int(budget * (p["confidence"] / total_confidence))
            else:
                amount = 0
                
            allocations.append({
                "number": p["number"],
                "confidence": p["confidence"],
                "token_bet": max(1, amount) if budget > 0 else 0
            })
            
        return allocations
