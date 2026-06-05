# -*- coding: utf-8 -*-
import threading
from typing import Callable, Optional

from config import settings
from data.tracker import Tracker
from predictors.tf_lstm_engine import TfLstmEngine
from predictors.heuristic_engine import HeuristicEngine
from core.wheel_math import WheelMath

class MainViewModel:
    """
    ViewModel coordinating the UI state and the business logic.
    Decouples CustomTkinter from TensorFlow and Pandas.
    """
    
    def __init__(self):
        self.tracker = Tracker()
        self.lstm_engine = TfLstmEngine()
        self.heuristic_engine = HeuristicEngine()
        self.wheel_math = WheelMath(settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS)
        
        self._lock = threading.Lock()
        
        # State
        self.current_capital = self.tracker.data.get("current_capital", 1000)
        self.selected_engine = "TF-LSTM" # Default
        
        self.latest_ev = None
        self.latest_prob = None
        
        self.risk_percentage = settings.DEFAULT_RISK_PCT
        self.history_length = 100
        self.manual_percentages = {num: 0.0 for num in settings.VALID_NUMBERS}
        
        self.is_processing = False
        
        # Train TF initially if we have history
        history = self.tracker.get_recent_actuals(1000)
        if len(history) > settings.LSTM_SEQUENCE_LENGTH:
            # We do this in a thread to not block UI startup if possible
            threading.Thread(target=self._initial_train, args=(history,), daemon=True).start()
            
    def _initial_train(self, history):
        self.lstm_engine.train(history, epochs=5)
        
    def get_current_history(self, limit=100) -> list:
        return self.tracker.get_recent_actuals(limit)
        
    def process_new_actual(self, actual: int, predicted: Optional[int], 
                          profit_change: int, callback: Callable):
        """Processes a new spin result."""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        def _task():
            try:
                # 1. Update Tracker
                with self._lock:
                    self.tracker.record_result(actual, predicted, profit_change)
                    self.current_capital = self.tracker.data["current_capital"]
                    self.latest_ev = self.wheel_math.calculate_ev(actual)
                    self.latest_prob = self.wheel_math.true_probs.get(actual, 0.0)
                
                # 2. Incremental Train TF (if we have enough history)
                history = self.tracker.get_recent_actuals(100)
                if len(history) > settings.LSTM_SEQUENCE_LENGTH:
                    self.lstm_engine.train(history, epochs=settings.TF_EPOCHS_INCREMENTAL)
                    
            finally:
                self.is_processing = False
                # Notify UI to refresh
                callback()
                
        threading.Thread(target=_task, daemon=True).start()
        
    def get_predictions(self) -> list:
        """Gets predictions from the currently selected engine."""
        history = self.tracker.get_recent_actuals(self.history_length)
        
        if self.selected_engine == "TF-LSTM":
            preds = self.lstm_engine.predict_next(history)
            if preds is None:
                return []
            allocs = self.lstm_engine.get_token_allocation(preds, self.current_capital, self.risk_percentage)
        else:
            preds = self.heuristic_engine.predict_next(history)
            if preds is None:
                return []
                
            # Kombinasi bobot Heuristik (0.65) dan Manual (0.35)
            if hasattr(self, 'manual_percentages') and sum(self.manual_percentages.values()) > 0:
                for p in preds:
                    num = p["number"]
                    manual_conf = self.manual_percentages.get(num, 0.0) / 100.0
                    p["confidence"] = (0.65 * p["confidence"]) + (0.35 * manual_conf)
                preds.sort(key=lambda x: x["confidence"], reverse=True)
                
            allocs = self.heuristic_engine.get_token_allocation(preds, self.current_capital, self.risk_percentage)
            
        return allocs

    def get_stats(self) -> dict:
        return self.tracker.get_stats()
        
    def get_tf_metrics(self) -> dict:
        if not self.lstm_engine.history_metrics['loss']:
            return {"loss": None, "accuracy": None}
        return {
            "loss": self.lstm_engine.history_metrics['loss'][-1],
            "accuracy": self.lstm_engine.history_metrics['accuracy'][-1]
        }
