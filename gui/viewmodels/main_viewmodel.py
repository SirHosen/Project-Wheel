# -*- coding: utf-8 -*-
import threading
from typing import Callable, Optional

from config import settings
from data.tracker import Tracker
from predictors.tf_lstm_engine import TfLstmEngine
from predictors.heuristic_engine import HeuristicEngine
from predictors.markov_engine import MarkovEngine
from core.wheel_math import WheelMath
from core.betting import kelly_allocation

class MainViewModel:
    """
    ViewModel coordinating the UI state and the business logic.
    Decouples CustomTkinter from TensorFlow and Pandas.
    """
    
    def __init__(self):
        self.tracker = Tracker()
        self.lstm_engine = TfLstmEngine()
        self.heuristic_engine = HeuristicEngine()
        self.markov_engine = MarkovEngine()
        self.wheel_math = WheelMath(settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS)
        
        self._lock = threading.Lock()
        
        # State
        self.current_capital = self.tracker.data.get("current_capital", 1000)
        # Default engine: Markov is the most robust for win-rate (frequency
        # prior when data is sparse, transition learning as data grows).
        self.selected_engine = "Markov"
        
        self.latest_ev = None
        self.latest_prob = None
        
        self.risk_percentage = settings.DEFAULT_RISK_PCT
        self.history_length = 100
        self.manual_percentages = {num: 0.0 for num in settings.VALID_NUMBERS}

        # --- Live adaptive prior (self-updating manual %) ---
        # When the user LOCKS their manual %, we treat it as a Bayesian prior
        # worth one full wheel of observations (54 segments). Every confirmed
        # spin then nudges the displayed % toward the wheel's real behaviour:
        #   live%[n] = (prior_counts[n] + observed_count[n]) / (54 + total_obs)
        # so early on it respects your input, and over many spins it converges
        # to the true frequency you are actually seeing.
        self.manual_locked = False
        self.WHEEL_PRIOR_STRENGTH = len(settings.SPINWHEEL_SEQUENCE)  # 54
        self.manual_prior_counts = {num: 0.0 for num in settings.VALID_NUMBERS}

        self.is_processing = False
        
        # Train TF initially if we have history
        history = self.tracker.get_recent_actuals(1000)
        if len(history) > settings.LSTM_SEQUENCE_LENGTH:
            # We do this in a thread to not block UI startup if possible
            threading.Thread(target=self._initial_train, args=(history,), daemon=True).start()
            
    def _initial_train(self, history):
        # If a GPU-trained model was restored from disk, a light refresh is
        # enough; otherwise do a full bulk train on the GPU and persist it so
        # future launches load instantly instead of retraining.
        if getattr(self.lstm_engine, "_trained", False):
            self.lstm_engine.train(history, epochs=settings.TF_EPOCHS_INCREMENTAL)
        else:
            self.lstm_engine.train(history, bulk=True)
            self.lstm_engine.save()
        
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
                    # Self-updating manual %: nudge the live distribution toward
                    # the wheel's real behaviour after each confirmed spin.
                    if self.manual_locked:
                        self.refresh_live_percentages()
                
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
        """Gets predictions from the currently selected engine.

        The manual probability input is treated as a user-supplied PRIOR and is
        blended into the output of WHICHEVER engine is active (weight 0.35),
        so it always has an effect — not only for the heuristic engine.
        """
        history = self.tracker.get_recent_actuals(self.history_length)

        if self.selected_engine == "TF-LSTM":
            engine = self.lstm_engine
        elif self.selected_engine == "Markov":
            engine = self.markov_engine
        else:
            engine = self.heuristic_engine

        preds = engine.predict_next(history)
        if not preds:
            return []

        # Blend manual probabilities (Signal weight 0.35) into any engine.
        if hasattr(self, "manual_percentages") and sum(self.manual_percentages.values()) > 0:
            for p in preds:
                manual_conf = self.manual_percentages.get(p["number"], 0.0) / 100.0
                p["confidence"] = (0.65 * p["confidence"]) + (0.35 * manual_conf)
            preds.sort(key=lambda x: x["confidence"], reverse=True)

        # EV-aware sizing: only stake on +EV numbers (half-Kelly), else recommend skip.
        return kelly_allocation(preds, self.current_capital, self.risk_percentage)

    def get_predictions_async(self, callback):
        """Run prediction OFF the UI thread, then invoke callback(allocs).
        Prevents the window from freezing while the LSTM model predicts."""
        def _task():
            try:
                allocs = self.get_predictions()
            except Exception:
                import logging
                logging.exception("Prediction failed")
                allocs = []
            callback(allocs)
        threading.Thread(target=_task, daemon=True).start()

    def set_initial_capital(self, amount: int):
        """Set the starting bankroll (modal token)."""
        with self._lock:
            self.current_capital = int(amount)
            self.tracker.data["current_capital"] = int(amount)
            self.tracker.save_data()

    # ------------------------------------------------------------------ #
    # Live adaptive manual percentages
    # ------------------------------------------------------------------ #
    def lock_manual_percentages(self) -> dict:
        """Freeze the current manual % as a Bayesian prior and switch to
        auto-update mode. From now on, every confirmed spin refreshes the
        percentages toward the wheel's observed behaviour.

        If the user left everything at 0, we seed the prior from the physical
        54-segment wheel layout so the feature is still useful out of the box.
        """
        from collections import Counter

        total = sum(self.manual_percentages.values())
        if total <= 0:
            counts = Counter(settings.SPINWHEEL_SEQUENCE)
            length = len(settings.SPINWHEEL_SEQUENCE)
            self.manual_percentages = {
                n: counts.get(n, 0) / length * 100.0 for n in settings.VALID_NUMBERS
            }
            total = sum(self.manual_percentages.values())

        # Normalise to exactly 100% so the prior is a clean distribution.
        self.manual_percentages = {
            n: (v / total * 100.0) for n, v in self.manual_percentages.items()
        }
        strength = self.WHEEL_PRIOR_STRENGTH
        self.manual_prior_counts = {
            n: self.manual_percentages[n] / 100.0 * strength
            for n in settings.VALID_NUMBERS
        }
        self.manual_locked = True
        self.refresh_live_percentages()
        return self.manual_percentages

    def unlock_manual_percentages(self):
        """Return to static manual mode (entries become editable again)."""
        self.manual_locked = False

    def refresh_live_percentages(self) -> dict:
        """Recompute the live percentages from the locked prior + observed
        results. No-op when not locked. Safe to call after every spin."""
        if not self.manual_locked:
            return self.manual_percentages
        from collections import Counter

        history = self.tracker.get_recent_actuals(1000)
        counts = Counter(history)
        denom = self.WHEEL_PRIOR_STRENGTH + len(history)
        if denom <= 0:
            return self.manual_percentages
        self.manual_percentages = {
            n: (self.manual_prior_counts.get(n, 0.0) + counts.get(n, 0)) / denom * 100.0
            for n in settings.VALID_NUMBERS
        }
        return self.manual_percentages

    def export_audit_report(self, md_path: str) -> list:
        """Generate a detailed audit bundle (Markdown + JSON + raw CSV) from the
        recorded history. Returns the list of written file paths."""
        from core.diagnostics import export_audit_bundle
        return export_audit_bundle(
            self.tracker.data,
            md_path,
            settings.SPINWHEEL_SEQUENCE,
            settings.VALID_NUMBERS,
            app_version=getattr(settings, "APP_VERSION", "dev"),
        )

    def get_stats(self) -> dict:
        return self.tracker.get_stats()

    def get_advanced_stats(self) -> dict:
        return self.tracker.get_advanced_stats()

    def get_bias_report(self) -> dict:
        """Walk-forward test: is the Markov engine's top pick beating the naive
        'always the most frequent number' baseline on the recorded history?
        A significant positive lift indicates exploitable autocorrelation
        (i.e. a real, profitable wheel bias)."""
        import math
        history = self.tracker.get_recent_actuals(1000)
        n = len(history)
        MIN_SAMPLE = 40
        if n < MIN_SAMPLE:
            return {
                "status": "insufficient",
                "message": (
                    f"Bias roda: belum diuji - kumpulkan {MIN_SAMPLE - n} hasil lagi "
                    f"(min {MIN_SAMPLE} untuk uji statistik)."
                ),
            }

        prior = self.markov_engine.prior
        base_top1 = max(prior, key=prior.get)

        mk_hits = base_hits = rounds = 0
        for i in range(6, n - 1):
            preds = self.markov_engine.predict_next(history[: i + 1])
            actual = history[i + 1]
            if preds and preds[0]["number"] == actual:
                mk_hits += 1
            if base_top1 == actual:
                base_hits += 1
            rounds += 1

        if rounds < 10:
            return {
                "status": "insufficient",
                "message": "Bias roda: belum cukup data untuk uji walk-forward.",
            }

        mk_rate = mk_hits / rounds
        base_rate = base_hits / rounds
        pooled = (mk_hits + base_hits) / (2 * rounds)
        denom = math.sqrt(max(1e-9, pooled * (1 - pooled) * (2 / rounds)))
        z = (mk_rate - base_rate) / denom if denom > 0 else 0.0

        if z > 1.96 and mk_rate > base_rate:
            status = "edge"
            head = "TERDETEKSI (signifikan)"
        else:
            status = "no_edge"
            head = "belum terbukti"
        return {
            "status": status,
            "message": (
                f"Bias roda: {head}. Akurasi model {mk_rate*100:.1f}% vs "
                f"baseline {base_rate*100:.1f}% atas {rounds} ronde (z={z:.2f})."
            ),
        }

    def get_tf_metrics(self) -> dict:
        if not self.lstm_engine.history_metrics['loss']:
            return {"loss": None, "accuracy": None}
        return {
            "loss": self.lstm_engine.history_metrics['loss'][-1],
            "accuracy": self.lstm_engine.history_metrics['accuracy'][-1]
        }
