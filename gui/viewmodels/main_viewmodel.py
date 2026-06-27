# -*- coding: utf-8 -*-
import os
import threading
from typing import Callable, Optional

from config import settings
from data.tracker import Tracker
from predictors.tf_lstm_engine import TfLstmEngine
from predictors.legacy.heuristic_engine import HeuristicEngine  # Lab Mode only
from predictors.markov_engine import MarkovEngine
from predictors.bayesian_optimal import BayesianOptimalEngine
from core.wheel_math import WheelMath
from core.betting import net_kelly_portfolio
from core.bootstrap_ci import attach_confidence_intervals
from core.continuous_engine import ContinuousLearningEngine
from core.tilt import TiltDetector
from core.calibration import ReliabilityTracker

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
        # Provably-optimal Dirichlet-Multinomial predictor + statistically-gated
        # EV engine. Best single predictor for an i.i.d. wheel; only bets when a
        # real, robust edge exists, otherwise recommends SKIP.
        self.bayesian_engine = BayesianOptimalEngine()
        # Probability-reliability tracker (audit V5 #2): records (confidence,
        # was_correct) for the engine's #1 pick every confirmed round, persisting
        # to models/calibration_state.json. Lets the UI show how honest each
        # engine's confidence is (Brier / log-loss / ECE / reliability bins) and
        # can isotonically RE-MAP raw confidence onto observed hit-rates.
        # Audit fix (PROMPT 21): kalibrasi HARUS ditulis ke runtime/ (sama seperti
        # default core/calibration.py) supaya "Reset Data" -- yang mem-purge
        # runtime/calibration_state.json -- benar-benar ikut menghapusnya. Dulu
        # di-hardcode ke models/ sehingga state kalibrasi LOLOS dari reset (desync).
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _calib_rel = getattr(settings, "CALIBRATION_STATE_PATH", "runtime/calibration_state.json")
        self.calibration = ReliabilityTracker(
            path=_calib_rel if os.path.isabs(_calib_rel) else os.path.join(_root, _calib_rel)
        )
        self.wheel_math = WheelMath(settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS)
        # Unified continuous-learning brain: fuses physics + bayes + markov +
        # the GPU LSTM and learns each one's trust weight from every spin.
        self.continuous = ContinuousLearningEngine(
            lstm_engine=self.lstm_engine, markov_engine=self.markov_engine,
        )
        
        self._lock = threading.Lock()
        
        # State
        self.current_capital = self.tracker.data.get("current_capital", 1000)
        # Default engine: the continuous Ensemble brain, which fuses physics +
        # Bayesian frequency + Markov + the GPU LSTM and adapts their weights
        # from every confirmed spin.
        self.selected_engine = "Ensemble"
        # PROMPT 12: Variance Harvest mode is OPT-IN and OFF by default. When ON,
        # the conservative EV gate is bypassed in favour of tiny lottery-ticket
        # stakes on high-multiplier numbers (high variance, long-run -EV).
        self.harvest_mode = settings.HARVEST_MODE_DEFAULT
        
        self.latest_ev = None
        self.latest_prob = None
        # Last computed allocations; logged as a full bet snapshot per round.
        self.current_predictions = []
        
        self.risk_percentage = settings.DEFAULT_RISK_PCT
        self.history_length = 100
        # PROMPT 21 auto-training: hitung spin sejak retrain LSTM terakhir. Otak
        # Ensemble tetap adaptif tiap spin; retrain LSTM yang berat digerbang ke
        # tiap AUTO_TRAIN_EVERY_N_SPINS spin. last_auto_train menyimpan ringkasan
        # retrain terakhir untuk ditampilkan di panel "Live Learning".
        self._spins_since_train = 0
        self.last_auto_train = None
        self.manual_percentages = {num: 0.0 for num in settings.VALID_NUMBERS}

        # --- Live adaptive prior (self-updating manual %) ---
        # When the user LOCKS their manual %, we treat it as a Bayesian prior
        # worth one full wheel of observations (54 segments). Every confirmed
        # spin then nudges the displayed % toward the wheel's real behaviour:
        #   live%[n] = (prior_counts[n] + observed_count[n]) / (54 + total_obs)
        # so early on it respects your input, and over many spins it converges
        # to the true frequency you are actually seeing.
        self.manual_locked = False
        # PROMPT 14: the manual % is a Dirichlet prior. ONE strength parameter
        # (pseudo-observation count) now drives BOTH (a) how strongly the manual
        # prior is blended into the engine and (b) how long the locked prior
        # resists the observed live frequencies. User-adjustable via the slider.
        self.manual_prior_strength = float(settings.MANUAL_PRIOR_STRENGTH_DEFAULT)
        self.manual_prior_counts = {num: 0.0 for num in settings.VALID_NUMBERS}

        # PROMPT 16: anti-tilt guard. Watches for loss-chasing (N losing rounds
        # in a short window with rising stakes) and enforces a cooldown. This
        # protects bankroll/discipline only; it cannot beat a fair wheel.
        self.tilt_detector = TiltDetector(
            n_losses=settings.TILT_N_LOSSES,
            window_minutes=settings.TILT_WINDOW_MINUTES,
            cooldown_seconds=settings.TILT_COOLDOWN_SECONDS,
            require_rising=settings.TILT_REQUIRE_RISING,
            strict_rising=settings.TILT_STRICT_RISING,
            enabled=settings.TILT_DETECT_ENABLED,
        )

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

    # ------------------------------------------------------------------ #
    # PROMPT 16: anti-tilt guard accessors
    # ------------------------------------------------------------------ #
    def tilt_status(self) -> dict:
        """Live cooldown status for the UI (auto-expires when time runs out)."""
        try:
            return self.tilt_detector.status()
        except Exception:
            return {"in_cooldown": False, "remaining_seconds": 0.0,
                    "tilt_triggered": False, "trigger_count": 0, "enabled": False}

    def acknowledge_tilt(self):
        """User chose to take a breather / dismiss -- clear the cooldown."""
        self.tilt_detector.clear_cooldown()
        
    def process_new_actual(self, actual: int, predicted: Optional[int], 
                          profit_change: int, callback: Callable,
                          bet_snapshot=None, engine_used=None, top1_hit=None):
        """Processes a new spin result."""
        if self.is_processing:
            return
            
        self.is_processing = True
        
        def _task():
            try:
                # Snapshot results BEFORE this spin so each model is graded fairly.
                history_before = self.tracker.get_recent_actuals(1000)
                # 1. Update Tracker
                with self._lock:
                    snap = bet_snapshot if bet_snapshot is not None else getattr(self, "current_predictions", None)
                    eng = engine_used if engine_used is not None else self.selected_engine
                    mode = "harvest" if getattr(self, "harvest_mode", False) else None
                    self.tracker.record_result(
                        actual, predicted, profit_change,
                        bet_snapshot=snap, engine_used=eng, mode=mode,
                        top1_hit=top1_hit,
                    )
                    self.current_capital = self.tracker.data["current_capital"]
                    self.latest_ev = self.wheel_math.calculate_ev(actual)
                    self.latest_prob = self.wheel_math.true_probs.get(actual, 0.0)
                    # Self-updating manual %: nudge the live distribution toward
                    # the wheel's real behaviour after each confirmed spin.
                    if self.manual_locked:
                        self.refresh_live_percentages()
                    # PROMPT 16: feed the anti-tilt guard. Total staked this
                    # round = sum of token_bet across the bet snapshot; a round
                    # counts as a loss when net profit is not positive.
                    try:
                        total_staked = 0.0
                        if snap:
                            total_staked = sum(
                                float(b.get("token_bet", 0) or 0) for b in snap
                            )
                        self.tilt_detector.record(
                            None, is_win=(profit_change > 0),
                            staked=total_staked,
                        )
                    except Exception:
                        import logging; logging.exception("tilt_detector.record failed")
                
                # 2. Continuous learning: grade every signal on this spin and
                # adapt the ensemble weights (physics / bayes / markov / lstm).
                try:
                    self.continuous.observe(actual, history_before)
                except Exception:
                    import logging; logging.exception("continuous.observe failed")

                # Calibration (audit V5 #2): grade the engine's TOP-1 pick's
                # stated confidence against the actual outcome, then persist.
                # This is what finally makes models/calibration_state.json real
                # and feeds the KALIBRASI panel + isotonic confidence re-map.
                try:
                    picks = snap or []
                    if picks:
                        top = max(picks, key=lambda b: (b.get("confidence") or 0.0))
                        conf = float(top.get("confidence") or 0.0)
                        was_correct = (top.get("number") == actual)
                        self.calibration.record(conf, was_correct, engine=eng)
                        self.calibration.fit_isotonic(eng)
                        self.calibration.save()
                except Exception:
                    import logging; logging.exception("calibration.record failed")

                # 3. AUTO-TRAINING (PROMPT 21): latih ulang LSTM tiap N spin, BUKAN
                # tiap spin. Otak Ensemble sudah mengadaptasi bobotnya tiap spin di
                # langkah continuous.observe() di atas (ringan); retrain LSTM yang
                # berat digerbang ke tiap AUTO_TRAIN_EVERY_N_SPINS spin agar hemat
                # CPU/DirectML. Setelah tiap retrain kita simpan modelnya.
                history = self.tracker.get_recent_actuals(100)
                auto_on = bool(getattr(settings, "AUTO_TRAIN_ENABLED", True))
                if auto_on and len(history) > settings.LSTM_SEQUENCE_LENGTH:
                    self._spins_since_train += 1
                    every = max(1, int(getattr(settings, "AUTO_TRAIN_EVERY_N_SPINS", 20)))
                    if self._spins_since_train >= every:
                        self._spins_since_train = 0
                        try:
                            epochs = int(getattr(settings, "AUTO_TRAIN_EPOCHS", 5))
                            self.lstm_engine.train(history, epochs=epochs)
                            self.lstm_engine.save()
                            from datetime import datetime
                            self.last_auto_train = {
                                "total_spins": int(self.tracker.get_stats().get("total", len(history))),
                                "history_used": len(history),
                                "epochs": epochs,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                            }
                        except Exception:
                            import logging; logging.exception("auto-train failed")
                    
            finally:
                self.is_processing = False
                # Notify UI to refresh
                callback()
                
        threading.Thread(target=_task, daemon=True).start()
        
    def get_predictions(self) -> list:
        """Gets predictions from the currently selected engine.

        The manual probability input is treated as a user-supplied DIRICHLET
        PRIOR and is blended into WHICHEVER engine is active with a
        data-dependent weight = strength / (strength + spins). The hunch counts
        most at cold start and fades automatically as real spins accumulate.
        """
        history = self.tracker.get_recent_actuals(self.history_length)

        # PROMPT 12: Variance Harvest mode (opt-in). Bypass the conservative EV
        # gate and buy tiny lottery tickets on high-multiplier numbers instead.
        if getattr(self, "harvest_mode", False):
            return self._harvest_predictions(history)

        # AI-Optimal: the Bayesian engine owns its own conservative EV sizing
        # (stakes only on a statistically-robust edge), so route it directly.
        if self.selected_engine == "AI-Optimal":
            allocs = self.bayesian_engine.recommend(
                history, self.current_capital, self.risk_percentage
            )
            # Blend the manual Dirichlet prior into the displayed confidence.
            # Weight = strength / (strength + spins): the hunch fades as data grows.
            from core.priors import apply_manual_prior
            apply_manual_prior(
                allocs, self.manual_percentages,
                self.manual_prior_strength, len(history),
            )
            self.current_predictions = allocs
            return allocs

        if self.selected_engine == "Ensemble":
            engine = self.continuous
        elif self.selected_engine == "TF-LSTM":
            engine = self.lstm_engine
        elif self.selected_engine == "Markov":
            engine = self.markov_engine
        else:
            engine = self.heuristic_engine

        preds = engine.predict_next(history)
        if not preds:
            self.current_predictions = []
            return []

        # Blend the manual Dirichlet prior into ANY engine. The weight is
        # data-dependent (strength / (strength + spins)), not a fixed 0.35.
        from core.priors import apply_manual_prior, has_manual_input
        _, _w = apply_manual_prior(
            preds, self.manual_percentages,
            self.manual_prior_strength, len(history),
        )
        if has_manual_input(self.manual_percentages) and _w > 0:
            preds.sort(key=lambda x: x["confidence"], reverse=True)

        # Quantify uncertainty: Bayesian preds already carry ci_low/ci_high/
        # support; for Heuristic/LSTM we bootstrap the history (resample 200x,
        # recompute confidence, 2.5/97.5 percentile). support falls back to the
        # observation count so the UI can badge evidence strength.
        try:
            attach_confidence_intervals(engine, history, preds)
        except Exception:
            import logging
            logging.exception("bootstrap CI failed")
        ci_map = {
            int(p["number"]): (p.get("ci_low"), p.get("ci_high"), p.get("support"))
            for p in preds
        }

        # EV-aware sizing: correlation-aware net-Kelly portfolio. Because exactly
        # one number wins per spin, stakes are mutually exclusive; this maximizes
        # expected log-growth over the full outcome distribution instead of
        # sizing each number independently (which over/under-counts that
        # correlation). Falls back to SKIP when nothing clears the EV+evidence
        # gate. Half-Kelly safety is the function default.
        allocs = net_kelly_portfolio(preds, self.current_capital, self.risk_percentage)

        # --- REALITY CHECK (anti-overconfidence guard) ---
        # A single model's self-reported confidence is NOT a real probability.
        # An under-trained LSTM can "collapse" onto a rare number (e.g. 30/40)
        # and look certain, tricking the EV sizer into a losing bet (the evidence
        # gate in kelly_allocation only fires for engines that expose a
        # `support` field; LSTM/Heuristic do not, so their fake confidence used
        # to slip through as a "bet 1" on a rare number). Before we risk ANY
        # tokens, cross-check each pick against the Bayesian posterior built from
        # the REAL observed frequencies. We keep a stake ONLY if that number is a
        # statistically-robust +EV edge (lower credible bound beats break-even);
        # otherwise we zero it out (SKIP). The engine may still SHOW its guess,
        # but it cannot stake tokens without statistical evidence -- this makes
        # every engine as safe as AI-Optimal when real tokens are on the line.
        try:
            robust = {
                r["number"]: bool(r.get("robust_positive"))
                for r in self.bayesian_engine.edge_report(history)
            }
            for a in allocs:
                if a.get("token_bet", 0) > 0 and not robust.get(a["number"], False):
                    a["token_bet"] = 0
                    a["is_positive_ev"] = False
        except Exception:
            import logging
            logging.exception("reality-check failed")

        # --- PROMPT 17: MULTI-ENGINE CONSENSUS FILTER ---
        # A single model's confidence is weak evidence. Cross-check each
        # surviving stake against the INDEPENDENT votes of the physics / bayes /
        # markov / lstm models: keep it only when >= CONSENSUS_MIN_AGREE of them
        # rank that number in their top-N, otherwise zero it (SKIP). This curbs
        # single-model overconfidence/noise. It FAILS OPEN when too few engines
        # are available, and -- to be honest -- it cannot manufacture an edge a
        # fair wheel doesn't give; it only filters risk.
        try:
            from core.consensus import apply_consensus_filter
            engine_dists = self.continuous.model_distributions(history)
            _, self.consensus_info = apply_consensus_filter(
                allocs, engine_dists,
                min_agree=settings.CONSENSUS_MIN_AGREE,
                top_n=settings.CONSENSUS_TOP_N,
                min_prob=settings.CONSENSUS_MIN_PROB,
                enabled=settings.CONSENSUS_FILTER_ENABLED,
            )
        except Exception:
            import logging
            logging.exception("consensus filter failed")

        # Carry the uncertainty bands onto the sized allocations so the
        # prediction cards can render CI + support badges.
        for a in allocs:
            ci = ci_map.get(int(a["number"]))
            if ci is not None:
                if a.get("ci_low") is None:
                    a["ci_low"] = ci[0]
                if a.get("ci_high") is None:
                    a["ci_high"] = ci[1]
                if a.get("support") is None:
                    a["support"] = ci[2]

        # Attach isotonically-calibrated confidence for DISPLAY only (audit V5
        # #2). Sizing/EV still use the raw confidence above; this extra field
        # lets the UI show "what this stated confidence has historically meant".
        # calibrate() returns the input unchanged until enough data is recorded,
        # so this is a safe no-op early on.
        try:
            for a in allocs:
                a["confidence_calibrated"] = self.calibration.calibrate(
                    a.get("confidence", 0.0), self.selected_engine
                )
        except Exception:
            import logging; logging.exception("confidence calibration map failed")

        self.current_predictions = allocs
        return allocs

    def _harvest_predictions(self, history) -> list:
        """Variance Harvest sizing: tiny lottery tickets on high-multiplier
        numbers. Bypasses the EV/evidence gate ON PURPOSE -- this is opt-in,
        capped, and long-run -EV by design (see core/harvest.py).
        """
        from core.harvest import harvest_picks
        # Belief distribution from the selected engine; fall back to Markov,
        # which always yields a full distribution.
        try:
            if self.selected_engine == "AI-Optimal":
                base = self.bayesian_engine.recommend(
                    history, self.current_capital, self.risk_percentage)
            elif self.selected_engine == "Ensemble":
                base = self.continuous.predict_next(history)
            elif self.selected_engine == "TF-LSTM":
                base = self.lstm_engine.predict_next(history)
            elif self.selected_engine == "Markov":
                base = self.markov_engine.predict_next(history)
            else:
                base = self.markov_engine.predict_next(history)
        except Exception:
            import logging
            logging.exception("harvest belief engine failed; using Markov")
            base = self.markov_engine.predict_next(history)
        if not base:
            base = self.markov_engine.predict_next(history)
        # Blend the manual Dirichlet prior (consistent with the other engine
        # paths) before handing the belief to the harvest sizer.
        from core.priors import apply_manual_prior
        apply_manual_prior(
            base, self.manual_percentages,
            self.manual_prior_strength, len(history),
        )
        allocs = harvest_picks(base, self.current_capital)
        self.current_predictions = allocs
        return allocs

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
        strength = self.manual_prior_strength
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
        denom = self.manual_prior_strength + len(history)
        if denom <= 0:
            return self.manual_percentages
        self.manual_percentages = {
            n: (self.manual_prior_counts.get(n, 0.0) + counts.get(n, 0)) / denom * 100.0
            for n in settings.VALID_NUMBERS
        }
        return self.manual_percentages

    def set_manual_prior_strength(self, value) -> float:
        """Set the Dirichlet prior strength (pseudo-observation count).

        Clamped to [MIN, MAX]. If the manual % is currently locked, the prior
        pseudo-counts are rescaled to the new strength and the live percentages
        are recomputed immediately so the slider has an instant, visible effect.
        """
        from core.priors import clamp_strength

        self.manual_prior_strength = clamp_strength(
            value,
            settings.MANUAL_PRIOR_STRENGTH_MIN,
            settings.MANUAL_PRIOR_STRENGTH_MAX,
            settings.MANUAL_PRIOR_STRENGTH_DEFAULT,
        )
        if self.manual_locked:
            self.manual_prior_counts = {
                n: self.manual_percentages.get(n, 0.0) / 100.0 * self.manual_prior_strength
                for n in settings.VALID_NUMBERS
            }
            self.refresh_live_percentages()
        return self.manual_prior_strength

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

    def get_calibration_report(self, engine=None) -> dict:
        """Confidence-reliability summary for the KALIBRASI panel (audit V5 #2).

        Returns Brier score, log-loss, Expected Calibration Error (ECE) and the
        reliability bins (mean predicted confidence vs observed hit-rate) for
        BOTH the active engine and the global pool. Read-only; does not touch
        betting stats. On a fair wheel, honest confidence hugs the diagonal and
        ECE stays small; a large ECE means the engine is over/under-confident.
        """
        eng = engine or self.selected_engine
        empty = {"n": 0, "brier": None, "log_loss": None, "ece": None, "bins": []}
        try:
            return {
                "engine": eng,
                "global": self.calibration.summary(None),
                "per_engine": self.calibration.summary(eng),
            }
        except Exception:
            import logging; logging.exception("calibration report failed")
            return {"engine": eng, "global": dict(empty), "per_engine": dict(empty)}

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

    def _usable_vision_observations(self, stopped_only=False, min_confidence=0.0):
        """Filtered camera observations (read-only helper for the vision panel)."""
        from vision.observation_log import load_observations
        return [
            o for o in load_observations()
            if o["number"] in settings.VALID_NUMBERS
            and (not stopped_only or o["stopped"])
            and (o["confidence"] is None or o["confidence"] >= min_confidence)
        ]

    def get_vision_analysis(self, stopped_only=False, min_confidence=0.0) -> dict:
        """Chi-square fairness of the PHYSICAL wheel vs its design layout, built
        from camera observations. READ-ONLY: does not touch betting stats or the
        wheel prior. Detects long-run physical bias if any; not a spin forecast."""
        from collections import Counter
        from core.wheel_bias import chi_square_gof, design_distribution, standardized_residual

        usable = self._usable_vision_observations(stopped_only, min_confidence)
        n = len(usable)
        counts = Counter(o["number"] for o in usable)
        design = design_distribution(settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS)
        expected = {k: n * design[k] for k in settings.VALID_NUMBERS}
        rows = [
            {
                "number": k,
                "observed": counts.get(k, 0),
                "expected": expected[k],
                "share": design[k],
                "resid": standardized_residual(counts.get(k, 0), expected[k]),
            }
            for k in settings.VALID_NUMBERS
        ]
        loaded = int(sum(self.bayesian_engine.observed_counts.values()))
        if n == 0:
            return {
                "status": "empty", "n": 0, "rows": rows, "loaded_into_engine": loaded,
                "message": ("Belum ada observasi kamera. Jalankan dulu: "
                            "python scripts/wheel_cam.py --rounds 50"),
            }
        chi2, dof, p = chi_square_gof(counts, expected)
        biased = (p < 0.05) and (n >= 30)
        return {
            "status": "biased" if biased else "fair", "n": n, "chi2": chi2,
            "dof": dof, "p": p, "biased": biased, "rows": rows,
            "loaded_into_engine": loaded,
        }

    def apply_vision_learning(self, stopped_only=False, min_confidence=0.0) -> dict:
        """Write the vision report + models/wheel_prior.json from camera
        observations, then RELOAD the Bayesian engine so the observed counts feed
        its posterior immediately (no app restart needed). Betting stats are
        untouched."""
        import json
        import os
        from collections import Counter
        from datetime import datetime, timezone
        from core.wheel_bias import chi_square_gof, design_distribution
        from scripts.learn_from_vision import build_report, REPORT_PATH, WHEEL_PRIOR_PATH

        usable = self._usable_vision_observations(stopped_only, min_confidence)
        n = len(usable)
        if n == 0:
            return {"status": "empty", "n": 0,
                    "message": "Tidak ada observasi kamera untuk dipelajari."}
        counts = Counter(o["number"] for o in usable)
        design = design_distribution(settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS)
        expected = {k: n * design[k] for k in settings.VALID_NUMBERS}
        chi2, dof, p = chi_square_gof(counts, expected)
        biased = (p < 0.05) and (n >= 30)

        report, ts = build_report(
            usable, counts, design, expected, chi2, dof, p, biased,
            stopped_only, False, 30,
        )
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        payload = {
            "counts": {str(k): int(counts.get(k, 0)) for k in settings.VALID_NUMBERS},
            "n_obs": n, "chi_square": round(chi2, 4), "dof": dof,
            "p_value": round(p, 6), "biased": bool(biased),
            "stopped_only": bool(stopped_only), "min_confidence": min_confidence,
            "updated_at": ts,
            "source": "vision camera observations (GUI panel)",
        }
        os.makedirs(os.path.dirname(WHEEL_PRIOR_PATH), exist_ok=True)
        with open(WHEEL_PRIOR_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # Engine picks up the new observations right away.
        self.bayesian_engine.reload_observed_counts()
        return {
            "status": "ok", "n": n, "chi2": chi2, "dof": dof, "p": p,
            "biased": biased, "report_path": REPORT_PATH, "prior_path": WHEEL_PRIOR_PATH,
        }

    def get_tf_metrics(self) -> dict:
        if not self.lstm_engine.history_metrics['loss']:
            return {"loss": None, "accuracy": None}
        return {
            "loss": self.lstm_engine.history_metrics['loss'][-1],
            "accuracy": self.lstm_engine.history_metrics['accuracy'][-1]
        }

    def get_learning_status(self) -> dict:
        """Live status of the continuous-learning ensemble (for the UI panel)."""
        try:
            return self.continuous.learning_status()
        except Exception:
            return {"n_observed": 0, "weights": {}, "accuracy": {}, "lstm_ready": False}

    def auto_train_status(self) -> dict:
        """Status auto-training LSTM untuk panel UI: tiap berapa spin retrain,
        sudah berapa spin sejak retrain terakhir, sisa berapa lagi."""
        every = max(1, int(getattr(settings, "AUTO_TRAIN_EVERY_N_SPINS", 20)))
        since = int(getattr(self, "_spins_since_train", 0))
        return {
            "enabled": bool(getattr(settings, "AUTO_TRAIN_ENABLED", True)),
            "every": every,
            "since": since,
            "remaining": max(0, every - since),
            "last": getattr(self, "last_auto_train", None),
        }
