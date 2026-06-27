import os
import json
import pandas as pd
from datetime import datetime

from data.sqlite_store import SqliteStore, empty_data
from config import settings

class Tracker:
    """Unified Data Pipeline and History Tracker."""
    
    def __init__(self, history_file=None, db_file=None):
        # Default lokasi riwayat = folder runtime/ (di-gitignore). Tetap bisa
        # dioverride lewat argumen (dipakai test & profil alternatif).
        if history_file is None:
            history_file = getattr(settings, "HISTORY_PATH", "runtime/history.json")
        self.history_file = history_file
        # Derive the SQLite path as a sibling of history_file when not given
        # explicitly. This keeps the production default identical
        # (data/history.json -> data/history.db) while ensuring any custom
        # history_file (tests, alternate profiles) gets ISOLATED storage
        # instead of silently sharing the global data/history.db.
        if db_file is None:
            base, _ = os.path.splitext(history_file)
            db_file = base + ".db"
        self.db_file = db_file
        self.ensure_directory()
        # PROMPT 18: SQLite is now the durable source of truth. history.json is
        # kept as a human-readable mirror/backup and stays the import/export
        # format. Legacy history.json files auto-migrate on first launch.
        self.store = SqliteStore(db_file)
        self.data = self.load_data()
        
    def ensure_directory(self):
        d = os.path.dirname(self.history_file)
        if d:
            os.makedirs(d, exist_ok=True)
        
    def load_data(self) -> dict:
        # 1) SQLite already holds data -> it is the source of truth.
        try:
            if not self.store.is_empty():
                return self.store.load()
        except Exception as e:
            print(f"Error reading SQLite store: {e}")
        # 2) Empty DB: auto-migrate a legacy history.json exactly once.
        if os.path.exists(self.history_file):
            try:
                if self.store.migrate_from_json(self.history_file):
                    import shutil
                    import logging
                    backup = self.history_file + ".migrated"
                    try:
                        shutil.copy2(self.history_file, backup)
                        logging.info(
                            f"Migrated {self.history_file} -> SQLite ({self.db_file}); "
                            f"backup kept at {backup}"
                        )
                    except Exception:
                        pass
                    return self.store.load()
            except Exception:
                import shutil
                import logging
                corrupt_file = self.history_file.replace(".json", ".corrupt.json")
                try:
                    shutil.copy2(self.history_file, corrupt_file)
                    logging.error(
                        f"Failed to migrate {self.history_file}. Backed up to {corrupt_file}"
                    )
                except Exception:
                    pass
        # 3) Fresh start.
        return empty_data()
        
    def save_data(self):
        # Full durable save: primary SQLite store (transactional) + JSON mirror.
        try:
            self.store.save_all(self.data)
        except Exception as e:
            print(f"Error saving to SQLite: {e}")
        self._save_json_mirror()

    def _save_json_mirror(self):
        # Mirror to JSON as a backup + automatic export (additive legacy format).
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Error saving data: {e}")
            
    def record_result(self, actual_number: int, predicted_number: int, profit_change: int,
                      bet_snapshot=None, engine_used=None, mode=None, top1_hit=None):
        # A round counts as a betting WIN only when an actual stake landed on the
        # winning number. The UI passes predicted_number = actual ONLY when the
        # matched card had token_bet > 0 (audit V3 #1/#2), so this stays honest:
        # a correct-but-zero-stake (SKIP) guess is NOT a win.
        is_win = (actual_number == predicted_number) if predicted_number else False
        
        self.data["total_predictions"] += 1
        if is_win:
            self.data["wins"] += 1
        else:
            self.data["losses"] += 1
            
        self.data["profit"] += profit_change
        self.data["current_capital"] += profit_change
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "actual_number": actual_number,
            "predicted_number": predicted_number,
            "profit_change": profit_change,
            "is_win": is_win,
        }
        # PROMPT 1: full per-round bet snapshot, stored ALWAYS (win or loss) so
        # per-number accuracy / ROI can be audited. Backward compatible: legacy
        # records simply carry an empty "bets" list.
        if bet_snapshot:
            record["bets"] = [
                {
                    "number": b.get("number"),
                    "token_bet": b.get("token_bet", 0),
                    "confidence": b.get("confidence"),
                    "ev_per_token": b.get("ev_per_token"),
                    "is_positive_ev": b.get("is_positive_ev"),
                    "support": b.get("support"),
                }
                for b in bet_snapshot
            ]
        else:
            record["bets"] = []
        if engine_used is not None:
            record["engine_used"] = engine_used
        # PROMPT 12: tag the betting mode so Variance Harvest rounds can be
        # audited separately. Legacy/conservative rounds simply omit this key.
        if mode is not None:
            record["mode"] = mode
        # Honest top-1 guess accuracy, tracked SEPARATELY from betting win rate
        # (audit V3 #1). None for legacy calls that don't supply it.
        if top1_hit is not None:
            record["top1_hit"] = bool(top1_hit)
            # Mark as an HONEST, live-captured top-1 grade: recorded EVERY round
            # (win OR loss) at spin time. Only these count toward top-1 accuracy,
            # so legacy/backfilled grades (stored ONLY on wins -> selection bias)
            # can never inflate it to a fake ~100% (audit V5 #1).
            record["top1_graded_live"] = True
        self.data["history"].append(record)

        # Incremental O(1) durable write (audit V3 #3): append a single row to
        # SQLite instead of rewriting the whole table every spin, then mirror to
        # JSON. Falls back to a full save if the incremental write fails.
        try:
            meta = {
                k: self.data[k]
                for k in ("current_capital", "total_predictions",
                          "wins", "losses", "profit")
                if k in self.data
            }
            self.store.append_record(record, meta)
        except Exception as e:
            print(f"Error appending to SQLite: {e}")
            try:
                self.store.save_all(self.data)
            except Exception as e2:
                print(f"Error saving to SQLite: {e2}")
        self._save_json_mirror()
        
    def get_sessions(self, gap_minutes=30) -> list:
        """Split history into sessions separated by idle gaps >= gap_minutes.

        Returns a list of sessions, each a list of records in order. Records
        without parseable timestamps cannot start a new session boundary; they
        stay attached to the current run (backward compatible with legacy data).
        """
        from core.sessions import parse_ts
        history = self.data.get("history", [])
        sessions = []
        cur = []
        prev_ts = None
        for rec in history:
            ts = parse_ts(rec.get("timestamp", ""))
            if cur and prev_ts is not None and ts is not None:
                gap_min = (ts - prev_ts).total_seconds() / 60.0
                if gap_min >= gap_minutes:
                    sessions.append(cur)
                    cur = []
            cur.append(rec)
            if ts is not None:
                prev_ts = ts
        if cur:
            sessions.append(cur)
        return sessions

    def per_session_stats(self, gap_minutes=30, valid_numbers=None, sequence=None) -> list:
        """Per-session summary: n_spins, dominant number, win rate, profit, chi^2."""
        from collections import Counter
        from core.sessions import chi_square_gof
        if valid_numbers is None or sequence is None:
            try:
                from config import settings
                valid_numbers = valid_numbers or settings.VALID_NUMBERS
                sequence = sequence or settings.SPINWHEEL_SEQUENCE
            except Exception:
                valid_numbers = valid_numbers or []
                sequence = sequence or []
        out = []
        for i, sess in enumerate(self.get_sessions(gap_minutes)):
            actuals = [r.get("actual_number") for r in sess if r.get("actual_number") is not None]
            n = len(actuals)
            cnt = Counter(actuals)
            dominant = cnt.most_common(1)[0][0] if cnt else None
            wins = sum(1 for r in sess if r.get("is_win"))
            profit = sum(r.get("profit_change", 0) for r in sess)
            out.append({
                "session_id": i,
                "start": sess[0].get("timestamp") if sess else None,
                "end": sess[-1].get("timestamp") if sess else None,
                "n_spins": n,
                "dominant_num": dominant,
                "win_rate": (wins / n * 100.0) if n else 0.0,
                "profit": profit,
                "chi_square": chi_square_gof(actuals, valid_numbers, sequence),
                "actuals": actuals,
            })
        return out

    def session_drift(self, gap_minutes=30, alpha=0.05) -> dict:
        """KS 2-sample drift verdict across the session timeline."""
        from core.sessions import detect_session_drift
        stats = self.per_session_stats(gap_minutes=gap_minutes)
        return detect_session_drift([s["actuals"] for s in stats], alpha=alpha)

    def get_recent_actuals(self, limit=100) -> list:
        """Returns a list of the recent actual numbers for prediction models."""
        history = self.data.get("history", [])
        return [record["actual_number"] for record in history][-limit:]
        
    def get_stats(self) -> dict:
        total = self.data["total_predictions"]
        win_rate = (self.data["wins"] / total * 100) if total > 0 else 0.0
        # Honest top-1 guess accuracy (audit V3 #1): share of rounds whose #1
        # pick matched the result, INDEPENDENT of staking. Distinct from the
        # betting win rate above; only graded over rounds that recorded it.
        history = self.data.get("history", [])
        # Only grade rounds captured LIVE (top1_graded_live). Legacy rounds were
        # recorded under the old win-only rule, so their backfilled top1_hit is
        # selection-biased (stored only on wins -> a fake ~100%). Excluding them
        # keeps this metric honest: it starts clean and fills from new rounds.
        graded = [r for r in history if r.get("top1_graded_live")]
        top1_acc = (sum(1 for r in graded if r.get("top1_hit")) / len(graded) * 100) if graded else None
        return {
            "capital": self.data["current_capital"],
            "total": total,
            "wins": self.data["wins"],
            "losses": self.data["losses"],
            "profit": self.data["profit"],
            "win_rate": win_rate,
            "top1_accuracy": top1_acc,
            "top1_graded": len(graded),
        }
        
    def export_csv(self, output_path=None):
        """Exports history to CSV using Pandas for analytics.

        The nested per-round bet snapshot is encoded as VALID JSON (audit V3 #4)
        so the `bets` column can be parsed back later, instead of an
        un-parseable Python dict repr (single quotes).
        """
        if output_path is None:
            output_path = os.path.join(
                getattr(settings, "EXPORT_DIR", "runtime"), "history.csv")
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        rows = []
        for rec in self.data["history"]:
            r = dict(rec)
            if "bets" in r:
                r["bets"] = json.dumps(r["bets"], ensure_ascii=False)
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path

    def export_json(self, output_path=None):
        """PROMPT 18: export full history + meta to a JSON file (canonical shape)."""
        if output_path is None:
            output_path = os.path.join(
                getattr(settings, "EXPORT_DIR", "runtime"), "history_export.json")
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=4)
        return output_path

    def import_json(self, input_path, mode="replace"):
        """PROMPT 18: import a JSON history file into SQLite, then reload.

        mode='replace' overwrites everything; mode='append' keeps the existing
        history and appends the imported records (totals are recomputed).
        Returns the refreshed stats dict.
        """
        self.data = self.store.import_json(input_path, mode=mode)
        # Keep the JSON mirror in sync after an import.
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception:
            pass
        return self.get_stats()

    def get_streak(self) -> int:
        """Returns the current streak of wins."""
        history = self.data.get("history", [])
        streak = 0
        for record in reversed(history):
            if record.get("is_win", False):
                streak += 1
            else:
                break
        return streak

    def get_max_win_streak(self) -> int:
        """Returns the longest win streak ever achieved."""
        history = self.data.get("history", [])
        best = cur = 0
        for record in history:
            if record.get("is_win", False):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def get_number_frequency(self) -> dict:
        """Returns how often each actual number has appeared."""
        from collections import Counter
        history = self.data.get("history", [])
        return dict(Counter(r["actual_number"] for r in history))

    def get_advanced_stats(self) -> dict:
        """Richer analytics for the dashboard (best/worst/avg/streak/frequency)."""
        base = self.get_stats()
        history = self.data.get("history", [])
        profits = [h.get("profit_change", 0) for h in history]
        base.update({
            "best_round": max(profits) if profits else 0,
            "worst_round": min(profits) if profits else 0,
            "avg_profit": (sum(profits) / len(profits)) if profits else 0.0,
            "current_streak": self.get_streak(),
            "max_streak": self.get_max_win_streak(),
            "frequency": self.get_number_frequency(),
        })
        return base
        
    def get_per_number_bet_stats(self) -> dict:
        """Per-number betting outcomes from the full snapshot log (PROMPT 1).
        Returns {number: {bets, wins, total_staked, net_profit, hit_rate, roi}}.
        """
        from core.diagnostics import per_number_bet_stats
        return per_number_bet_stats(self.data.get("history", []))

    def get_engine_bet_distribution(self) -> dict:
        """Aggregate performance grouped by the engine that made each call."""
        from core.diagnostics import engine_bet_distribution
        return engine_bet_distribution(self.data.get("history", []))

    def reset_data(self):
        """Reset SEMUA data ke kondisi awal yang benar-benar bersih.

        Audit V7 bug fix: dulu reset hanya mengosongkan riwayat spin, sementara
        state belajar (continuous_state), kalibrasi, risk, dan model LSTM tetap
        tertinggal -> desync (mis. history 0 record tapi n_observed masih 112).
        Sekarang reset juga membersihkan SEMUA state turunan itu, jadi "Reset
        Data" benar-benar mengembalikan otak app ke nol.
        """
        self.data = {
            "current_capital": 1000,
            "total_predictions": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0,
            "history": []
        }
        self.save_data()
        self._purge_learning_state()

    def _purge_learning_state(self):
        """Hapus file state belajar/training turunan supaya tidak desync dengan
        riwayat yang baru saja di-reset. Best-effort: kegagalan diabaikan."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _resolve(name, default):
            p = getattr(settings, name, default)
            return p if os.path.isabs(p) else os.path.join(root, p)

        targets = [
            _resolve("LEARNING_STATE_PATH", "runtime/continuous_state.json"),
            _resolve("CALIBRATION_STATE_PATH", "runtime/calibration_state.json"),
            _resolve("RISK_STATE_PATH", "runtime/risk_state.json"),
        ]
        # Model LSTM bisa .keras (TF>=2.11) atau .h5 (TF 2.10).
        lstm = _resolve("LSTM_MODEL_PATH", "runtime/lstm_spinwheel.keras")
        base, _ = os.path.splitext(lstm)
        targets += [lstm, base + ".keras", base + ".h5"]

        for path in set(targets):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

