# -*- coding: utf-8 -*-
"""PROMPT 18: durable SQLite backing store for the spin history.

The app's in-memory state has always been a single dict:

    {current_capital, total_predictions, wins, losses, profit, history: [...]}

This module persists that EXACT shape in a SQLite database
(`data/history.db`) so storage is transactional and crash-safe, while keeping
full fidelity (every record round-trips through a JSON blob, so any extra keys
survive). JSON stays a first-class import/export + mirror format, and existing
`history.json` files auto-migrate on first launch -- nothing is ever deleted.

Pure & dependency-light: stdlib `sqlite3` + `json` only (no pandas / GUI), so
it is fully unit-testable headless.
"""
import json
import os
import sqlite3

META_KEYS = ("current_capital", "total_predictions", "wins", "losses", "profit")
META_DEFAULTS = {
    "current_capital": 1000,
    "total_predictions": 0,
    "wins": 0,
    "losses": 0,
    "profit": 0,
}


def empty_data() -> dict:
    """A fresh, valid data dict in the canonical shape."""
    d = dict(META_DEFAULTS)
    d["history"] = []
    return d


class SqliteStore:
    """SQLite-backed persistence for the tracker's data dict."""

    def __init__(self, db_path="data/history.db"):
        self.db_path = db_path
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Schema / connection
    # ------------------------------------------------------------------ #
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    actual_number INTEGER,
                    predicted_number INTEGER,
                    profit_change INTEGER,
                    is_win INTEGER,
                    engine_used TEXT,
                    mode TEXT,
                    data TEXT NOT NULL
                )"""
            )
            conn.commit()

    # ------------------------------------------------------------------ #
    # State checks
    # ------------------------------------------------------------------ #
    def is_empty(self) -> bool:
        """True when neither history rows nor meta rows exist yet."""
        with self._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            m = conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
        return n == 0 and m == 0

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]

    # ------------------------------------------------------------------ #
    # Load / save
    # ------------------------------------------------------------------ #
    def load(self) -> dict:
        """Reconstruct the canonical data dict from the database."""
        data = empty_data()
        with self._connect() as conn:
            for row in conn.execute("SELECT key, value FROM meta"):
                try:
                    data[row["key"]] = json.loads(row["value"])
                except Exception:
                    data[row["key"]] = row["value"]
            history = []
            for row in conn.execute("SELECT data FROM history ORDER BY id ASC"):
                try:
                    history.append(json.loads(row["data"]))
                except Exception:
                    continue
            data["history"] = history
        return data

    @staticmethod
    def _record_row(record):
        """Map a record dict to the typed columns + a full JSON blob."""
        return (
            record.get("timestamp"),
            record.get("actual_number"),
            record.get("predicted_number"),
            record.get("profit_change", 0),
            1 if record.get("is_win") else 0,
            record.get("engine_used"),
            record.get("mode"),
            json.dumps(record),
        )

    def save_all(self, data: dict):
        """Full, transactional replace of meta + history from a data dict."""
        history = data.get("history", []) or []
        with self._connect() as conn:
            conn.execute("DELETE FROM history")
            conn.execute("DELETE FROM meta")
            meta_rows = []
            for k in META_KEYS:
                if k in data:
                    meta_rows.append((k, json.dumps(data[k])))
            # Preserve any non-standard top-level keys too (additive safety).
            for k, v in data.items():
                if k == "history" or k in META_KEYS:
                    continue
                meta_rows.append((k, json.dumps(v)))
            conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", meta_rows
            )
            conn.executemany(
                """INSERT INTO history
                    (timestamp, actual_number, predicted_number, profit_change,
                     is_win, engine_used, mode, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [self._record_row(r) for r in history],
            )
            conn.commit()

    def append_record(self, record: dict, meta: dict):
        """Append one history row and upsert meta (incremental write path)."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO history
                    (timestamp, actual_number, predicted_number, profit_change,
                     is_win, engine_used, mode, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                self._record_row(record),
            )
            if meta:
                conn.executemany(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    [(k, json.dumps(meta[k])) for k in meta],
                )
            conn.commit()

    # ------------------------------------------------------------------ #
    # Migration / import / export
    # ------------------------------------------------------------------ #
    def migrate_from_json(self, json_path) -> bool:
        """Import a legacy history.json into the DB. Returns True on success.

        Only meant to run when the DB is empty; the caller decides that.
        """
        if not os.path.exists(json_path):
            return False
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        self.save_all(data)
        return True

    def import_json(self, json_path, mode="replace") -> dict:
        """Load a JSON file into the DB. mode='replace' overwrites; 'append'
        keeps existing history and appends imported records. Returns the
        resulting data dict."""
        with open(json_path, "r") as f:
            incoming = json.load(f)
        if not isinstance(incoming, dict):
            raise ValueError("JSON file is not a valid history object")
        if mode == "append":
            current = self.load()
            current_hist = current.get("history", [])
            incoming_hist = incoming.get("history", [])
            merged = dict(current)
            merged["history"] = current_hist + incoming_hist
            # Recompute aggregate meta so totals stay consistent.
            merged.update(self._recompute_meta(merged["history"],
                                               current.get("current_capital", 1000)))
            self.save_all(merged)
            return merged
        self.save_all(incoming)
        return self.load()

    def export_json(self, json_path) -> str:
        """Write the current DB contents to a JSON file (canonical shape)."""
        data = self.load()
        d = os.path.dirname(json_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        return json_path

    @staticmethod
    def _recompute_meta(history, base_capital):
        """Recompute totals/profit from a history list (used on append import)."""
        wins = sum(1 for r in history if r.get("is_win"))
        total = len(history)
        profit = sum(r.get("profit_change", 0) or 0 for r in history)
        return {
            "total_predictions": total,
            "wins": wins,
            "losses": total - wins,
            "profit": profit,
        }
