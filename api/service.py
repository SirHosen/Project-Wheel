# -*- coding: utf-8 -*-
"""PROMPT 19: framework-agnostic REST routing core.

The HTTP framework (FastAPI when installed, else stdlib http.server) is only a
thin transport. ALL routing, validation and serialization live here in pure
stdlib code so the API behaves identically regardless of transport and is fully
unit-testable without any web framework.

A `backend` object supplies the data/logic. This keeps the heavy ViewModel
(TensorFlow, GUI engines) out of the routing layer and lets tests inject a
lightweight fake.

Backend protocol:
    version() -> str
    current_engine() -> str
    stats() -> dict
    history(limit: int) -> list[dict]
    predict(engine: str|None, history_length: int|None) -> list[dict]
    record(actual_number, predicted_number, profit_change,
           bets, engine_used, mode) -> dict   # returns refreshed stats
"""
from urllib.parse import parse_qs, urlparse

ENDPOINTS = [
    {"method": "GET", "path": "/health", "desc": "Liveness + app version"},
    {"method": "GET", "path": "/stats", "desc": "Aggregate stats (capital, win rate, streaks)"},
    {"method": "GET", "path": "/history?limit=N", "desc": "Recent spin records"},
    {"method": "POST", "path": "/predict", "desc": "Predictions from an engine {engine?, history_length?}"},
    {"method": "POST", "path": "/record", "desc": "Record a spin result {actual_number, profit_change, ...}"},
]


class ApiError(Exception):
    """A client-facing error carrying an HTTP status code."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class ApiService:
    def __init__(self, backend):
        self.backend = backend

    # ------------------------------------------------------------------ #
    # Central dispatch
    # ------------------------------------------------------------------ #
    def handle(self, method, path, query=None, body=None):
        """Dispatch one request. Returns (status_code, json_dict).

        Never raises for ordinary client mistakes -- those become 4xx JSON
        envelopes. Unexpected errors become a 500 envelope.
        """
        method = (method or "GET").upper()
        parsed = urlparse(path or "/")
        clean = parsed.path.rstrip("/") or "/"
        q = dict(query or {})
        if parsed.query:
            for k, v in parse_qs(parsed.query).items():
                q.setdefault(k, v[0] if isinstance(v, list) else v)
        try:
            return self._route(method, clean, q, body if body is not None else {})
        except ApiError as e:
            return e.status, {"error": e.message}
        except Exception as e:  # pragma: no cover - defensive
            return 500, {"error": f"internal error: {e}"}

    def _route(self, method, path, q, body):
        if not isinstance(body, dict):
            raise ApiError(400, "request body must be a JSON object")

        if path == "/" and method == "GET":
            return 200, {
                "service": "Spin Wheel Predictor API",
                "version": self.backend.version(),
                "endpoints": ENDPOINTS,
            }
        if path == "/health" and method == "GET":
            return 200, {"status": "ok", "version": self.backend.version()}
        if path == "/stats" and method == "GET":
            return 200, dict(self.backend.stats())
        if path == "/history" and method == "GET":
            limit = self._int(q.get("limit"), name="limit", default=50, lo=1, hi=100000)
            records = list(self.backend.history(limit))
            return 200, {"count": len(records), "records": records}
        if path == "/predict" and method == "POST":
            engine = body.get("engine")
            if engine is not None and not isinstance(engine, str):
                raise ApiError(400, "engine must be a string")
            hl = body.get("history_length")
            if hl is not None:
                hl = self._int(hl, name="history_length", lo=1, hi=1000000)
            preds = list(self.backend.predict(engine=engine, history_length=hl))
            return 200, {
                "engine": engine or self.backend.current_engine(),
                "count": len(preds),
                "predictions": preds,
            }
        if path == "/record" and method == "POST":
            if "actual_number" not in body:
                raise ApiError(400, "actual_number is required")
            actual = self._int(body.get("actual_number"), name="actual_number")
            predicted = body.get("predicted_number")
            if predicted is not None:
                predicted = self._int(predicted, name="predicted_number")
            if "profit_change" not in body:
                raise ApiError(400, "profit_change is required")
            profit = self._num(body.get("profit_change"), name="profit_change")
            bets = body.get("bets")
            if bets is not None and not isinstance(bets, list):
                raise ApiError(400, "bets must be a list")
            engine_used = body.get("engine_used")
            mode = body.get("mode")
            stats = self.backend.record(
                actual_number=actual,
                predicted_number=predicted,
                profit_change=profit,
                bets=bets,
                engine_used=engine_used,
                mode=mode,
            )
            return 200, {"ok": True, "stats": dict(stats)}
        raise ApiError(404, f"no route for {method} {path}")

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _int(v, name, default=None, lo=None, hi=None):
        if v is None:
            if default is not None:
                return default
            raise ApiError(400, f"{name} is required")
        if isinstance(v, bool):
            raise ApiError(400, f"{name} must be an integer")
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise ApiError(400, f"{name} must be an integer")
        if lo is not None and iv < lo:
            raise ApiError(400, f"{name} must be >= {lo}")
        if hi is not None and iv > hi:
            raise ApiError(400, f"{name} must be <= {hi}")
        return iv

    @staticmethod
    def _num(v, name):
        if isinstance(v, bool) or v is None:
            raise ApiError(400, f"{name} must be a number")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ApiError(400, f"{name} must be a number")
        return int(f) if f.is_integer() else f


class RealBackend:
    """Production backend wrapping the real Tracker + (lazily) the ViewModel.

    `/stats`, `/history`, `/record` only need the lightweight Tracker, so they
    work even when TensorFlow is unavailable. `/predict` lazily constructs the
    full ViewModel (which initializes the engines) on first use.
    """

    def __init__(self, vm=None, tracker=None):
        self._vm = vm
        self._tracker = tracker

    def version(self):
        try:
            from config import settings
            return getattr(settings, "APP_VERSION", "?")
        except Exception:
            return "?"

    @property
    def tracker(self):
        # Once the VM exists, always use its tracker to avoid divergence.
        if self._vm is not None:
            return self._vm.tracker
        if self._tracker is None:
            from data.tracker import Tracker
            self._tracker = Tracker()
        return self._tracker

    @property
    def vm(self):
        if self._vm is None:
            from gui.viewmodels.main_viewmodel import MainViewModel
            self._vm = MainViewModel()
        return self._vm

    def current_engine(self):
        if self._vm is not None:
            return getattr(self._vm, "selected_engine", "Ensemble")
        return "Ensemble"

    def stats(self):
        return self.tracker.get_advanced_stats()

    def history(self, limit):
        hist = self.tracker.data.get("history", [])
        return hist[-limit:]

    def predict(self, engine=None, history_length=None):
        vm = self.vm
        if engine:
            vm.selected_engine = engine
        if history_length:
            vm.history_length = history_length
        return vm.get_predictions()

    def record(self, actual_number, predicted_number, profit_change,
               bets=None, engine_used=None, mode=None):
        self.tracker.record_result(
            actual_number, predicted_number, profit_change,
            bet_snapshot=bets, engine_used=engine_used, mode=mode,
        )
        return self.tracker.get_stats()
