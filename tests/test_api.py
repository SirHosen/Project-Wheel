# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Headless tests for PROMPT 19 REST API.

Covers the framework-agnostic routing core (ApiService) with a fake backend AND
a real end-to-end run of the stdlib http.server transport hit with `requests`.
No FastAPI needed (FastAPI path is a thin shim over the same service core).

Run: python test_api.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.service import ApiService, ApiError  # noqa: E402
from api.server import make_stdlib_server  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  - {name}")
    else:
        FAIL += 1
        print(f"  FAIL- {name}")
    assert cond, name  # pytest: surface failures as assertion errors


class FakeBackend:
    """Lightweight in-memory backend (no Tracker/VM/TF)."""

    def __init__(self):
        self.records = []
        self.engine = "Ensemble"
        self._history = [
            {"timestamp": "t1", "actual_number": 5, "profit_change": 10, "is_win": True},
            {"timestamp": "t2", "actual_number": 2, "profit_change": -5, "is_win": False},
            {"timestamp": "t3", "actual_number": 8, "profit_change": 20, "is_win": True},
        ]

    def version(self):
        return "1.26.0"

    def current_engine(self):
        return self.engine

    def stats(self):
        return {"capital": 1025, "total": 3, "wins": 2, "losses": 1,
                "profit": 25, "win_rate": 66.7}

    def history(self, limit):
        return self._history[-limit:]

    def predict(self, engine=None, history_length=None):
        if engine:
            self.engine = engine
        return [{"number": 5, "confidence": 0.4, "token_bet": 10},
                {"number": 2, "confidence": 0.25, "token_bet": 0}]

    def record(self, actual_number, predicted_number, profit_change,
               bets=None, engine_used=None, mode=None):
        self.records.append({
            "actual_number": actual_number,
            "predicted_number": predicted_number,
            "profit_change": profit_change,
            "bets": bets, "engine_used": engine_used, "mode": mode,
        })
        return {"capital": 1025 + profit_change, "total": 3 + len(self.records)}


def svc():
    return ApiService(FakeBackend())


# ---------------------------------------------------------------------- #
# Unit: routing core
# ---------------------------------------------------------------------- #
def test_root_and_health():
    s = svc()
    st, body = s.handle("GET", "/")
    check("GET / -> 200", st == 200)
    check("GET / lists endpoints", isinstance(body.get("endpoints"), list) and body["endpoints"])
    st, body = s.handle("GET", "/health")
    check("GET /health -> 200", st == 200)
    check("health status ok", body.get("status") == "ok")
    check("health carries version", body.get("version") == "1.26.0")


def test_trailing_slash():
    s = svc()
    st, _ = s.handle("GET", "/health/")
    check("trailing slash normalized", st == 200)


def test_stats():
    s = svc()
    st, body = s.handle("GET", "/stats")
    check("GET /stats -> 200", st == 200)
    check("stats has capital", body.get("capital") == 1025)


def test_history_default_and_limit():
    s = svc()
    st, body = s.handle("GET", "/history")
    check("GET /history -> 200", st == 200)
    check("history default returns all 3", body.get("count") == 3)
    # limit via query string in the path
    st, body = s.handle("GET", "/history?limit=2")
    check("history?limit=2 count 2", body.get("count") == 2)
    check("history limit keeps most recent", body["records"][-1]["actual_number"] == 8)
    # limit via explicit query dict
    st, body = s.handle("GET", "/history", query={"limit": 1})
    check("history query-dict limit 1", body.get("count") == 1)


def test_history_bad_limit():
    s = svc()
    st, body = s.handle("GET", "/history?limit=abc")
    check("history bad limit -> 400", st == 400)
    st, body = s.handle("GET", "/history?limit=0")
    check("history limit<1 -> 400", st == 400)


def test_predict():
    s = svc()
    st, body = s.handle("POST", "/predict", body={"engine": "Markov", "history_length": 50})
    check("POST /predict -> 200", st == 200)
    check("predict echoes engine", body.get("engine") == "Markov")
    check("predict returns list", body.get("count") == 2 and len(body["predictions"]) == 2)


def test_predict_defaults_engine():
    s = svc()
    st, body = s.handle("POST", "/predict", body={})
    check("predict no-engine -> current_engine", body.get("engine") == "Ensemble")


def test_predict_bad_types():
    s = svc()
    st, _ = s.handle("POST", "/predict", body={"history_length": "lots"})
    check("predict bad history_length -> 400", st == 400)
    st, _ = s.handle("POST", "/predict", body={"engine": 123})
    check("predict non-string engine -> 400", st == 400)


def test_record_valid():
    s = svc()
    st, body = s.handle("POST", "/record", body={
        "actual_number": 5, "predicted_number": 5, "profit_change": 50,
        "bets": [{"number": 5, "token_bet": 10}], "engine_used": "Ensemble",
    })
    check("POST /record -> 200", st == 200)
    check("record ok flag", body.get("ok") is True)
    check("record returns stats", body["stats"]["capital"] == 1075)
    check("backend stored record", s.backend.records[0]["actual_number"] == 5)


def test_record_missing_fields():
    s = svc()
    st, _ = s.handle("POST", "/record", body={"profit_change": 10})
    check("record missing actual_number -> 400", st == 400)
    st, _ = s.handle("POST", "/record", body={"actual_number": 5})
    check("record missing profit_change -> 400", st == 400)


def test_record_bad_types():
    s = svc()
    st, _ = s.handle("POST", "/record", body={"actual_number": "x", "profit_change": 1})
    check("record non-int actual -> 400", st == 400)
    st, _ = s.handle("POST", "/record",
                     body={"actual_number": 5, "profit_change": 1, "bets": "nope"})
    check("record bets not list -> 400", st == 400)


def test_record_float_profit():
    s = svc()
    st, body = s.handle("POST", "/record",
                        body={"actual_number": 5, "profit_change": 12.5})
    check("record accepts float profit", st == 200)


def test_unknown_route_and_method():
    s = svc()
    st, _ = s.handle("GET", "/nope")
    check("unknown path -> 404", st == 404)
    st, _ = s.handle("DELETE", "/stats")
    check("unsupported method -> 404", st == 404)


def test_non_dict_body():
    s = svc()
    st, _ = s.handle("POST", "/record", body=[1, 2, 3])
    check("non-dict body -> 400", st == 400)


# ---------------------------------------------------------------------- #
# End-to-end: stdlib http.server + requests
# ---------------------------------------------------------------------- #
def test_end_to_end_stdlib():
    try:
        import requests
    except Exception:
        print("  skip- e2e stdlib (requests not installed)")
        return
    service = svc()
    httpd = make_stdlib_server(service, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        r = requests.get(base + "/health", timeout=5)
        check("e2e /health 200", r.status_code == 200 and r.json()["status"] == "ok")
        r = requests.get(base + "/stats", timeout=5)
        check("e2e /stats capital", r.json()["capital"] == 1025)
        r = requests.get(base + "/history?limit=2", timeout=5)
        check("e2e /history limit 2", r.json()["count"] == 2)
        r = requests.post(base + "/predict", json={"engine": "Bayesian"}, timeout=5)
        check("e2e /predict engine echo", r.json()["engine"] == "Bayesian")
        r = requests.post(base + "/record",
                          json={"actual_number": 8, "profit_change": 100}, timeout=5)
        check("e2e /record ok", r.status_code == 200 and r.json()["ok"] is True)
        # malformed JSON body -> 400
        r = requests.post(base + "/record", data=b"{not json",
                          headers={"Content-Type": "application/json"}, timeout=5)
        check("e2e malformed JSON -> 400", r.status_code == 400)
        r = requests.get(base + "/nope", timeout=5)
        check("e2e unknown -> 404", r.status_code == 404)
    finally:
        httpd.shutdown()
        httpd.server_close()


def main():
    print("== PROMPT 19: REST API tests ==")
    test_root_and_health()
    test_trailing_slash()
    test_stats()
    test_history_default_and_limit()
    test_history_bad_limit()
    test_predict()
    test_predict_defaults_engine()
    test_predict_bad_types()
    test_record_valid()
    test_record_missing_fields()
    test_record_bad_types()
    test_record_float_profit()
    test_unknown_route_and_method()
    test_non_dict_body()
    test_end_to_end_stdlib()
    print(f"\n== {PASS} passed, {FAIL} failed ==")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
