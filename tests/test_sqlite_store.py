# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Headless tests for PROMPT 18 SQLite storage + auto-migration + JSON I/O.
Run: python test_sqlite_store.py  (stdlib only; Tracker test skipped if pandas missing)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sqlite_store import SqliteStore, empty_data, META_KEYS  # noqa: E402

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


def _tmp(suffix=".db"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.remove(path)  # we just want a unique unused path
    return path


def sample_data():
    return {
        "current_capital": 1234,
        "total_predictions": 3,
        "wins": 2,
        "losses": 1,
        "profit": 234,
        "history": [
            {
                "timestamp": "2026-06-01T10:00:00",
                "actual_number": 5,
                "predicted_number": 5,
                "profit_change": 100,
                "is_win": True,
                "bets": [{"number": 5, "token_bet": 10, "confidence": 0.4,
                          "ev_per_token": 0.1, "is_positive_ev": True, "support": 20}],
                "engine_used": "Ensemble",
                "mode": "conservative",
            },
            {
                "timestamp": "2026-06-01T10:01:00",
                "actual_number": 2,
                "predicted_number": None,
                "profit_change": -10,
                "is_win": False,
                "bets": [],
            },
            {
                "timestamp": "2026-06-01T10:02:00",
                "actual_number": 8,
                "predicted_number": 8,
                "profit_change": 144,
                "is_win": True,
                "bets": [],
                "extra_future_key": {"nested": [1, 2, 3]},
            },
        ],
    }


def test_empty_data_shape():
    d = empty_data()
    check("empty_data has all meta keys", all(k in d for k in META_KEYS))
    check("empty_data history is empty list", d["history"] == [])
    check("empty_data default capital 1000", d["current_capital"] == 1000)


def test_fresh_is_empty():
    p = _tmp()
    try:
        s = SqliteStore(p)
        check("fresh store is_empty", s.is_empty() is True)
        check("fresh store count 0", s.count() == 0)
        check("fresh load == empty_data", s.load() == empty_data())
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_save_load_roundtrip():
    p = _tmp()
    try:
        s = SqliteStore(p)
        data = sample_data()
        s.save_all(data)
        check("after save not empty", s.is_empty() is False)
        check("count == 3", s.count() == 3)
        loaded = s.load()
        check("meta capital preserved", loaded["current_capital"] == 1234)
        check("meta wins preserved", loaded["wins"] == 2)
        check("history length 3", len(loaded["history"]) == 3)
        check("order preserved (first ts)",
              loaded["history"][0]["timestamp"] == "2026-06-01T10:00:00")
        check("bets list preserved",
              loaded["history"][0]["bets"][0]["token_bet"] == 10)
        check("is_win bool True preserved",
              loaded["history"][0]["is_win"] is True)
        check("profit_change int preserved",
              loaded["history"][0]["profit_change"] == 100 and
              isinstance(loaded["history"][0]["profit_change"], int))
        check("predicted_number None preserved",
              loaded["history"][1]["predicted_number"] is None)
        check("engine_used preserved",
              loaded["history"][0]["engine_used"] == "Ensemble")
        check("mode preserved",
              loaded["history"][0]["mode"] == "conservative")
        check("missing engine_used absent (additive)",
              "engine_used" not in loaded["history"][1])
        check("extra future key round-trips",
              loaded["history"][2]["extra_future_key"] == {"nested": [1, 2, 3]})
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_save_all_replaces():
    p = _tmp()
    try:
        s = SqliteStore(p)
        s.save_all(sample_data())
        check("count 3 before replace", s.count() == 3)
        s.save_all(empty_data())
        check("count 0 after replace with empty", s.count() == 0)
        check("meta reset to defaults", s.load()["current_capital"] == 1000)
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_append_record():
    p = _tmp()
    try:
        s = SqliteStore(p)
        s.save_all(empty_data())
        rec = {"timestamp": "t", "actual_number": 1, "predicted_number": 1,
               "profit_change": 5, "is_win": True, "bets": []}
        s.append_record(rec, {"current_capital": 1005, "wins": 1,
                              "total_predictions": 1, "losses": 0, "profit": 5})
        loaded = s.load()
        check("append adds one row", len(loaded["history"]) == 1)
        check("append upserts meta", loaded["current_capital"] == 1005)
        check("append record fidelity", loaded["history"][0]["actual_number"] == 1)
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_migrate_from_json():
    p = _tmp()
    jp = _tmp(".json")
    try:
        with open(jp, "w") as f:
            json.dump(sample_data(), f)
        s = SqliteStore(p)
        check("db empty before migrate", s.is_empty() is True)
        ok = s.migrate_from_json(jp)
        check("migrate returns True", ok is True)
        check("db has 3 rows after migrate", s.count() == 3)
        check("migrated capital matches", s.load()["current_capital"] == 1234)
    finally:
        for f in (p, jp):
            if os.path.exists(f):
                os.remove(f)


def test_migrate_missing_and_bad():
    p = _tmp()
    try:
        s = SqliteStore(p)
        check("migrate missing file -> False",
              s.migrate_from_json("/nonexistent/nope.json") is False)
        # bad (non-dict) json
        jp = _tmp(".json")
        with open(jp, "w") as f:
            json.dump([1, 2, 3], f)
        check("migrate non-dict json -> False", s.migrate_from_json(jp) is False)
        if os.path.exists(jp):
            os.remove(jp)
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_import_replace():
    p = _tmp()
    jp = _tmp(".json")
    try:
        s = SqliteStore(p)
        s.save_all(sample_data())
        # import a smaller dataset in replace mode
        small = {"current_capital": 500, "total_predictions": 1, "wins": 0,
                 "losses": 1, "profit": -50,
                 "history": [{"timestamp": "x", "actual_number": 1,
                              "predicted_number": 2, "profit_change": -50,
                              "is_win": False, "bets": []}]}
        with open(jp, "w") as f:
            json.dump(small, f)
        result = s.import_json(jp, mode="replace")
        check("replace import count 1", len(result["history"]) == 1)
        check("replace import capital 500", result["current_capital"] == 500)
        check("db reflects replace", s.count() == 1)
    finally:
        for f in (p, jp):
            if os.path.exists(f):
                os.remove(f)


def test_import_append():
    p = _tmp()
    jp = _tmp(".json")
    try:
        s = SqliteStore(p)
        s.save_all(sample_data())  # 3 records, wins=2
        add = {"history": [{"timestamp": "y", "actual_number": 1,
                            "predicted_number": 1, "profit_change": 20,
                            "is_win": True, "bets": []}]}
        with open(jp, "w") as f:
            json.dump(add, f)
        result = s.import_json(jp, mode="append")
        check("append import count 4", len(result["history"]) == 4)
        check("append recomputes total_predictions", result["total_predictions"] == 4)
        check("append recomputes wins (2+1)", result["wins"] == 3)
        check("append recomputes losses", result["losses"] == 1)
        check("append recomputes profit (234+20)", result["profit"] == 254)
    finally:
        for f in (p, jp):
            if os.path.exists(f):
                os.remove(f)


def test_export_json_roundtrip():
    p = _tmp()
    jp = _tmp(".json")
    try:
        s = SqliteStore(p)
        s.save_all(sample_data())
        s.export_json(jp)
        check("export file exists", os.path.exists(jp))
        with open(jp) as f:
            exported = json.load(f)
        check("exported capital matches", exported["current_capital"] == 1234)
        check("exported history len 3", len(exported["history"]) == 3)
        # re-import into a clean store and compare
        p2 = _tmp()
        s2 = SqliteStore(p2)
        s2.import_json(jp, mode="replace")
        check("export->import preserves count", s2.count() == 3)
        if os.path.exists(p2):
            os.remove(p2)
    finally:
        for f in (p, jp):
            if os.path.exists(f):
                os.remove(f)


def test_persistence_reopen():
    p = _tmp()
    try:
        s = SqliteStore(p)
        s.save_all(sample_data())
        # reopen a brand new store object on the same file
        s2 = SqliteStore(p)
        check("reopened store sees data", s2.is_empty() is False)
        check("reopened count 3", s2.count() == 3)
        check("reopened capital", s2.load()["current_capital"] == 1234)
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_tracker_integration():
    """End-to-end via Tracker (needs pandas). Skipped gracefully if missing."""
    try:
        import pandas  # noqa: F401
    except Exception:
        print("  skip- Tracker integration (pandas not installed)")
        return
    from data.tracker import Tracker
    tmpdir = tempfile.mkdtemp()
    jp = os.path.join(tmpdir, "history.json")
    dbp = os.path.join(tmpdir, "history.db")
    # seed a legacy json
    with open(jp, "w") as f:
        json.dump(sample_data(), f)
    t = Tracker(history_file=jp, db_file=dbp)
    check("tracker auto-migrated capital", t.data["current_capital"] == 1234)
    check("tracker migrated history len 3", len(t.data["history"]) == 3)
    check("tracker created db file", os.path.exists(dbp))
    check("tracker kept json backup", os.path.exists(jp + ".migrated"))
    # record a new result -> persists to both db and json mirror
    t.record_result(5, 5, 50, bet_snapshot=[{"number": 5, "token_bet": 5}],
                    engine_used="Ensemble")
    t2 = Tracker(history_file=jp, db_file=dbp)
    check("reopen reads from sqlite (4 records)", len(t2.data["history"]) == 4)
    check("reopen sees updated capital", t2.data["current_capital"] == 1234 + 50)
    # reset clears db
    t2.reset_data()
    t3 = Tracker(history_file=jp, db_file=dbp)
    check("reset clears sqlite", len(t3.data["history"]) == 0)
    check("reset capital back to 1000", t3.data["current_capital"] == 1000)


def main():
    print("== PROMPT 18: SQLite storage tests ==")
    test_empty_data_shape()
    test_fresh_is_empty()
    test_save_load_roundtrip()
    test_save_all_replaces()
    test_append_record()
    test_migrate_from_json()
    test_migrate_missing_and_bad()
    test_import_replace()
    test_import_append()
    test_export_json_roundtrip()
    test_persistence_reopen()
    test_tracker_integration()
    print(f"\n== {PASS} passed, {FAIL} failed ==")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
