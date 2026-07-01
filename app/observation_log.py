# -*- coding: utf-8 -*-
"""Append-only CSV of observed results. Each row is one real i.i.d. draw from
the physical wheel, captured automatically by the screen reader. Used as the
training data for the AI and as evidence for the bias tracker.
"""
import csv
import os
from datetime import datetime, timezone

from config import OBSERVATIONS_PATH, VALID_NUMBERS

FIELDS = ["timestamp", "number", "spin_index", "spike", "layout"]


def log_result(result, path=None):
    """Append one ResultReader event dict to the CSV log."""
    path = path or OBSERVATIONS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "number": result.get("number"),
        "spin_index": result.get("spin_index"),
        "spike": result.get("spike"),
        "layout": result.get("layout"),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    return row


def load_numbers(path=None):
    """Return the recorded results as a clean list of ints (for training)."""
    path = path or OBSERVATIONS_PATH
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                n = int(r.get("number"))
            except (TypeError, ValueError):
                continue
            if n in VALID_NUMBERS:
                out.append(n)
    return out
