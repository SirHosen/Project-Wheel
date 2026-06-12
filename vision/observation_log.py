# -*- coding: utf-8 -*-
"""Log append-only observasi roda dari kamera + loader (vision learning loop).

Tiap spin yang diamati scripts/wheel_cam.py adalah draw i.i.d. NYATA dari roda
fisik. Kita simpan di sini supaya scripts/learn_from_vision.py bisa:
  * menguji apakah roda fisik menyimpang dari layout desain (chi-square),
  * melipat observasi itu ke posterior BayesianOptimalEngine sebagai evidence
    nyata tambahan -- TANPA mengotori statistik win-rate taruhan.

Ini observasi, bukan prediksi.
"""
import csv
import os
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBSERVATIONS_PATH = os.path.join(_ROOT, "reports", "vision_observations.csv")

FIELDS = ["timestamp", "number", "segment_index", "angle", "confidence", "stopped"]


def log_observation(result, path=None):
    """Append satu dict observasi (dari CameraWheelTracker.run) ke CSV log."""
    path = path or OBSERVATIONS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "number": result.get("number"),
        "segment_index": result.get("index"),
        "angle": round(float(result.get("angle", 0.0) or 0.0), 2),
        "confidence": round(float(result.get("confidence", 0.0) or 0.0), 4),
        "stopped": bool(result.get("stopped")),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    return row


def load_observations(path=None):
    """Baca semua observasi tercatat sebagai list dict (sudah di-typing)."""
    path = path or OBSERVATIONS_PATH
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw = r.get("number")
            try:
                num = int(raw) if raw not in (None, "", "None") else None
            except (TypeError, ValueError):
                num = None
            out.append({
                "timestamp": r.get("timestamp"),
                "number": num,
                "segment_index": r.get("segment_index"),
                "angle": float(r["angle"]) if r.get("angle") else None,
                "confidence": float(r["confidence"]) if r.get("confidence") else None,
                "stopped": (str(r.get("stopped")).lower() == "true"),
            })
    return out
