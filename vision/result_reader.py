# -*- coding: utf-8 -*-
"""Reads the winning number off the game's result row by watching which number
cell lights up after a spin.

Method (robust, no OCR): for each number cell we track its brightness (HSV Value)
against a rolling baseline (40th percentile over a few seconds). A detection
fires only when ONE cell:
  1. spikes >= `margin` above its own baseline, AND
  2. beats the 2nd-brightest cell by >= `min_separation` (so a glow bleeding
     into a neighbouring cell can't cause a misread), AND
  3. stays the clear winner for `stable_frames` consecutive frames.
After firing we "re-arm" only once the highlight fades.

Cell geometry is CONFIGURABLE (see config.RESULT_LANDSCAPE / RESULT_PORTRAIT and
the `--snapshot` calibration tool) because the exact on-screen position depends
on your window size and layout. If detection misreads, run a calibration
snapshot and nudge the numbers in config.py -- no code changes needed.

Degrades gracefully: without opencv/numpy, opencv_available() is False and the
caller shows a friendly install hint instead of crashing.
"""
from collections import deque

try:
    import cv2
    import numpy as np
    _CV_ERR = None
except Exception as e:  # pragma: no cover - env dependent
    cv2 = None
    np = None
    _CV_ERR = e

from config import (RESULT_LANDSCAPE, RESULT_MIN_SEPARATION, RESULT_PORTRAIT)

NUMS = [1, 2, 5, 8, 10, 15, 20, 30, 40]


def opencv_available():
    return cv2 is not None and np is not None


def status_line():
    if opencv_available():
        return "result reader ready (opencv)"
    name = type(_CV_ERR).__name__ if _CV_ERR else "missing"
    return (f"result reader UNAVAILABLE: opencv/numpy ({name})"
            "  -> pip install -r requirements-vision.txt")


_OCR_STATE = {"checked": False, "available": False}


def ocr_available():
    """True only if opencv + pytesseract + the tesseract binary are ALL present.
    Cached after the first check. OCR is a purely OPTIONAL cross-check; the
    brightness-based reader works fully without it, so this returns False
    (never raises) when anything is missing.
    """
    if _OCR_STATE["checked"]:
        return _OCR_STATE["available"]
    ok = False
    if opencv_available():
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            ok = True
        except Exception:
            ok = False
    _OCR_STATE["checked"] = True
    _OCR_STATE["available"] = ok
    return ok


def _landscape_cells():
    """9 numbers in a horizontal result row, evenly spaced (from config)."""
    c = RESULT_LANDSCAPE
    fy = c["fy"]
    fx0, fx1 = c["fx_start"], c["fx_end"]
    step = (fx1 - fx0) / (len(NUMS) - 1)
    return [(NUMS[i], fx0 + i * step, fy) for i in range(len(NUMS))]


def _portrait_cells():
    """3x3 grid of numbers, row-major (from config)."""
    c = RESULT_PORTRAIT
    rows, cols = c["rows"], c["cols"]
    cells = []
    for r, fy in enumerate(rows):
        for col, fx in enumerate(cols):
            cells.append((NUMS[r * 3 + col], fx, fy))
    return cells


def layout_spec(layout):
    if layout == "portrait":
        return {"cells": _portrait_cells(), "bw": RESULT_PORTRAIT["bw"],
                "bh": RESULT_PORTRAIT["bh"]}
    return {"cells": _landscape_cells(), "bw": RESULT_LANDSCAPE["bw"],
            "bh": RESULT_LANDSCAPE["bh"]}


def detect_layout(frame_w, frame_h):
    """Landscape vs portrait from the aspect ratio."""
    ratio = float(frame_w) / float(frame_h) if frame_h else 1.0
    return "portrait" if ratio < 1.25 else "landscape"


class ResultReader:
    def __init__(self, frame_w, frame_h, layout=None, fps=15,
                 baseline_window_s=8.0, margin=26.0, stable_frames=3,
                 rearm_ratio=0.4, min_separation=RESULT_MIN_SEPARATION,
                 cells_override=None, ocr_verify=False, ocr_strict=False):
        self.w = int(frame_w)
        self.h = int(frame_h)
        self.layout = layout or detect_layout(frame_w, frame_h)
        spec = layout_spec(self.layout)
        self.bw = spec["bw"]
        self.bh = spec["bh"]
        cells = cells_override or spec["cells"]
        # Precompute an integer pixel box per cell.
        self.boxes = []
        for number, fx, fy in cells:
            cx, cy = int(fx * self.w), int(fy * self.h)
            hw = max(1, int(self.bw * self.w / 2))
            hh = max(1, int(self.bh * self.h / 2))
            self.boxes.append((number, cx - hw, cy - hh, cx + hw, cy + hh))
        self.margin = float(margin)
        self.min_separation = float(min_separation)
        self.stable_frames = int(stable_frames)
        self.rearm_ratio = float(rearm_ratio)
        maxlen = max(4, int(round(baseline_window_s * fps)))
        self.baselines = [deque(maxlen=maxlen) for _ in self.boxes]
        self._min_hist = max(1, maxlen // 4)
        self._armed = True
        self._cand = None
        self._cand_streak = 0
        self._spin_index = 0
        # OCR is an OPTIONAL digit cross-check (see ocr_available). ocr_strict
        # implies ocr_verify; strict mode vetoes any detection OCR can't confirm.
        self.ocr_strict = bool(ocr_strict)
        self.ocr_verify = bool(ocr_verify) or self.ocr_strict

    def _cell_brightness(self, frame, box):
        _, x0, y0, x1, y1 = box
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.w, x1)
        y1 = min(self.h, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = frame[y0:y1, x0:x1]
        v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
        return float(np.mean(v))

    def _ocr_read_number(self, frame, box):
        """Optional OCR cross-check: read the digits inside a cell. Returns the
        integer if it is one of the known numbers, else None. Never raises."""
        if not ocr_available():
            return None
        try:
            import pytesseract
            _, x0, y0, x1, y1 = box
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(self.w, x1), min(self.h, y1)
            if x1 <= x0 or y1 <= y0:
                return None
            roi = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=3.0, fy=3.0,
                              interpolation=cv2.INTER_CUBIC)
            _, th = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if float(np.mean(th)) > 127:   # ensure dark digits on light bg
                th = cv2.bitwise_not(th)
            txt = pytesseract.image_to_string(
                th, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            digits = "".join(ch for ch in txt if ch.isdigit())
            if not digits:
                return None
            val = int(digits)
            return val if val in NUMS else None
        except Exception:
            return None

    def update(self, frame, t=None):
        """Feed one BGR frame. Returns an event dict once per detected spin,
        else None."""
        spikes = []
        for i, box in enumerate(self.boxes):
            b = self._cell_brightness(frame, box)
            hist = self.baselines[i]
            if len(hist) >= self._min_hist:
                baseline = float(np.percentile(np.asarray(hist), 40))
            else:
                baseline = b
            spikes.append(b - baseline)
            hist.append(b)

        order = np.argsort(spikes)[::-1]
        best = int(order[0])
        best_spike = spikes[best]
        second_spike = spikes[int(order[1])] if len(order) > 1 else -1e9
        number = self.boxes[best][0]
        clear_winner = (best_spike - second_spike) >= self.min_separation

        if not self._armed:
            # Wait for the highlight to fade before detecting the next spin.
            if best_spike < self.rearm_ratio * self.margin:
                self._armed = True
                self._cand = None
                self._cand_streak = 0
            return None

        if best_spike >= self.margin and clear_winner:
            if self._cand == number:
                self._cand_streak += 1
            else:
                self._cand = number
                self._cand_streak = 1
            if self._cand_streak >= self.stable_frames:
                self._spin_index += 1
                self._armed = False
                self._cand_streak = 0
                ocr_number = None
                ocr_confirmed = None
                if self.ocr_verify and ocr_available():
                    ocr_number = self._ocr_read_number(frame, self.boxes[best])
                    ocr_confirmed = (ocr_number == number
                                     if ocr_number is not None else False)
                    if self.ocr_strict and ocr_confirmed is False:
                        # Strict mode: veto a detection OCR cannot confirm.
                        return None
                return {"number": number, "t": t, "spin_index": self._spin_index,
                        "spike": best_spike, "separation": best_spike - second_spike,
                        "layout": self.layout, "ocr": ocr_number,
                        "ocr_confirmed": ocr_confirmed}
        else:
            self._cand = None
            self._cand_streak = 0
        return None

    def annotate(self, frame):
        """Return a copy of the frame with each cell box + number drawn on it.
        Used by the `--snapshot` calibration tool so you can visually check
        that the boxes sit on the real number cells."""
        out = frame.copy()
        for number, x0, y0, x1, y1 in self.boxes:
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(out, str(number), (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(out, f"layout={self.layout}  {self.w}x{self.h}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return out
