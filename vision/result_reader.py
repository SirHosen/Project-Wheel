# -*- coding: utf-8 -*-
"""Reads the winning number off the game's result display by watching which
cell lights up.

Method (robust, no OCR): for each number cell we track its brightness (HSV Value)
against a rolling baseline (40th percentile over a few seconds). When one cell
spikes `margin` above its baseline and stays the brightest for `stable_frames`
consecutive frames, we emit that number once, then "re-arm" only after the
highlight fades. This matched a real landscape capture 10/10 in testing.

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

NUMS = [1, 2, 5, 8, 10, 15, 20, 30, 40]


def opencv_available():
    return cv2 is not None and np is not None


def status_line():
    if opencv_available():
        return "result reader ready (opencv)"
    name = type(_CV_ERR).__name__ if _CV_ERR else "missing"
    return (f"result reader UNAVAILABLE: opencv/numpy ({name})"
            "  -> pip install -r requirements-vision.txt")


def _landscape_cells():
    """9 numbers in a horizontal result bar, evenly spaced."""
    fy = 0.8573
    fx0, fx1 = 0.2730, 0.7416
    step = (fx1 - fx0) / (len(NUMS) - 1)
    return [(NUMS[i], fx0 + i * step, fy) for i in range(len(NUMS))]


def _portrait_cells():
    """3x3 grid of numbers (row-major)."""
    rows = [0.5708, 0.6750, 0.7833]
    cols = [0.4010, 0.5000, 0.5990]
    cells = []
    for r, fy in enumerate(rows):
        for c, fx in enumerate(cols):
            cells.append((NUMS[r * 3 + c], fx, fy))
    return cells


LAYOUTS = {
    "landscape": {"cells": _landscape_cells(), "bw": 0.0145, "bh": 0.0285},
    "portrait": {"cells": _portrait_cells(), "bw": 0.0271, "bh": 0.0271},
}


def detect_layout(frame_w, frame_h):
    """Landscape vs portrait from the aspect ratio."""
    ratio = float(frame_w) / float(frame_h) if frame_h else 1.0
    return "portrait" if ratio < 1.25 else "landscape"


class ResultReader:
    def __init__(self, frame_w, frame_h, layout=None, fps=15,
                 baseline_window_s=8.0, margin=26.0, stable_frames=3,
                 rearm_ratio=0.4, cells_override=None):
        self.w = int(frame_w)
        self.h = int(frame_h)
        self.layout = layout or detect_layout(frame_w, frame_h)
        spec = LAYOUTS[self.layout]
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
        self.stable_frames = int(stable_frames)
        self.rearm_ratio = float(rearm_ratio)
        maxlen = max(4, int(round(baseline_window_s * fps)))
        self.baselines = [deque(maxlen=maxlen) for _ in self.boxes]
        self._min_hist = max(1, maxlen // 4)
        self._armed = True
        self._cand = None
        self._cand_streak = 0
        self._spin_index = 0

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

        best = int(np.argmax(spikes))
        best_spike = spikes[best]
        number = self.boxes[best][0]

        if not self._armed:
            # Wait for the highlight to fade before detecting the next spin.
            if best_spike < self.rearm_ratio * self.margin:
                self._armed = True
                self._cand = None
                self._cand_streak = 0
            return None

        if best_spike >= self.margin:
            if self._cand == number:
                self._cand_streak += 1
            else:
                self._cand = number
                self._cand_streak = 1
            if self._cand_streak >= self.stable_frames:
                self._spin_index += 1
                self._armed = False
                self._cand_streak = 0
                return {"number": number, "t": t, "spin_index": self._spin_index,
                        "spike": best_spike, "layout": self.layout}
        else:
            self._cand = None
            self._cand_streak = 0
        return None
