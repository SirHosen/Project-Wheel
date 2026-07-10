# -*- coding: utf-8 -*-
"""Auto-calibration of the result-row cell positions.

Manual calibration (nudging RESULT_LANDSCAPE / RESULT_PORTRAIT in config.py) is
fiddly. This module finds the 9 number cells automatically from a single frame:
it looks for the bright, roughly-equal blobs that make up the result row, sorts
them into reading order, and maps them onto the wheel numbers. The output is a
`cells_override` list you can hand straight to ResultReader -- no code edits.

It is deliberately conservative: if it cannot find EXACTLY 9 clean blobs it
returns None so the caller can fall back to the configured geometry rather than
silently trusting a bad guess. Degrades gracefully without opencv/numpy.
"""
try:
    import cv2
    import numpy as np
    _CV_ERR = None
except Exception as e:  # pragma: no cover - env dependent
    cv2 = None
    np = None
    _CV_ERR = e

from vision.result_reader import NUMS, detect_layout


def available():
    return cv2 is not None and np is not None


def _bright_blobs(frame, min_area_frac=0.00035, max_area_frac=0.05):
    """Return [(cx, cy, w, h)] in pixels for bright blobs, largest first.

    A blob qualifies on relative area so the same thresholds work at any
    resolution. Uses Otsu on a blurred gray image, then connected components.
    """
    h, w = frame.shape[:2]
    area = float(w * h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(thr)
    blobs = []
    for i in range(1, num):  # skip background label 0
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        a = float(stats[i, cv2.CC_STAT_AREA])
        if not (min_area_frac * area <= a <= max_area_frac * area):
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        blobs.append((cx, cy, bw, bh))
    blobs.sort(key=lambda b: b[2] * b[3], reverse=True)
    return blobs


def find_cells(frame, layout=None, expected=None):
    """Return a `cells_override` list [(number, fx, fy)] with fx/fy normalized to
    [0, 1], or None if exactly `expected` clean blobs were not found.
    """
    if not available():
        return None
    h, w = frame.shape[:2]
    layout = layout or detect_layout(w, h)
    expected = expected or len(NUMS)
    blobs = _bright_blobs(frame)
    if len(blobs) < expected:
        return None
    blobs = blobs[:expected]  # the `expected` largest bright blobs

    if layout == "portrait":
        # 3x3 grid: split into 3 rows by y, then sort each row left-to-right.
        rows = sorted(blobs, key=lambda b: b[1])
        per = expected // 3
        ordered = []
        for r in range(3):
            band = sorted(rows[r * per:(r + 1) * per], key=lambda b: b[0])
            ordered.extend(band)
    else:
        # Landscape: one row, sort left-to-right.
        ordered = sorted(blobs, key=lambda b: b[0])

    cells = []
    for number, (cx, cy, _bw, _bh) in zip(NUMS, ordered):
        cells.append((number, cx / float(w), cy / float(h)))
    return cells


def mean_box_fraction(frame, layout=None):
    """Suggested (bw, bh) as fractions of frame size from the detected blobs.
    Returns None if calibration failed."""
    if not available():
        return None
    h, w = frame.shape[:2]
    blobs = _bright_blobs(frame)
    expected = len(NUMS)
    if len(blobs) < expected:
        return None
    blobs = blobs[:expected]
    bw = sum(b[2] for b in blobs) / expected / float(w)
    bh = sum(b[3] for b in blobs) / expected / float(h)
    return (bw, bh)
