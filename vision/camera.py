# -*- coding: utf-8 -*-
"""PROMPT 20/21: OpenCV vision PIPELINE for wheel tracking (OPTIONAL).

This is the only module that touches OpenCV. It is import-safe even when cv2 is
NOT installed: `opencv_available()` tells you, and callers raise a clear,
friendly error instead of crashing the whole app.

The heavy lifting (angle math, stop detection, number mapping) lives in
vision/wheel_tracker.py as pure numpy and is fully unit-tested. Here we only:
  1. turn a BGR frame + HSV color range into a marker centroid, then an angle,
  2. find the wheel center.

PROMPT 21: the webcam driver (the old `CameraWheelTracker`) was REMOVED -- the
game is played on-screen (e.g. Chrome on the laptop), so capture is ALWAYS
screen-based. The live source now lives in vision/screen.py (one spin) and
vision/live_tracker.py (continuous, multi-spin, for the in-app LIVE panel). Both
reuse the SAME functions below, so the pipeline stays single-sourced and tested.
"""
from vision.wheel_tracker import (
    angle_from_point,
    mask_centroid,
)

try:
    import cv2
    _CV2_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on environment
    cv2 = None
    _CV2_IMPORT_ERROR = e

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# A reasonable default: a bright green marker stuck on the wheel rim.
DEFAULT_HSV_RANGES = [((40, 80, 80), (85, 255, 255))]


def opencv_available():
    """True if OpenCV (cv2) imported successfully."""
    return cv2 is not None


def opencv_status():
    """Human-readable status line for diagnostics / CLI."""
    if cv2 is not None:
        return f"OpenCV {cv2.__version__} available"
    return (
        "OpenCV NOT available ("
        f"{type(_CV2_IMPORT_ERROR).__name__ if _CV2_IMPORT_ERROR else 'missing'}"
        "). Install with: pip install -r requirements-vision.txt"
    )


def build_mask(frame_bgr, hsv_ranges=None):
    """Boolean-ish uint8 mask of pixels falling in any HSV range.

    Supports multiple ranges (e.g. red wraps around hue 0/180), OR'd together.
    """
    if cv2 is None:
        raise RuntimeError(opencv_status())
    ranges = hsv_ranges or DEFAULT_HSV_RANGES
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def frame_marker_angle(frame_bgr, center, hsv_ranges=None, min_pixels=15):
    """Detect the colored marker in a BGR frame and return its angle vs center.

    Returns (angle_degrees, centroid_xy) or (None, None) if the marker is not
    confidently visible (fewer than `min_pixels` matching pixels).
    """
    mask = build_mask(frame_bgr, hsv_ranges)
    if int((mask > 0).sum()) < int(min_pixels):
        return None, None
    centroid = mask_centroid(mask)
    if centroid is None:
        return None, None
    return angle_from_point(center, centroid), centroid


def detect_wheel_center(frame_bgr):
    """Best-effort wheel center via Hough circles; falls back to frame center."""
    if cv2 is None:
        raise RuntimeError(opencv_status())
    h, w = frame_bgr.shape[:2]
    fallback = (w / 2.0, h / 2.0)
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w),
            param1=100, param2=60, minRadius=int(min(h, w) * 0.2),
            maxRadius=int(min(h, w) * 0.55),
        )
        if circles is not None and len(circles) > 0:
            c = circles[0][0]
            return (float(c[0]), float(c[1]))
    except Exception:
        pass
    return fallback
