# -*- coding: utf-8 -*-
"""PROMPT 20: OpenCV capture layer for wheel tracking (OPTIONAL).

This is the only module that touches OpenCV. It is import-safe even when cv2 is
NOT installed: `opencv_available()` tells you, and the camera class raises a
clear, friendly error instead of crashing the whole app.

The heavy lifting (angle math, stop detection, number mapping) lives in
vision/wheel_tracker.py as pure numpy and is fully unit-tested. Here we only:
  1. turn a BGR frame + HSV color range into a marker centroid, then an angle,
  2. drive an AngleTracker across frames from a camera/video.
"""
import time

from vision.wheel_tracker import (
    AngleTracker,
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


class CameraWheelTracker:
    """Drive an AngleTracker from a webcam or video file.

    Construction never touches the camera. Call `is_available()` first; `run()`
    raises a clear RuntimeError if cv2 is missing or the source can't be opened.
    """

    def __init__(self, source=0, center=None, hsv_ranges=None,
                 stop_speed_deg_s=8.0, stable_frames=6, sequence=None):
        self.source = source
        self.center = center
        self.hsv_ranges = hsv_ranges or DEFAULT_HSV_RANGES
        self.tracker = AngleTracker(stop_speed_deg_s, stable_frames)
        if sequence is None:
            try:
                from config import settings
                sequence = settings.SPINWHEEL_SEQUENCE
            except Exception:
                sequence = []
        self.sequence = sequence

    @staticmethod
    def is_available():
        return opencv_available()

    def run(self, duration=None, max_frames=None, show=False, on_update=None):
        """Capture frames and track the wheel until it stops / time runs out.

        Returns the final result dict from AngleTracker.result(). Raises
        RuntimeError (never a bare cv2 crash) on missing OpenCV / bad source.
        """
        if cv2 is None:
            raise RuntimeError(opencv_status())
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video source: {self.source!r}")
        start = time.time()
        frames = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                if self.center is None:
                    self.center = detect_wheel_center(frame)
                angle, centroid = frame_marker_angle(
                    frame, self.center, self.hsv_ranges
                )
                if angle is not None:
                    state = self.tracker.add(time.time(), angle)
                    if on_update:
                        on_update(state, centroid)
                    if show:
                        self._draw(frame, centroid, state)
                    if state["stopped"]:
                        break
                if show:
                    cv2.imshow("Wheel Tracker (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if duration is not None and (time.time() - start) >= duration:
                    break
                if max_frames is not None and frames >= max_frames:
                    break
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
        return self.tracker.result(self.sequence)

    def _draw(self, frame, centroid, state):  # pragma: no cover - visual only
        if self.center is not None:
            cx, cy = int(self.center[0]), int(self.center[1])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        if centroid is not None:
            px, py = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
            if self.center is not None:
                cv2.line(frame, (cx, cy), (px, py), (0, 255, 255), 2)
        label = f"v={state['velocity']:.0f}deg/s {'STOP' if state['stopped'] else ''}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
