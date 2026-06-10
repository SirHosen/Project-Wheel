# -*- coding: utf-8 -*-
"""PROMPT 20: pure-numpy core for wheel angle tracking.

Deliberately has NO OpenCV dependency so every bit of geometry/state logic is
unit-testable on a plain Python+numpy install. OpenCV only ever feeds (x, y)
marker positions into here (see vision/camera.py).

Honest scope: this estimates the wheel's CURRENT/REST angle and maps it to a
segment number. It is observation, not prediction.
"""
import math
from collections import deque

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dep elsewhere
    np = None


def normalize_angle(a):
    """Wrap an angle into [0, 360)."""
    return float(a) % 360.0


def angular_delta(a, b):
    """Shortest signed rotation a -> b, in [-180, 180]."""
    d = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    # Map the -180 boundary to +180 for stability when exactly opposite.
    return 180.0 if d == -180.0 else d


def angle_from_point(center, point):
    """Angle (deg, [0,360)) of `point` relative to `center`.

    Uses screen coordinates (y grows downward) but returns a MATH-style angle:
    0 deg = to the right, 90 deg = straight up, 180 = left, 270 = down.
    """
    cx, cy = center
    px, py = point
    dx = float(px) - float(cx)
    dy = float(cy) - float(py)  # invert y so "up" is positive
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return normalize_angle(math.degrees(math.atan2(dy, dx)))


def angle_to_segment_index(angle, n_segments=54, offset=0.0):
    """Map an angle to an equal-sized segment index in [0, n_segments)."""
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    seg = 360.0 / n_segments
    return int(normalize_angle(angle - offset) // seg) % n_segments


def angle_to_number(angle, sequence, offset=0.0):
    """Map an angle to a wheel number using the physical segment sequence.

    Returns dict {number, index, confidence}. `confidence` is how centered the
    angle sits inside its segment (1.0 dead-center, ->0 near a boundary), a
    proxy for how trustworthy the read is.
    """
    n = len(sequence)
    if n == 0:
        raise ValueError("sequence must be non-empty")
    seg = 360.0 / n
    pos = normalize_angle(angle - offset)
    idx = int(pos // seg) % n
    frac = (pos - idx * seg) / seg  # 0..1 within the segment
    confidence = 1.0 - 2.0 * abs(frac - 0.5)  # 1 at center, 0 at edges
    return {"number": sequence[idx], "index": idx, "confidence": round(confidence, 4)}


def mask_centroid(mask):
    """Centroid (x, y) of all truthy pixels in a 2D mask, or None if empty.

    x = column (horizontal), y = row (vertical) to match image conventions.
    """
    if np is None:
        raise RuntimeError("numpy is required for mask_centroid")
    arr = np.asarray(mask)
    ys, xs = np.nonzero(arr)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


class AngleTracker:
    """Stateful estimator of angular velocity + when the wheel has stopped.

    Feed (timestamp_seconds, angle_degrees) samples via `add`. Handles 0/360
    wraparound. A 'stop' is declared only AFTER real motion was seen and the
    last `stable_frames` speed readings are all below `stop_speed_deg_s` -- so
    the initial stationary state is not mistaken for a finished spin.
    """

    def __init__(self, stop_speed_deg_s=8.0, stable_frames=5):
        self.stop_speed_deg_s = float(stop_speed_deg_s)
        self.stable_frames = int(stable_frames)
        self._prev = None  # (t, angle)
        self._speeds = deque(maxlen=self.stable_frames)
        self.seen_motion = False
        self.rest_angle = None
        self.last_angle = None
        self.last_velocity = 0.0
        self.stopped = False
        self.n_samples = 0

    def add(self, t, angle):
        """Add one sample. Returns dict {angle, velocity, stopped}."""
        angle = normalize_angle(angle)
        self.n_samples += 1
        self.last_angle = angle
        velocity = 0.0
        if self._prev is not None:
            pt, pa = self._prev
            dt = float(t) - float(pt)
            if dt > 0:
                velocity = angular_delta(pa, angle) / dt
        self._prev = (t, angle)
        self.last_velocity = velocity
        speed = abs(velocity)
        if speed > self.stop_speed_deg_s:
            self.seen_motion = True
        self._speeds.append(speed)

        stopped = (
            self.seen_motion
            and len(self._speeds) >= self.stable_frames
            and all(s < self.stop_speed_deg_s for s in self._speeds)
        )
        if stopped and not self.stopped:
            self.rest_angle = angle  # latch the resting angle
        self.stopped = stopped
        return {"angle": angle, "velocity": velocity, "stopped": stopped}

    def result(self, sequence, offset=0.0):
        """Map the current rest (or last) angle to a wheel number."""
        angle = self.rest_angle if self.rest_angle is not None else self.last_angle
        if angle is None:
            return None
        out = angle_to_number(angle, sequence, offset=offset)
        out["angle"] = angle
        out["stopped"] = self.stopped
        return out

    def reset(self):
        self.__init__(self.stop_speed_deg_s, self.stable_frames)
