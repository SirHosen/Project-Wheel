# -*- coding: utf-8 -*-
"""Screen capture via mss. Grabs BGR frames for the result reader and degrades
gracefully when mss/numpy are not installed.
"""
try:
    import mss
    _MSS_ERR = None
except Exception as e:  # pragma: no cover - env dependent
    mss = None
    _MSS_ERR = e

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def is_available():
    return mss is not None and np is not None


def status_line():
    if mss is None:
        name = type(_MSS_ERR).__name__ if _MSS_ERR else "missing"
        return (f"capture UNAVAILABLE: mss ({name})"
                "  -> pip install -r requirements-vision.txt")
    if np is None:
        return "capture UNAVAILABLE: numpy missing  -> pip install -r requirements-vision.txt"
    return "screen capture available (mss)"


def list_monitors():
    """mss monitor geometry dicts. Index 0 = all screens, 1 = primary."""
    if not is_available():
        return []
    with mss.mss() as sct:
        return [dict(m) for m in sct.monitors]


def parse_region(text):
    """Parse 'x,y,w,h' into an mss region dict."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("region must be 'x,y,w,h'")
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError:
        raise ValueError("region values must be numbers: 'x,y,w,h'")
    if w <= 0 or h <= 0:
        raise ValueError("region width and height must be positive")
    return {"left": x, "top": y, "width": w, "height": h}


class ScreenSource:
    """Context manager yielding BGR frames from a screen region or monitor.

        with ScreenSource(monitor=1) as src:
            w, h = src.size
            frame = src.grab()
    """

    def __init__(self, region=None, monitor=1):
        if not is_available():
            raise RuntimeError(status_line())
        self.region = region
        self.monitor = int(monitor)
        self._sct = None
        self._grab_region = None

    def __enter__(self):
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if self.region is not None:
            self._grab_region = self.region
        else:
            idx = self.monitor if 0 <= self.monitor < len(monitors) else 1
            self._grab_region = monitors[idx]
        return self

    @property
    def size(self):
        r = self._grab_region
        return int(r["width"]), int(r["height"])

    def grab(self):
        """Return one BGR frame as a contiguous numpy array."""
        shot = self._sct.grab(self._grab_region)
        # mss returns BGRA; drop alpha -> BGR (what OpenCV expects).
        return np.ascontiguousarray(np.array(shot)[:, :, :3])

    def __exit__(self, exc_type, exc, tb):
        if self._sct is not None:
            self._sct.close()
            self._sct = None
        return False
