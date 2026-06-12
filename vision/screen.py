# -*- coding: utf-8 -*-
"""Screen-capture wheel tracker (alternatif kamera).

Kalau roda muncul DI LAYAR (game online, video, animasi), kita nggak butuh
webcam: tangkap sepetak layar lalu jalankan pipeline yang SAMA seperti kamera
(HSV marker -> sudut -> segmen -> angka). Modul ini import-safe walau `mss`
atau `cv2` belum terpasang -- `mss_available()` / `mss_status()` kasih tahu, dan
`ScreenWheelTracker.run()` melempar RuntimeError yang ramah, bukan crash.

FRAMING TETAP JUJUR: ini MENGAMATI roda (sudut berhenti -> angka), BUKAN meramal
spin berikutnya. Sama seperti kamera, ia butuh sebuah MARKER warna yang ikut
berputar di dalam region (defaultnya hijau; atur via --hsv kalau beda).
"""
import time

# Reuse the camera pipeline (angle math, HSV mask, center detection, drawing).
import vision.camera as _cam
from vision.wheel_tracker import AngleTracker

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dep elsewhere
    np = None

try:
    import mss as _mss
    _MSS_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on environment
    _mss = None
    _MSS_IMPORT_ERROR = e


def mss_available():
    """True if the `mss` screen-capture library imported successfully."""
    return _mss is not None


def mss_status():
    """Human-readable status line for diagnostics / CLI."""
    if _mss is not None:
        return "mss (screen capture) available"
    return (
        "mss NOT available ("
        f"{type(_MSS_IMPORT_ERROR).__name__ if _MSS_IMPORT_ERROR else 'missing'}"
        "). Install with: pip install -r requirements-vision.txt"
    )


# --------------------------------------------------------------------------- #
# Pure-stdlib CLI parsers (fully unit-testable without mss/cv2/numpy).
# --------------------------------------------------------------------------- #
def parse_region(text):
    """\"x,y,w,h\" -> mss region dict {left, top, width, height}.

    Raises ValueError on bad shape or non-positive width/height.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("region harus 'x,y,w,h' (4 angka, contoh: 100,80,640,640)")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise ValueError("region harus berisi bilangan bulat: 'x,y,w,h'")
    if w <= 0 or h <= 0:
        raise ValueError("lebar & tinggi region harus > 0")
    return {"left": x, "top": y, "width": w, "height": h}


def parse_center(text):
    """\"x,y\" -> (float, float) pusat roda RELATIF terhadap region. None if empty."""
    if text is None or str(text).strip() == "":
        return None
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError("center harus 'x,y' (2 angka)")
    return (float(parts[0]), float(parts[1]))


def parse_hsv(text):
    """Parse HSV ranges into [((lo_h,lo_s,lo_v),(hi_h,hi_s,hi_v)), ...].

    Format: \"loH,loS,loV:hiH,hiS,hiV\" -- pisahkan beberapa range dengan ';'
    (berguna untuk merah yang membungkus hue 0/180). None/empty -> None (pakai
    DEFAULT_HSV_RANGES dari vision.camera).
    """
    if text is None or str(text).strip() == "":
        return None
    ranges = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        lo_hi = chunk.split(":")
        if len(lo_hi) != 2:
            raise ValueError("hsv harus 'loH,loS,loV:hiH,hiS,hiV' (opsional ';' antar range)")
        lo = tuple(int(v.strip()) for v in lo_hi[0].split(","))
        hi = tuple(int(v.strip()) for v in lo_hi[1].split(","))
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError("tiap batas HSV butuh tepat 3 angka (H,S,V)")
        ranges.append((lo, hi))
    if not ranges:
        return None
    return ranges


class ScreenWheelTracker:
    """Drive an AngleTracker from a region of the screen via `mss`.

    Construction never touches the screen. Call `is_available()` first; `run()`
    raises a clear RuntimeError if mss/cv2 is missing.

    Args:
        region: mss dict {left, top, width, height}, or None to grab a whole
            monitor (see `monitor`).
        monitor: mss monitor index when region is None (1 = primary, 0 = all).
        center: (x, y) wheel center RELATIVE to the captured region; auto-
            detected once if None.
        hsv_ranges: list of (lo, hi) HSV tuples; defaults to camera's green.
        throttle: optional seconds to sleep between grabs (caps CPU/FPS).
    """

    def __init__(self, region=None, monitor=1, center=None, hsv_ranges=None,
                 stop_speed_deg_s=8.0, stable_frames=6, sequence=None,
                 throttle=0.0):
        self.region = region
        self.monitor = int(monitor)
        self.center = center
        self.hsv_ranges = hsv_ranges or _cam.DEFAULT_HSV_RANGES
        self.throttle = float(throttle)
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
        """True only if BOTH screen capture (mss) and OpenCV are usable."""
        return mss_available() and _cam.opencv_available()

    @staticmethod
    def list_monitors():
        """Return mss' monitor list (index 0 = virtual 'all', 1+ = each screen)."""
        if _mss is None:
            raise RuntimeError(mss_status())
        with _mss.mss() as sct:
            return list(sct.monitors)

    def _grab_bgr(self, sct, region):
        """Grab one frame and return a contiguous uint8 BGR array."""
        raw = sct.grab(region)
        # mss returns BGRA; drop alpha -> BGR (what vision.camera expects).
        return np.ascontiguousarray(np.array(raw)[:, :, :3])

    def run(self, duration=None, max_frames=None, show=False, on_update=None):
        """Capture screen frames and track the wheel until it stops / time runs out.

        Returns the final result dict from AngleTracker.result(). Raises a clear
        RuntimeError (never a bare crash) when mss or OpenCV is unavailable.
        """
        if not mss_available():
            raise RuntimeError(mss_status())
        if _cam.cv2 is None:
            raise RuntimeError(_cam.opencv_status())
        cv2 = _cam.cv2
        start = time.time()
        frames = 0
        try:
            with _mss.mss() as sct:
                region = self.region
                if region is None:
                    mons = sct.monitors
                    idx = self.monitor if 0 <= self.monitor < len(mons) else 1
                    region = mons[idx]
                while True:
                    frame = self._grab_bgr(sct, region)
                    frames += 1
                    if self.center is None:
                        self.center = _cam.detect_wheel_center(frame)
                    angle, centroid = _cam.frame_marker_angle(
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
                        cv2.imshow("Screen Wheel Tracker (q to quit)", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    if duration is not None and (time.time() - start) >= duration:
                        break
                    if max_frames is not None and frames >= max_frames:
                        break
                    if self.throttle > 0:
                        time.sleep(self.throttle)
        finally:
            if show:
                _cam.cv2.destroyAllWindows()
        return self.tracker.result(self.sequence)

    def _draw(self, frame, centroid, state):  # pragma: no cover - visual only
        cv2 = _cam.cv2
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
