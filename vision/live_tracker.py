# -*- coding: utf-8 -*-
"""PROMPT 21: LIVE screen-capture wheel tracker for the integrated GUI panel.

Unlike vision/screen.py's ScreenWheelTracker.run() -- which observes EXACTLY ONE
spin and then returns -- this drives a long-running BACKGROUND loop suitable for
a live preview window inside the app:

  * keeps grabbing frames from a monitor/region (game runs on screen, e.g.
    Chrome -- so the source is ALWAYS the screen; webcam is removed),
  * emits every frame to `on_frame(rgb, state)` so the GUI can show a live feed,
  * fires `on_spin(result)` each time the wheel comes to rest, then RE-ARMS the
    AngleTracker so the NEXT spin is detected too (multi-spin session),
  * stops cleanly via stop().

Import-safe without mss/cv2 (is_available()/status_line() report the truth). It
reuses the SAME, unit-tested pipeline as the camera/screen modules
(HSV marker -> angle -> stop detection -> segment number).

HONEST FRAMING (unchanged): this OBSERVES a wheel (rest angle -> segment number).
It does NOT predict the next spin.
"""
import threading
import time

# Reuse the camera pipeline (angle math, HSV mask, center detection) + tracker.
import vision.camera as _cam
from vision.screen import mss_available, mss_status
from vision.wheel_tracker import AngleTracker

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dep elsewhere
    np = None

try:
    import mss as _mss
except Exception:  # pragma: no cover - depends on environment
    _mss = None


class LiveScreenTracker:
    """Continuous, multi-spin screen tracker that runs on a background thread.

    Construction never touches the screen. Call is_available() first; start()
    spawns a daemon thread, stop() joins it. All callbacks run ON THE WORKER
    THREAD -- they MUST be cheap and thread-safe (e.g. push into a queue/slot
    that the GUI main thread drains). Do NOT call Tkinter directly from them.

    Args:
        monitor: mss monitor index when `region` is None (1 = primary, 0 = all).
        region: mss dict {left, top, width, height}, or None for a whole monitor.
        center: (x, y) wheel center RELATIVE to the region; auto-detected if None.
        hsv_ranges: list of (lo, hi) HSV tuples; defaults to camera's green.
        fps: capture/preview frame-rate cap (CPU saver).
        on_frame: callback(rgb_uint8_HxWx3, state_or_None) per frame.
        on_spin: callback(result_dict) when a spin comes to rest.
        on_status: callback({"error": str}) on fatal capture errors.
    """

    def __init__(self, monitor=1, region=None, center=None, hsv_ranges=None,
                 stop_speed_deg_s=8.0, stable_frames=6, sequence=None,
                 fps=15, on_frame=None, on_spin=None, on_status=None):
        self.monitor = int(monitor)
        self.region = region
        self.center = center
        self.hsv_ranges = hsv_ranges or _cam.DEFAULT_HSV_RANGES
        self.stop_speed_deg_s = float(stop_speed_deg_s)
        self.stable_frames = int(stable_frames)
        if sequence is None:
            try:
                from config import settings
                sequence = settings.SPINWHEEL_SEQUENCE
            except Exception:
                sequence = []
        self.sequence = sequence
        self.fps = max(1, int(fps))
        self.on_frame = on_frame
        self.on_spin = on_spin
        self.on_status = on_status

        self._tracker = AngleTracker(self.stop_speed_deg_s, self.stable_frames)
        self._stop = threading.Event()
        self._thread = None
        # Public, read-only-ish counters for the UI.
        self.spins = 0
        self.last_result = None

    # ------------------------------------------------------------------
    # Availability helpers (import-safe; never touch the screen).
    # ------------------------------------------------------------------
    @staticmethod
    def is_available():
        """True only if BOTH screen capture (mss) and OpenCV are usable."""
        return mss_available() and _cam.opencv_available()

    @staticmethod
    def status_line():
        """Combined diagnostic line for both dependencies."""
        return mss_status() + "  |  " + _cam.opencv_status()

    @staticmethod
    def list_monitors():
        """mss monitor list (index 0 = virtual 'all', 1+ = each screen)."""
        if _mss is None:
            raise RuntimeError(mss_status())
        with _mss.mss() as sct:
            return list(sct.monitors)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Spawn the capture thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._tracker.reset()
        self.spins = 0
        self._thread = threading.Thread(
            target=self._run, name="LiveScreenTracker", daemon=True
        )
        self._thread.start()

    def stop(self):
        """Signal the loop to end and join the thread (safe to call twice)."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _grab_bgr(self, sct, region):
        """Grab one frame and return a contiguous uint8 BGR array."""
        raw = sct.grab(region)
        # mss returns BGRA; drop alpha -> BGR (what vision.camera expects).
        return np.ascontiguousarray(np.array(raw)[:, :, :3])

    def _draw_overlay(self, cv2, frame, centroid, state):
        """Annotate the frame in-place: center dot, marker dot+line, status text."""
        try:
            if self.center is not None:
                cx, cy = int(self.center[0]), int(self.center[1])
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            if centroid is not None:
                px, py = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
                if self.center is not None:
                    cv2.line(frame, (cx, cy), (px, py), (0, 255, 255), 2)
            v = float(state["velocity"]) if state else 0.0
            tag = "STOP" if (state and state.get("stopped")) else ""
            label = f"v={v:.0f}deg/s  spins={self.spins}  {tag}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        except Exception:
            pass

    def _run(self):
        if not self.is_available():
            if self.on_status:
                self.on_status({"error": self.status_line()})
            return
        cv2 = _cam.cv2
        interval = 1.0 / self.fps
        try:
            with _mss.mss() as sct:
                region = self.region
                if region is None:
                    mons = sct.monitors
                    idx = self.monitor if 0 <= self.monitor < len(mons) else 1
                    region = mons[idx]
                while not self._stop.is_set():
                    t0 = time.time()
                    frame = self._grab_bgr(sct, region)
                    if self.center is None:
                        self.center = _cam.detect_wheel_center(frame)
                    angle, centroid = _cam.frame_marker_angle(
                        frame, self.center, self.hsv_ranges
                    )
                    state = None
                    if angle is not None:
                        state = self._tracker.add(time.time(), angle)
                    self._draw_overlay(cv2, frame, centroid, state)
                    if self.on_frame is not None:
                        # BGR -> RGB view; consumer copies if it needs to keep it.
                        self.on_frame(frame[:, :, ::-1], state)
                    if state is not None and state.get("stopped"):
                        result = self._tracker.result(self.sequence)
                        if result and result.get("number") is not None:
                            self.spins += 1
                            self.last_result = result
                            if self.on_spin is not None:
                                self.on_spin(result)
                        # RE-ARM: require fresh motion before the next 'stop'.
                        self._tracker.reset()
                    dt = time.time() - t0
                    if dt < interval and not self._stop.is_set():
                        time.sleep(interval - dt)
        except Exception as e:  # pragma: no cover - hardware/runtime dependent
            if self.on_status:
                self.on_status({"error": f"{type(e).__name__}: {e}"})
