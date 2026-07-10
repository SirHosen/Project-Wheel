# -*- coding: utf-8 -*-
"""Synthetic result-row frames for testing the vision pipeline WITHOUT a real
screen capture or a bundled video.

We render exactly the cells that ResultReader is watching (its own boxes),
lighting up one number at a time in a chosen sequence. Feeding these frames back
into a ResultReader lets us regression-test detection deterministically -- if a
change ever breaks the brightness/separation/stability logic, the synthetic
replay stops matching the known sequence and the test fails.

Degrades gracefully without opencv/numpy (available() -> False).
"""
try:
    import cv2
    import numpy as np
    _CV_ERR = None
except Exception as e:  # pragma: no cover - env dependent
    cv2 = None
    np = None
    _CV_ERR = e


def available():
    return cv2 is not None and np is not None


def render_result_frame(reader, lit_number=None, bright=220, dark=12):
    """Render one BGR frame for `reader`'s geometry, with `lit_number`'s cell
    bright and every other cell dark. `lit_number=None` renders an idle frame
    (used for warmup and for the gaps between spins so the reader re-arms).
    """
    frame = np.full((reader.h, reader.w, 3), dark, dtype=np.uint8)
    for number, x0, y0, x1, y1 in reader.boxes:
        value = bright if number == lit_number else dark
        cv2.rectangle(frame, (x0, y0), (x1, y1), (value, value, value), -1)
    return frame


def session_frames(reader, sequence, warmup=25, lit_frames=6, gap_frames=12,
                   bright=220, dark=12):
    """Yield a full synthetic session: `warmup` idle frames to establish the
    brightness baseline, then for each number in `sequence` a burst of
    `lit_frames` lit frames followed by `gap_frames` idle frames.
    """
    for _ in range(warmup):
        yield render_result_frame(reader, None, bright, dark)
    for number in sequence:
        for _ in range(lit_frames):
            yield render_result_frame(reader, number, bright, dark)
        for _ in range(gap_frames):
            yield render_result_frame(reader, None, bright, dark)


def replay(reader, sequence, **kwargs):
    """Convenience: drive `reader` over a synthetic session and return the list
    of detected numbers. Should equal `sequence` for a healthy pipeline."""
    detected = []
    for frame in session_frames(reader, sequence, **kwargs):
        ev = reader.update(frame)
        if ev:
            detected.append(ev["number"])
    return detected
