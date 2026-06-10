# -*- coding: utf-8 -*-
"""Headless tests for PROMPT 20 wheel-vision core.

Pure-numpy geometry/state logic is tested directly. The OpenCV layer is tested
with SYNTHETIC frames (no camera needed). Camera/video capture loops are not
exercised (no device), but they delegate to these tested helpers.

Run: python _test_wheel_tracker.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from vision.wheel_tracker import (  # noqa: E402
    AngleTracker,
    angle_from_point,
    angle_to_number,
    angle_to_segment_index,
    angular_delta,
    mask_centroid,
    normalize_angle,
)

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  - {name}")
    else:
        FAIL += 1
        print(f"  FAIL- {name}")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


SEQ = [1, 5, 2, 10, 1, 2, 20, 1, 8, 2, 1, 5, 1, 10, 2, 1, 5, 2, 1, 40, 2, 1, 8,
       1, 5, 1, 15, 2, 1, 10, 1, 5, 1, 20, 2, 1, 8, 2, 1, 2, 10, 1, 2, 5, 1, 2,
       30, 1, 8, 1, 5, 1, 2, 15]


def test_normalize_and_delta():
    check("normalize 370 -> 10", approx(normalize_angle(370), 10))
    check("normalize -10 -> 350", approx(normalize_angle(-10), 350))
    check("delta 350->10 = +20", approx(angular_delta(350, 10), 20))
    check("delta 10->350 = -20", approx(angular_delta(10, 350), -20))
    check("delta 0->180 = 180", approx(abs(angular_delta(0, 180)), 180))
    check("delta 0->90 = 90", approx(angular_delta(0, 90), 90))


def test_angle_from_point():
    c = (100, 100)
    check("right -> 0deg", approx(angle_from_point(c, (150, 100)), 0))
    check("up -> 90deg", approx(angle_from_point(c, (100, 50)), 90))
    check("left -> 180deg", approx(angle_from_point(c, (50, 100)), 180))
    check("down -> 270deg", approx(angle_from_point(c, (100, 150)), 270))
    check("center -> 0deg", approx(angle_from_point(c, (100, 100)), 0))


def test_segment_index():
    n = 54
    seg = 360.0 / n
    check("angle 0 -> idx 0", angle_to_segment_index(0, n) == 0)
    check("angle just under seg -> idx 0", angle_to_segment_index(seg - 0.1, n) == 0)
    check("angle just over seg -> idx 1", angle_to_segment_index(seg + 0.1, n) == 1)
    check("angle 359.9 -> idx 53", angle_to_segment_index(359.9, n) == 53)
    check("wrap 360 -> idx 0", angle_to_segment_index(360.0, n) == 0)
    # offset shifts the mapping
    check("offset shifts index", angle_to_segment_index(seg + 0.1, n, offset=seg) == 0)


def test_angle_to_number():
    n = len(SEQ)
    seg = 360.0 / n
    r0 = angle_to_number(seg * 0.5, SEQ)  # dead center of segment 0
    check("number maps to seq[0]", r0["number"] == SEQ[0] and r0["index"] == 0)
    check("center confidence ~1", r0["confidence"] > 0.98)
    redge = angle_to_number(0.05, SEQ)  # near segment boundary
    check("edge confidence low", redge["confidence"] < 0.2)
    r3 = angle_to_number(seg * 3 + seg * 0.5, SEQ)
    check("segment 3 -> seq[3]", r3["number"] == SEQ[3] and r3["index"] == 3)


def test_mask_centroid():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 6:8] = 1  # rows 2..4, cols 6..7 -> centroid (6.5, 3)
    c = mask_centroid(m)
    check("centroid x", approx(c[0], 6.5))
    check("centroid y", approx(c[1], 3.0))
    check("empty mask -> None", mask_centroid(np.zeros((4, 4))) is None)


def test_angle_tracker_spin_and_stop():
    tr = AngleTracker(stop_speed_deg_s=8.0, stable_frames=5)
    t = 0.0
    angle = 0.0
    vel = 400.0  # deg/s, decaying
    stopped_seen = False
    # decelerating spin over time
    for _ in range(200):
        t += 0.05
        angle = (angle + vel * 0.05)
        st = tr.add(t, angle % 360.0)
        if st["stopped"]:
            stopped_seen = True
            break
        vel *= 0.95  # exponential decay
    check("spin eventually stops", stopped_seen)
    check("rest_angle latched", tr.rest_angle is not None)
    res = tr.result(SEQ)
    check("result maps a valid number", res is not None and res["number"] in set(SEQ))
    check("result flagged stopped", res["stopped"] is True)


def test_angle_tracker_never_moves():
    tr = AngleTracker(stop_speed_deg_s=8.0, stable_frames=5)
    for i in range(10):
        st = tr.add(i * 0.1, 123.0)  # stationary the whole time
    check("stationary never counts as stopped", st["stopped"] is False)
    check("no motion seen", tr.seen_motion is False)


def test_angle_tracker_wrap_velocity():
    tr = AngleTracker()
    tr.add(0.0, 350.0)
    st = tr.add(0.1, 10.0)  # crossed 360 boundary: +20 deg in 0.1s = +200 deg/s
    check("wrap velocity positive ~200", approx(st["velocity"], 200.0, tol=1.0))


# ---------------------------------------------------------------------- #
# OpenCV layer with synthetic frames
# ---------------------------------------------------------------------- #
def test_opencv_layer():
    from vision.camera import (
        build_mask,
        frame_marker_angle,
        opencv_available,
    )
    check("opencv available in sandbox", opencv_available() is True)

    center = (100, 100)
    # green marker to the RIGHT of center -> angle ~0
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[95:105, 150:170] = (0, 255, 0)  # BGR green
    ang, cen = frame_marker_angle(frame, center)
    check("green-right detected", ang is not None)
    check("green-right angle ~0", ang is not None and (ang < 5 or ang > 355))

    # green marker ABOVE center -> angle ~90
    frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
    frame2[30:50, 95:105] = (0, 255, 0)
    ang2, _ = frame_marker_angle(frame2, center)
    check("green-above angle ~90", ang2 is not None and approx(ang2, 90, tol=8))

    # no marker -> None
    black = np.zeros((200, 200, 3), dtype=np.uint8)
    ang3, cen3 = frame_marker_angle(black, center)
    check("no marker -> None", ang3 is None and cen3 is None)

    # build_mask picks up the green pixels
    mask = build_mask(frame)
    check("mask has green pixels", int((mask > 0).sum()) > 100)


def main():
    print("== PROMPT 20: wheel-vision tests ==")
    test_normalize_and_delta()
    test_angle_from_point()
    test_segment_index()
    test_angle_to_number()
    test_mask_centroid()
    test_angle_tracker_spin_and_stop()
    test_angle_tracker_never_moves()
    test_angle_tracker_wrap_velocity()
    test_opencv_layer()
    print(f"\n== {PASS} passed, {FAIL} failed ==")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
