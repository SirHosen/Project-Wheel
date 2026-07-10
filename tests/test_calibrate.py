# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Auto-calibration: recover known synthetic cell positions; degrade cleanly."""
from vision import calibrate as cal
from vision.result_reader import NUMS


def _draw_cells(w, h, centers, box):
    import numpy as np
    import cv2
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    bw, bh = box
    for (fx, fy) in centers:
        cx, cy = int(fx * w), int(fy * h)
        hw, hh = int(bw * w / 2), int(bh * h / 2)
        cv2.rectangle(frame, (cx - hw, cy - hh), (cx + hw, cy + hh),
                      (255, 255, 255), -1)
    return frame


def test_available_is_bool():
    assert isinstance(cal.available(), bool)
    print("OK calibrate.available() returns a bool")


def test_landscape_autocalibration():
    if not cal.available():
        print("SKIP landscape calibration: opencv/numpy unavailable")
        return
    w, h = 1912, 974
    xs = [0.10 + i * (0.80 / 8) for i in range(9)]
    centers = [(x, 0.5) for x in xs]
    frame = _draw_cells(w, h, centers, box=(0.05, 0.10))
    cells = cal.find_cells(frame, layout="landscape")
    assert cells is not None and len(cells) == 9
    assert [c[0] for c in cells] == NUMS  # mapped in reading order
    for (num, fx, fy), (tx, ty) in zip(cells, centers):
        assert abs(fx - tx) < 0.01, (num, fx, tx)
        assert abs(fy - ty) < 0.01, (num, fy, ty)
    print("OK landscape auto-calibration recovers the 9 cell centers")


def test_portrait_rowmajor_mapping():
    if not cal.available():
        print("SKIP portrait calibration: opencv/numpy unavailable")
        return
    w, h = 960, 960
    xs = [0.25, 0.50, 0.75]
    ys = [0.30, 0.50, 0.70]
    centers = [(x, y) for y in ys for x in xs]  # row-major truth
    frame = _draw_cells(w, h, centers, box=(0.12, 0.08))
    cells = cal.find_cells(frame, layout="portrait")
    assert cells is not None and [c[0] for c in cells] == NUMS
    for (num, fx, fy), (tx, ty) in zip(cells, centers):
        assert abs(fx - tx) < 0.02 and abs(fy - ty) < 0.02, (num, fx, fy, tx, ty)
    print("OK portrait auto-calibration maps a 3x3 grid row-major")


def test_returns_none_when_too_few_blobs():
    if not cal.available():
        print("SKIP too-few-blobs: opencv/numpy unavailable")
        return
    w, h = 1912, 974
    frame = _draw_cells(w, h, [(0.3, 0.5), (0.6, 0.5)], box=(0.05, 0.10))
    assert cal.find_cells(frame, layout="landscape") is None
    print("OK calibration refuses to guess with < 9 blobs")


if __name__ == "__main__":
    test_available_is_bool()
    test_landscape_autocalibration()
    test_portrait_rowmajor_mapping()
    test_returns_none_when_too_few_blobs()
    print("ALL CHECKS PASSED")
