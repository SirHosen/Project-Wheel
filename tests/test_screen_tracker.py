# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Tests untuk parser CLI screen-capture (pure stdlib, tanpa mss/cv2)."""
from vision.screen import parse_region, parse_center, parse_hsv


def test_parse_region_ok():
    assert parse_region("100,80,640,480") == {
        "left": 100, "top": 80, "width": 640, "height": 480
    }
    # whitespace toleran
    assert parse_region(" 0, 0 , 50,50 ")["width"] == 50


def test_parse_region_bad():
    for bad in ["1,2,3", "1,2,3,4,5", "a,b,c,d", "10,10,0,100", "10,10,100,-1"]:
        try:
            parse_region(bad)
        except ValueError:
            continue
        raise AssertionError(f"region tidak valid lolos: {bad!r}")


def test_parse_center():
    assert parse_center(None) is None
    assert parse_center("") is None
    assert parse_center("320.5, 240") == (320.5, 240.0)
    try:
        parse_center("1,2,3")
    except ValueError:
        pass
    else:
        raise AssertionError("center 3-angka harusnya error")


def test_parse_hsv():
    assert parse_hsv(None) is None
    assert parse_hsv("  ") is None
    one = parse_hsv("40,80,80:85,255,255")
    assert one == [((40, 80, 80), (85, 255, 255))], one
    # dua rentang (merah membungkus hue)
    two = parse_hsv("0,120,70:10,255,255; 170,120,70:180,255,255")
    assert len(two) == 2 and two[1][0] == (170, 120, 70), two
    for bad in ["40,80:85,255,255", "40,80,80-85,255,255", "40,80,80:85,255"]:
        try:
            parse_hsv(bad)
        except ValueError:
            continue
        raise AssertionError(f"hsv tidak valid lolos: {bad!r}")


def main():
    test_parse_region_ok()
    test_parse_region_bad()
    test_parse_center()
    test_parse_hsv()
    print("ALL SCREEN TRACKER TESTS PASSED")


if __name__ == "__main__":
    main()
