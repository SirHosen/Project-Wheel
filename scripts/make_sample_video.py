#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate samples/spin2.mp4 -- a synthetic result-row video.

This is OPTIONAL. It writes a small video whose highlighted numbers follow the
exact sequence that tests/test_result_reader.py::test_replay_video expects, so
running this script turns that skipped test into a live end-to-end check.

    python scripts/make_sample_video.py

Requires opencv (requirements-vision.txt). If your OpenCV build cannot encode
mp4, the script says so and exits without writing a broken file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision import synthetic  # noqa: E402
from vision.result_reader import ResultReader, opencv_available  # noqa: E402

# Must match tests/test_result_reader.py::test_replay_video.
EXPECTED = [30, 1, 30, 8, 40, 15, 1, 10, 40, 30]


def main():
    if not opencv_available():
        print("[make-sample] opencv/numpy unavailable; "
              "pip install -r requirements-vision.txt")
        return 2
    import cv2
    w, h, fps = 1912, 974, 10.0
    reader = ResultReader(w, h, layout="landscape", fps=fps,
                          margin=26.0, stable_frames=3)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "samples")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "spin2.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print("[make-sample] This OpenCV build cannot encode mp4 (mp4v). "
              "No file written.")
        return 2
    frames = 0
    for frame in synthetic.session_frames(reader, EXPECTED):
        writer.write(frame)
        frames += 1
    writer.release()
    print(f"[make-sample] wrote {frames} frames -> {out_path}")
    print("[make-sample] Now `python run_tests.py` runs the replay end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
