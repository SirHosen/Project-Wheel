# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""End-to-end vision regression using deterministic synthetic frames (no real
video or codec needed). Guards the brightness/separation/stability detection.
"""
from vision import synthetic
from vision.result_reader import ResultReader, opencv_available


def test_landscape_replay_matches_sequence():
    if not opencv_available():
        print("SKIP landscape replay: opencv/numpy unavailable")
        return
    reader = ResultReader(1912, 974, layout="landscape", fps=10,
                          margin=26.0, stable_frames=3)
    seq = [30, 1, 40, 8, 15, 1, 10, 40, 30, 2]
    detected = synthetic.replay(reader, seq)
    assert detected == seq, f"got {detected} expected {seq}"
    print(f"OK synthetic landscape replay: {detected}")


def test_portrait_replay_matches_sequence():
    if not opencv_available():
        print("SKIP portrait replay: opencv/numpy unavailable")
        return
    reader = ResultReader(960, 960, layout="portrait", fps=10,
                          margin=26.0, stable_frames=3)
    seq = [1, 5, 40, 20, 8]
    detected = synthetic.replay(reader, seq)
    assert detected == seq, f"got {detected} expected {seq}"
    print(f"OK synthetic portrait replay: {detected}")


def test_no_false_detections_when_idle():
    if not opencv_available():
        print("SKIP idle replay: opencv/numpy unavailable")
        return
    reader = ResultReader(1912, 974, layout="landscape", fps=10,
                          margin=26.0, stable_frames=3)
    detected = synthetic.replay(reader, [], warmup=60)
    assert detected == [], detected
    print("OK no false detections on an idle screen")


if __name__ == "__main__":
    test_landscape_replay_matches_sequence()
    test_portrait_replay_matches_sequence()
    test_no_false_detections_when_idle()
    print("ALL CHECKS PASSED")
