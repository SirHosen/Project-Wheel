# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Screen reader: layout detection + (if a sample video is bundled) end-to-end
replay. Skips cleanly when opencv or the sample video is unavailable.
"""
from vision import result_reader as rr


def test_layout_detection():
    assert rr.detect_layout(1912, 974) == "landscape"
    assert rr.detect_layout(960, 960) == "portrait"
    print("OK layout detection from aspect ratio")


def test_cells_built():
    # 9 numbered boxes per layout, inside the frame.
    r = rr.ResultReader(1912, 974, layout="landscape") if rr.opencv_available() else None
    if r is None:
        print("SKIP cells: opencv/numpy unavailable")
        return
    assert len(r.boxes) == 9
    assert [b[0] for b in r.boxes] == rr.NUMS
    print("OK 9 cells built for landscape layout")


def test_replay_video():
    if not rr.opencv_available():
        print("SKIP replay: opencv/numpy unavailable")
        return
    import cv2
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "samples", "spin2.mp4")
    if not os.path.exists(path):
        print("SKIP replay: no sample video at samples/spin2.mp4")
        return
    expected = [30, 1, 30, 8, 40, 15, 1, 10, 40, 30]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / 10.0)))
    reader = rr.ResultReader(w, h, fps=10.0, margin=26.0, stable_frames=3)
    detected, idx = [], 0
    while cap.grab():
        if idx % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                ev = reader.update(fr, t=idx / fps)
                if ev:
                    detected.append(ev["number"])
        idx += 1
    cap.release()
    assert detected == expected, f"got {detected} expected {expected}"
    print(f"OK replay: {detected}")


if __name__ == "__main__":
    test_layout_detection()
    test_cells_built()
    test_replay_video()
    print("ALL CHECKS PASSED")
