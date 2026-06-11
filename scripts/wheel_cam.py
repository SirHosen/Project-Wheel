# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""PROMPT 20: experimental camera wheel tracker CLI.

    python wheel_cam.py                  # webcam 0, live window
    python wheel_cam.py --source clip.mp4 --no-show
    python wheel_cam.py --duration 15

HONEST DISCLAIMER: this OBSERVES a spinning wheel (angle / when it stops / which
segment it rests on). It does NOT predict future spins -- a real spin is chaotic.
Use it to auto-record results, not to "beat" the wheel.

Degrades gracefully: if OpenCV isn't installed it prints how to install it and
exits cleanly instead of crashing.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental wheel camera tracker")
    parser.add_argument("--source", default="0",
                        help="Camera index (e.g. 0) or path to a video file.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: until wheel stops).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open a preview window (headless).")
    args = parser.parse_args(argv)

    from vision.camera import CameraWheelTracker, opencv_status

    if not CameraWheelTracker.is_available():
        print("[vision] " + opencv_status())
        print("[vision] Camera tracking unavailable -- the rest of the app still works.")
        return 2
    print("[vision] " + opencv_status())
    print("[vision] NOTE: this observes the wheel; it does NOT predict spins.")

    # "0" -> webcam index 0; otherwise treat as a file path.
    source = int(args.source) if args.source.isdigit() else args.source
    tracker = CameraWheelTracker(source=source)
    try:
        result = tracker.run(
            duration=args.duration,
            max_frames=args.max_frames,
            show=not args.no_show,
        )
    except RuntimeError as e:
        print(f"[vision] {e}")
        return 2

    if not result:
        print("[vision] No marker detected -- check lighting / marker color / center.")
        return 1
    print("\n=== Hasil pengamatan ===")
    print(f"  Sudut istirahat : {result.get('angle'):.1f} deg")
    print(f"  Segmen          : index {result.get('index')}")
    print(f"  Angka terbaca   : {result.get('number')}")
    print(f"  Keyakinan baca  : {result.get('confidence'):.2f} (1=tepat tengah segmen)")
    print(f"  Berhenti?       : {result.get('stopped')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
