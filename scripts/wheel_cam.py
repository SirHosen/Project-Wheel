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


def _print_result(result, round_no=None):
    head = "=== Hasil pengamatan ===" if round_no is None else f"=== Spin #{round_no} ==="
    print("\n" + head)
    print(f"  Sudut istirahat : {result.get('angle'):.1f} deg")
    print(f"  Segmen          : index {result.get('index')}")
    print(f"  Angka terbaca   : {result.get('number')}")
    print(f"  Keyakinan baca  : {result.get('confidence'):.2f} (1=tepat tengah segmen)")
    print(f"  Berhenti?       : {result.get('stopped')}")


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
    parser.add_argument("--rounds", type=int, default=1,
                        help="Observe N consecutive spins, logging each (default 1).")
    parser.add_argument("--no-log", action="store_true",
                        help="Do NOT append observations to the learning log.")
    args = parser.parse_args(argv)

    from vision.camera import CameraWheelTracker, opencv_status
    from vision.observation_log import OBSERVATIONS_PATH, log_observation

    if not CameraWheelTracker.is_available():
        print("[vision] " + opencv_status())
        print("[vision] Camera tracking unavailable -- the rest of the app still works.")
        return 2
    print("[vision] " + opencv_status())
    print("[vision] CATATAN: ini MENGAMATI roda (bukan meramal spin berikutnya).")
    if not args.no_log:
        print(f"[vision] Tiap spin dicatat ke: {OBSERVATIONS_PATH}")
        print("[vision] Lalu jalankan: python scripts/learn_from_vision.py (uji bias + update prior).")

    # "0" -> webcam index 0; otherwise treat as a file path.
    source = int(args.source) if args.source.isdigit() else args.source
    rounds = max(1, int(args.rounds))
    logged = 0
    last_rc = 1
    for r in range(1, rounds + 1):
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

        if not result or result.get("number") is None:
            print("[vision] No marker detected -- check lighting / marker color / center.")
            last_rc = 1
        else:
            _print_result(result, round_no=(r if rounds > 1 else None))
            if not result.get("stopped"):
                print("  (catatan: roda belum benar-benar berhenti; pakai sudut terakhir)")
            if not args.no_log:
                log_observation(result)
                logged += 1
            last_rc = 0

        if r < rounds:
            try:
                input(f"\n-- Spin #{r} selesai. Tekan Enter untuk spin #{r + 1} (Ctrl+C berhenti) --")
            except (EOFError, KeyboardInterrupt):
                print("\n[vision] Dihentikan.")
                break

    if logged:
        print(f"\n[vision] {logged} observasi dicatat. Analisis dengan: python scripts/learn_from_vision.py")
    return last_rc


if __name__ == "__main__":
    raise SystemExit(main())
