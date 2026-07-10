# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Auto-watch: small always-on-top panel that reads results off the screen and
updates the Bayesian bias brain live. No manual input.

    python scripts/auto_watch.py --list-monitors
    python scripts/auto_watch.py --monitor 1
    python scripts/auto_watch.py --region 60,40,1912,974
    python scripts/auto_watch.py --monitor 1 --no-ui     # text mode
    python scripts/auto_watch.py --snapshot --monitor 1  # CALIBRATION snapshot

CALIBRATION: if the detected numbers look wrong, run --snapshot. It grabs one
frame, draws the green detection boxes on it, and saves runtime/calibration.png.
Open that image; if the green boxes are not sitting on the game's result-row
numbers, nudge RESULT_LANDSCAPE / RESULT_PORTRAIT in config.py and snapshot
again until they line up. Accurate boxes = accurate data = trustworthy stats.
"""
import argparse


def _snapshot(region, monitor):
    """Grab one frame, overlay detection boxes, save to runtime/calibration.png."""
    import os
    from config import RUNTIME_DIR
    from vision.capture import ScreenSource
    from vision.result_reader import ResultReader
    import cv2
    with ScreenSource(region=region, monitor=monitor) as src:
        w, h = src.size
        frame = src.grab()
    reader = ResultReader(w, h)
    annotated = reader.annotate(frame)
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    out = os.path.join(RUNTIME_DIR, "calibration.png")
    cv2.imwrite(out, annotated)
    print(f"[snapshot] layout={reader.layout}  frame={w}x{h}")
    print(f"[snapshot] saved -> {out}")
    print("[snapshot] Open it. If the green boxes are NOT on the result-row")
    print("[snapshot] numbers, edit RESULT_LANDSCAPE/RESULT_PORTRAIT in config.py")
    print("[snapshot] (fx_start = center of '1', fx_end = center of '40', fy = row height)")
    return 0


def _auto_calibrate(region, monitor):
    """Grab one frame, auto-detect the 9 result cells, and print a
    cells_override you can pass to ResultReader (or eyeball against config)."""
    import os
    from config import RUNTIME_DIR
    from vision.capture import ScreenSource
    from vision.result_reader import ResultReader, detect_layout
    from vision import calibrate
    import cv2
    with ScreenSource(region=region, monitor=monitor) as src:
        w, h = src.size
        frame = src.grab()
    layout = detect_layout(w, h)
    cells = calibrate.find_cells(frame, layout=layout)
    if cells is None:
        print("[auto-calibrate] Could not find 9 clean cells automatically.")
        print("[auto-calibrate] Fall back to --snapshot + manual config nudging.")
        return 2
    print(f"[auto-calibrate] layout={layout}  frame={w}x{h}")
    print("[auto-calibrate] Detected cells (number, fx, fy):")
    for number, fx, fy in cells:
        print(f"    ({number:>2}, {fx:.4f}, {fy:.4f})")
    box = calibrate.mean_box_fraction(frame, layout=layout)
    if box:
        print(f"[auto-calibrate] suggested bw={box[0]:.4f}  bh={box[1]:.4f}")
    # Save a snapshot annotated with the AUTO-detected boxes for a visual check.
    reader = ResultReader(w, h, layout=layout, cells_override=cells)
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    out = os.path.join(RUNTIME_DIR, "calibration.png")
    cv2.imwrite(out, reader.annotate(frame))
    print(f"[auto-calibrate] annotated snapshot -> {out}")
    print("[auto-calibrate] If the boxes sit on the numbers, you're calibrated.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Auto-watch screen reader + live panel")
    p.add_argument("--region", default=None, help="Screen region 'x,y,w,h'. Empty = whole monitor.")
    p.add_argument("--monitor", type=int, default=1, help="Monitor index (1=primary, 0=all).")
    p.add_argument("--no-ui", action="store_true", help="Text mode (no Tk panel).")
    p.add_argument("--no-log", action="store_true", help="Do not write to the results log.")
    p.add_argument("--list-monitors", action="store_true", help="Print monitors and exit.")
    p.add_argument("--snapshot", action="store_true",
                   help="Save an annotated calibration frame and exit.")
    p.add_argument("--auto-calibrate", action="store_true",
                   help="Auto-detect the 9 cells from one frame and print a "
                        "ready-to-paste cells_override; also saves a snapshot.")
    args = p.parse_args(argv)

    from vision import capture
    from vision.result_reader import opencv_available, status_line

    if args.list_monitors:
        if not capture.is_available():
            print("[auto-watch] " + capture.status_line())
            return 2
        for i, m in enumerate(capture.list_monitors()):
            tag = " (all screens)" if i == 0 else ""
            print(f"[auto-watch] monitor {i}{tag}: {m}")
        return 0

    if not (capture.is_available() and opencv_available()):
        print("[auto-watch] " + capture.status_line())
        print("[auto-watch] " + status_line())
        print("[auto-watch] Install: pip install -r requirements-vision.txt")
        return 2

    try:
        region = capture.parse_region(args.region) if args.region else None
    except ValueError as e:
        print(f"[auto-watch] Invalid region: {e}")
        return 2

    if args.snapshot:
        return _snapshot(region, args.monitor)

    if args.auto_calibrate:
        return _auto_calibrate(region, args.monitor)

    from app.panel import main as panel_main
    return panel_main(region=region, monitor=args.monitor, no_ui=args.no_ui,
                      do_log=not args.no_log)


if __name__ == "__main__":
    raise SystemExit(main())
