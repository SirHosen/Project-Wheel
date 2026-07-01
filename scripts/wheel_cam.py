# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Screen helper CLI (kept for compatibility with the old command).

    python scripts/wheel_cam.py --source screen --list-monitors
    python scripts/wheel_cam.py --source screen --monitor 1     # -> launches auto-watch

The game is played on screen (a video/canvas stream), so capture is always from
the screen -- no webcam, no Selenium. For actually watching results live, use
the small panel (scripts/auto_watch.py); this CLI mainly prints monitor geometry
so you can pick a --region or --monitor.

Observes results only -- it does NOT predict the next spin.
"""
import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description="Screen helper (screen capture only)")
    p.add_argument("--source", default="screen",
                   help="Only 'screen' is supported (webcam removed).")
    p.add_argument("--list-monitors", action="store_true",
                   help="Print monitor geometry and exit.")
    p.add_argument("--monitor", type=int, default=1,
                   help="Monitor index (1=primary, 0=all). Used to launch auto-watch.")
    p.add_argument("--region", default=None, help="Screen region 'x,y,w,h'.")
    p.add_argument("--no-ui", action="store_true", help="Launch auto-watch in text mode.")
    args = p.parse_args(argv)

    from vision import capture
    from vision.result_reader import status_line as rr_status

    src = str(args.source).strip().lower()
    if src not in ("screen", ""):
        print("[vision] Only screen capture is supported (webcam/Selenium not used).")
        print("[vision] Use: --source screen")
        return 2

    if args.list_monitors:
        if not capture.is_available():
            print("[vision] " + capture.status_line())
            return 2
        for i, m in enumerate(capture.list_monitors()):
            tag = " (all screens)" if i == 0 else ""
            print(f"[vision] monitor {i}{tag}: {m}")
        print("[vision] Use --region 'x,y,w,h' or --monitor N to pick an area.")
        return 0

    if not capture.is_available():
        print("[vision] " + capture.status_line())
        print("[vision] " + rr_status())
        print("[vision] Install: pip install -r requirements-vision.txt")
        return 2

    try:
        region = capture.parse_region(args.region) if args.region else None
    except ValueError as e:
        print(f"[vision] Invalid region: {e}")
        return 2

    print("[vision] Launching auto-watch (small live panel). Close it to stop.")
    from app.panel import main as panel_main
    return panel_main(region=region, monitor=args.monitor, no_ui=args.no_ui)


if __name__ == "__main__":
    raise SystemExit(main())
