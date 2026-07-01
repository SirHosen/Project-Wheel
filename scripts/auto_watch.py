# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Auto-watch: small always-on-top panel that reads results off the screen and
updates the Bayesian bias brain live. No manual input.

    python scripts/auto_watch.py --list-monitors
    python scripts/auto_watch.py --monitor 1
    python scripts/auto_watch.py --region 60,40,1912,974
    python scripts/auto_watch.py --monitor 1 --no-ui     # text mode
"""
import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description="Auto-watch screen reader + live panel")
    p.add_argument("--region", default=None, help="Screen region 'x,y,w,h'. Empty = whole monitor.")
    p.add_argument("--monitor", type=int, default=1, help="Monitor index (1=primary, 0=all).")
    p.add_argument("--no-ui", action="store_true", help="Text mode (no Tk panel).")
    p.add_argument("--no-log", action="store_true", help="Do not write to the results log.")
    p.add_argument("--list-monitors", action="store_true", help="Print monitors and exit.")
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

    from app.panel import main as panel_main
    return panel_main(region=region, monitor=args.monitor, no_ui=args.no_ui,
                      do_log=not args.no_log)


if __name__ == "__main__":
    raise SystemExit(main())
