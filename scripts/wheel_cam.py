# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""PROMPT 20/21: SCREEN wheel tracker CLI (webcam removed).

    python scripts/wheel_cam.py --list-monitors                  # daftar monitor
    python scripts/wheel_cam.py --region 100,80,640,640          # grab sepetak layar
    python scripts/wheel_cam.py --monitor 1                      # seluruh monitor utama
    python scripts/wheel_cam.py --rounds 5 --no-show             # rekam 5 spin headless

PROMPT 21: webcam DIHAPUS -- game dimainkan di layar laptop (mis. Chrome), jadi
tangkapan SELALU dari layar. Untuk menonton roda secara LANGSUNG sambil main,
pakai panel "LIVE VISION" di dalam aplikasi (lebih nyaman); CLI ini berguna buat
mencari geometri monitor (--list-monitors) dan merekam batch tanpa GUI.

HONEST DISCLAIMER: this OBSERVES a spinning wheel (angle / when it stops / which
segment it rests on). It does NOT predict future spins -- a real spin is chaotic.
Use it to auto-record results, not to "beat" the wheel.

Degrades gracefully: if OpenCV / mss isn't installed it prints how to install it
and exits cleanly instead of crashing.
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
    parser = argparse.ArgumentParser(
        description="Screen wheel tracker (screen-capture only; webcam removed)"
    )
    # --source dipertahankan demi kompatibilitas perintah lama; hanya 'screen'
    # yang valid sekarang (webcam dihapus). Nilai lain ditolak dengan ramah.
    parser.add_argument("--source", default="screen",
                        help="Hanya 'screen' (default). Webcam sudah dihapus -- "
                             "game dimainkan di layar laptop.")
    parser.add_argument("--region", default=None,
                        help="Region layar 'x,y,w,h'. Kosong = seluruh monitor "
                             "(--monitor).")
    parser.add_argument("--monitor", type=int, default=1,
                        help="Indeks monitor (1=utama, 0=semua layar). Dipakai "
                             "bila --region kosong.")
    parser.add_argument("--hsv", default=None,
                        help="Rentang HSV marker 'loH,loS,loV:hiH,hiS,hiV' "
                             "(pisah ';' untuk beberapa rentang). Default hijau.")
    parser.add_argument("--center", default=None,
                        help="Pusat roda 'x,y' relatif ke region "
                             "(default: deteksi otomatis).")
    parser.add_argument("--throttle", type=float, default=0.0,
                        help="Jeda detik antar grab layar (batasi CPU/FPS).")
    parser.add_argument("--list-monitors", action="store_true",
                        help="Cetak daftar monitor lalu keluar.")
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

    from vision.camera import opencv_status
    from vision.observation_log import OBSERVATIONS_PATH, log_observation
    from vision.screen import (
        ScreenWheelTracker, mss_status, parse_region, parse_center, parse_hsv,
    )

    # Webcam removed: reject non-screen sources with a clear, friendly message.
    src = str(args.source).strip().lower()
    if src not in ("screen", ""):
        print("[vision] Webcam sudah DIHAPUS dari aplikasi (game dimainkan di layar).")
        print("[vision] Gunakan tangkapan layar: --source screen (atau cukup tanpa --source).")
        return 2

    # --list-monitors: print monitor geometry and exit (screen mode helper).
    if args.list_monitors:
        if not ScreenWheelTracker.is_available():
            print("[vision] " + mss_status())
            print("[vision] " + opencv_status())
            return 2
        for i, m in enumerate(ScreenWheelTracker.list_monitors()):
            tag = " (semua layar)" if i == 0 else ""
            print(f"[vision] monitor {i}{tag}: {m}")
        print("[vision] Pakai --region 'x,y,w,h' atau --monitor N untuk memilih.")
        return 0

    # Parse shared optional knobs (HSV / center / region) with friendly errors.
    try:
        hsv_ranges = parse_hsv(args.hsv)
        center = parse_center(args.center)
        region = parse_region(args.region) if args.region else None
    except ValueError as e:
        print(f"[vision] Argumen tidak valid: {e}")
        return 2

    if not ScreenWheelTracker.is_available():
        print("[vision] " + mss_status())
        print("[vision] " + opencv_status())
        print("[vision] Screen capture tidak tersedia -- sisa app tetap jalan.")
        return 2
    print("[vision] " + mss_status() + " | " + opencv_status())
    print("[vision] Mode LAYAR: menangkap roda langsung dari layar (bukan webcam).")
    if region is None:
        print(f"[vision] Region kosong -> menangkap seluruh monitor {args.monitor}.")
    print("[vision] CATATAN: ini MENGAMATI roda (bukan meramal spin berikutnya).")
    if not args.no_log:
        print(f"[vision] Tiap spin dicatat ke: {OBSERVATIONS_PATH}")
        print("[vision] Lalu jalankan: python scripts/learn_from_vision.py (uji bias + update prior).")

    rounds = max(1, int(args.rounds))
    logged = 0
    last_rc = 1
    for r in range(1, rounds + 1):
        tracker = ScreenWheelTracker(
            region=region, monitor=args.monitor, center=center,
            hsv_ranges=hsv_ranges, throttle=args.throttle,
        )
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
