# -*- coding: utf-8 -*-
"""AUTO-WATCH: zero-manual-input screen reader + a clean always-on-top panel.

You just play the game. A background thread captures the screen, reads the
winning number off the result row (vision/result_reader), logs it, and updates
the Bayesian bias brain (core/bias_tracker). The panel shows the compute
backend (GREEN=GPU / RED=CPU), a big LAST number, spin count, the top-3
distribution with mini bars, the bias p-value, and an honest BET/SKIP advice.

Data safety: every detected spin is appended to runtime/observations.csv the
instant it happens, so nothing is ever lost. Close cleanly with 'Save & Close'
(or the window X, or Ctrl+C once); a short session summary is printed on exit.

Degrades gracefully: missing mss/opencv/Tk -> clear message, no crash.
"""
import queue
import threading
import time

from config import (CAPTURE_FPS, OBSERVATIONS_PATH, RESULT_MARGIN,
                    STABLE_FRAMES, UI_COLORS)


class CaptureWorker(threading.Thread):
    """Background thread: grab frames -> ResultReader -> push events to a queue.
    Tk-free by design (only enqueues plain dicts)."""

    def __init__(self, region=None, monitor=1, fps=CAPTURE_FPS,
                 margin=RESULT_MARGIN, stable_frames=STABLE_FRAMES,
                 out_queue=None, do_log=True):
        super().__init__(daemon=True)
        self.region, self.monitor = region, int(monitor)
        self.fps, self.margin = float(fps), float(margin)
        self.stable_frames = int(stable_frames)
        self.q = out_queue or queue.Queue()
        self.do_log = do_log
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from vision.capture import ScreenSource
            from vision.result_reader import ResultReader
            from app.observation_log import log_result
        except Exception as e:  # pragma: no cover - env dependent
            self.q.put({"type": "error", "msg": str(e)})
            return
        period = 1.0 / max(1.0, self.fps)
        try:
            with ScreenSource(region=self.region, monitor=self.monitor) as src:
                w, h = src.size
                reader = ResultReader(w, h, fps=self.fps, margin=self.margin,
                                      stable_frames=self.stable_frames)
                self.q.put({"type": "ready", "layout": reader.layout, "w": w, "h": h})
                while not self._stop.is_set():
                    t0 = time.time()
                    ev = reader.update(src.grab(), t=time.time())
                    if ev:
                        if self.do_log:
                            try:
                                log_result(ev)
                            except Exception as le:
                                self.q.put({"type": "warn", "msg": f"log: {le}"})
                        self.q.put({"type": "result", **ev})
                    dt = time.time() - t0
                    if dt < period:
                        time.sleep(period - dt)
        except Exception as e:
            self.q.put({"type": "error", "msg": str(e)})
        self.q.put({"type": "stopped"})


def _device_info():
    """Compute backend for the GREEN/RED indicator. Never raises."""
    try:
        from ai.device import detect
        return detect()
    except Exception:
        return {"backend": "CPU", "label": "CPU", "gpus": [], "has_tf": False}


def _stats(tracker, last_number, spins):
    """Compute the display values (shared by UI + headless)."""
    bt = tracker.bias_test()
    bb = tracker.best_bet()
    post = tracker.posterior()
    top = sorted(post.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:3]
    if bt["n"] < 25:
        bias = f"collecting ({bt['n']}/25 spins)"
    elif bt["biased"]:
        bias = f"BIASED  (p={bt['p_value']:.3f})"
    else:
        bias = f"looks fair  (p={bt['p_value']:.3f})"
    advice = (f"BET {bb['number']}   EV\u2193={bb['ev_lo']:+.3f}") if bb else "SKIP"
    return {"last": last_number, "spins": spins, "top": top,
            "bias": bias, "advice": advice, "has_bet": bb is not None}


def _panel_lines(tracker, last_number, spins):
    """Plain-text rows for headless mode."""
    s = _stats(tracker, last_number, spins)
    dist = "  ".join(f"{n}:{d['mean']*100:4.1f}%" for n, d in s["top"]) or "--"
    return [("LAST", str(s["last"]) if s["last"] is not None else "--"),
            ("SPINS", str(s["spins"])), ("TOP3", dist),
            ("BIAS", s["bias"]), ("ADVICE", s["advice"])]


def _print_close_summary(tracker, spins):
    print("\n[auto-watch] closing cleanly ...")
    print(f"[auto-watch] spins observed this session : {spins}")
    try:
        print(f"[auto-watch] advice                     : {tracker.summary()['recommendation']}")
    except Exception:
        pass
    print(f"[auto-watch] all results saved to        : {OBSERVATIONS_PATH}")
    print("[auto-watch] (results are written after EVERY spin, so nothing is lost.)")
    print("[auto-watch] train on them later with     : python scripts/train.py")


def run_ui(worker, tracker):
    """Clean always-on-top panel. Returns False if Tk is unavailable."""
    try:
        import tkinter as tk
        from tkinter import font as tkfont
    except Exception as e:  # pragma: no cover
        print(f"[auto-watch] Tk unavailable ({e}); use --no-ui for text mode.")
        return False
    C = UI_COLORS
    dev = _device_info()
    dev_color = C["gpu"] if dev.get("backend") == "GPU" else C["cpu"]

    PAD = 14
    root = tk.Tk()
    root.title("Spinwheel Auto-Watch")
    root.configure(bg=C["background"])
    root.attributes("-topmost", True)
    root.geometry("340x340+40+40")
    root.minsize(320, 320)

    f_title = tkfont.Font(family="Segoe UI", size=10, weight="bold")
    f_small = tkfont.Font(family="Segoe UI", size=8)
    f_label = tkfont.Font(family="Segoe UI", size=9)
    f_val = tkfont.Font(family="Segoe UI", size=11, weight="bold")
    f_big = tkfont.Font(family="Segoe UI", size=40, weight="bold")
    f_advice = tkfont.Font(family="Segoe UI", size=13, weight="bold")

    def sep():
        tk.Frame(root, bg=C["surface"], height=1).pack(fill="x", padx=PAD, pady=6)

    # ---- Title + compute dot -------------------------------------------
    top = tk.Frame(root, bg=C["background"])
    top.pack(fill="x", padx=PAD, pady=(PAD, 2))
    tk.Label(top, text="AUTO-WATCH", bg=C["background"], fg=C["text"],
             font=f_title).pack(side="left")
    dot = tk.Label(top, text="\u25CF", bg=C["background"], fg=dev_color, font=f_val)
    dot.pack(side="right")
    tk.Label(top, text=dev.get("label", dev.get("backend", "CPU")),
             bg=C["background"], fg=dev_color, font=f_small).pack(side="right", padx=(0, 4))
    subtitle = tk.Label(root, text="observe only \u2013 not a next-spin predictor",
                        bg=C["background"], fg=C["text_secondary"], font=f_small,
                        anchor="w")
    subtitle.pack(fill="x", padx=PAD)

    sep()

    # ---- Big LAST number + spins ---------------------------------------
    mid = tk.Frame(root, bg=C["background"])
    mid.pack(fill="x", padx=PAD)
    last_val = tk.Label(mid, text="--", bg=C["background"], fg=C["primary"], font=f_big)
    last_val.pack(side="left")
    rt = tk.Frame(mid, bg=C["background"])
    rt.pack(side="right", anchor="s", pady=(0, 8))
    tk.Label(rt, text="LAST RESULT", bg=C["background"], fg=C["text_secondary"],
             font=f_small).pack(anchor="e")
    spins_val = tk.Label(rt, text="0 spins", bg=C["background"], fg=C["text"], font=f_val)
    spins_val.pack(anchor="e")

    sep()

    # ---- Top-3 distribution with mini bars -----------------------------
    tk.Label(root, text="TOP DISTRIBUTION", bg=C["background"],
             fg=C["text_secondary"], font=f_small, anchor="w").pack(fill="x", padx=PAD)
    dist_rows = []
    for _ in range(3):
        row = tk.Frame(root, bg=C["background"])
        row.pack(fill="x", padx=PAD, pady=1)
        num = tk.Label(row, text="", width=4, anchor="w", bg=C["background"],
                       fg=C["text"], font=f_label)
        num.pack(side="left")
        bar_bg = tk.Frame(row, bg=C["surface"], height=14, width=170)
        bar_bg.pack(side="left", padx=6)
        bar_bg.pack_propagate(False)
        bar = tk.Frame(bar_bg, bg=C["primary"], height=14, width=0)
        bar.place(x=0, y=0, relheight=1)
        pct = tk.Label(row, text="", width=6, anchor="e", bg=C["background"],
                       fg=C["text_secondary"], font=f_small)
        pct.pack(side="right")
        dist_rows.append((num, bar, bar_bg, pct))

    sep()

    # ---- Bias + Advice --------------------------------------------------
    brow = tk.Frame(root, bg=C["background"])
    brow.pack(fill="x", padx=PAD)
    tk.Label(brow, text="BIAS", width=7, anchor="w", bg=C["background"],
             fg=C["text_secondary"], font=f_small).pack(side="left")
    bias_val = tk.Label(brow, text="--", anchor="w", bg=C["background"],
                        fg=C["text"], font=f_label)
    bias_val.pack(side="left")
    advice_val = tk.Label(root, text="SKIP", bg=C["background"], fg=C["cpu"],
                          font=f_advice, anchor="w")
    advice_val.pack(fill="x", padx=PAD, pady=(2, 0))

    state = {"last": None, "spins": 0, "closing": False}

    def on_close():
        if state["closing"]:
            return
        state["closing"] = True
        worker.stop()
        _print_close_summary(tracker, state["spins"])
        root.after(250, root.destroy)

    # ---- Footer: status + Save & Close ---------------------------------
    footer = tk.Frame(root, bg=C["background"])
    footer.pack(fill="x", padx=PAD, pady=(8, PAD), side="bottom")
    tk.Label(footer, text="saving each spin \u2713", bg=C["background"],
             fg=C["text_secondary"], font=f_small).pack(side="left")
    tk.Button(footer, text="Save & Close", command=on_close, bg=C["button"],
              fg=C["text"], activebackground=C["primary"], activeforeground=C["text"],
              relief="flat", font=f_label, padx=12, pady=3, cursor="hand2").pack(side="right")

    def poll():
        try:
            while True:
                msg = worker.q.get_nowait()
                kind = msg.get("type")
                if kind == "result":
                    tracker.observe(msg["number"])
                    state["last"], state["spins"] = msg["number"], msg["spin_index"]
                elif kind == "ready":
                    subtitle.config(text=f"layout={msg['layout']}  {msg['w']}x{msg['h']}"
                                         "  \u2013 observe only")
                elif kind == "error":
                    advice_val.config(text=str(msg["msg"])[:40], fg=C["cpu"])
        except queue.Empty:
            pass
        s = _stats(tracker, state["last"], state["spins"])
        last_val.config(text=str(s["last"]) if s["last"] is not None else "--")
        spins_val.config(text=f"{s['spins']} spins")
        maxpct = max((d["mean"] for _, d in s["top"]), default=1e-9)
        for i, (num, bar, bar_bg, pct) in enumerate(dist_rows):
            if i < len(s["top"]):
                n, d = s["top"][i]
                num.config(text=str(n))
                pct.config(text=f"{d['mean']*100:.1f}%")
                w = int(bar_bg.winfo_width() * (d["mean"] / maxpct)) if maxpct else 0
                bar.config(width=max(2, w))
            else:
                num.config(text="")
                pct.config(text="")
                bar.config(width=0)
        bias_val.config(text=s["bias"])
        advice_val.config(text=s["advice"],
                          fg=C["gpu"] if s["has_bet"] else C["cpu"])
        if not state["closing"]:
            root.after(150, poll)

    root.protocol("WM_DELETE_WINDOW", on_close)
    worker.start()
    poll()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_close()
    return True


def run_headless(worker, tracker):
    """Text-mode loop: print each detected spin + running bias summary."""
    dev = _device_info()
    tag = "GPU (green)" if dev.get("backend") == "GPU" else "CPU (red)"
    print(f"[auto-watch] compute: {tag} - {dev.get('label', '')}")
    print("[auto-watch] Text mode (no panel). Press Ctrl+C once to stop cleanly.")
    worker.start()
    last, spins = None, 0
    try:
        while worker.is_alive():
            try:
                msg = worker.q.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = msg.get("type")
            if kind == "result":
                tracker.observe(msg["number"])
                last, spins = msg["number"], msg["spin_index"]
                print("  | ".join(f"{k}: {v}" for k, v in _panel_lines(tracker, last, spins)))
            elif kind == "ready":
                print(f"[auto-watch] ready: layout={msg['layout']} {msg['w']}x{msg['h']}")
            elif kind == "error":
                print(f"[auto-watch] ERROR: {msg['msg']}")
                return 2
            elif kind == "stopped":
                break
    except KeyboardInterrupt:
        worker.stop()
    _print_close_summary(tracker, spins)
    return 0


def main(region=None, monitor=1, no_ui=False, do_log=True):
    from core.bias_tracker import OnlineBiasTracker
    tracker = OnlineBiasTracker()
    worker = CaptureWorker(region=region, monitor=monitor, do_log=do_log)
    print("[auto-watch] NOTE: this OBSERVES wheel results; it does NOT predict the next spin.")
    if no_ui or not run_ui(worker, tracker):
        return run_headless(worker, tracker)
    return 0
