# -*- coding: utf-8 -*-
"""AUTO-WATCH: zero-manual-input screen reader + a tiny always-on-top panel.

You just play the game. A background thread captures the screen, reads the
winning number off the result grid (vision/result_reader), logs it, and updates
the Bayesian bias brain (core/bias_tracker). A small landscape panel shows the
last number, spin count, top-3 distribution, the bias p-value, and an honest
BET/SKIP advice. No manual typing.

Degrades gracefully: missing mss/opencv/Tk -> clear message, no crash. Falls
back to a text loop when Tk is unavailable.
"""
import queue
import threading
import time

from config import CAPTURE_FPS, RESULT_MARGIN, STABLE_FRAMES, UI_COLORS


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


def _panel_lines(tracker, last_number, spins):
    """Build the panel rows (reused by headless printing)."""
    bt = tracker.bias_test()
    bb = tracker.best_bet()
    post = tracker.posterior()
    top = sorted(post.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:3]
    dist = "  ".join(f"{n}:{d['mean']*100:4.1f}%" for n, d in top)
    bias = ("BIASED (p=%.3f)" % bt["p_value"]) if bt["biased"] else (
        "no bias yet (p=%.3f)" % bt["p_value"] if bt["n"] else "--")
    rec = ("BET %d  EV_lo=%.3f" % (bb["number"], bb["ev_lo"])) if bb else "SKIP (no robust +EV)"
    return [("LAST", str(last_number) if last_number is not None else "--"),
            ("SPINS", str(spins)), ("TOP3", dist or "--"),
            ("BIAS", bias), ("ADVICE", rec)]


def run_ui(worker, tracker):
    """Tiny always-on-top landscape panel. Returns False if Tk is unavailable."""
    try:
        import tkinter as tk
    except Exception as e:  # pragma: no cover
        print(f"[auto-watch] Tk unavailable ({e}); use --no-ui for text mode.")
        return False
    C = UI_COLORS
    root = tk.Tk()
    root.title("Spinwheel Auto-Watch")
    root.configure(bg=C["background"])
    root.attributes("-topmost", True)
    root.geometry("360x150+40+40")
    root.minsize(300, 130)
    header = tk.Label(root, text="AUTO-WATCH  (observe only - no prediction)",
                      bg=C["background"], fg=C["text_secondary"], anchor="w",
                      font=("Segoe UI", 8))
    header.pack(fill="x", padx=10, pady=(8, 2))
    rows = {}
    for key in ("LAST", "SPINS", "TOP3", "BIAS", "ADVICE"):
        fr = tk.Frame(root, bg=C["background"])
        fr.pack(fill="x", padx=10)
        tk.Label(fr, text=key, width=7, anchor="w", bg=C["background"],
                 fg=C["text_secondary"], font=("Segoe UI", 9, "bold")).pack(side="left")
        val = tk.Label(fr, text="--", anchor="w", bg=C["background"],
                       fg=(C["primary"] if key in ("LAST", "ADVICE") else C["text"]),
                       font=("Segoe UI", 10))
        val.pack(side="left", fill="x", expand=True)
        rows[key] = val
    state = {"last": None, "spins": 0}

    def poll():
        try:
            while True:
                msg = worker.q.get_nowait()
                kind = msg.get("type")
                if kind == "result":
                    tracker.observe(msg["number"])
                    state["last"], state["spins"] = msg["number"], msg["spin_index"]
                elif kind == "ready":
                    header.config(text=f"AUTO-WATCH  layout={msg['layout']}  "
                                       f"{msg['w']}x{msg['h']}  (observe only)")
                elif kind == "error":
                    rows["ADVICE"].config(text=msg["msg"][:60])
        except queue.Empty:
            pass
        for key, txt in _panel_lines(tracker, state["last"], state["spins"]):
            rows[key].config(text=txt)
        root.after(150, poll)

    def on_close():
        worker.stop()
        root.after(200, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    worker.start()
    poll()
    root.mainloop()
    return True


def run_headless(worker, tracker):
    """Text-mode loop: print each detected spin + running bias summary."""
    print("[auto-watch] Text mode (no panel). Ctrl+C to stop.")
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
        print("\n[auto-watch] stopped.")
        worker.stop()
    return 0


def main(region=None, monitor=1, no_ui=False, do_log=True):
    from core.bias_tracker import OnlineBiasTracker
    tracker = OnlineBiasTracker()
    worker = CaptureWorker(region=region, monitor=monitor, do_log=do_log)
    print("[auto-watch] NOTE: this OBSERVES wheel results; it does NOT predict the next spin.")
    if no_ui or not run_ui(worker, tracker):
        return run_headless(worker, tracker)
    return 0
