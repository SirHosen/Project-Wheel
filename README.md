# Spinwheel-AI

A clean, minimal rebuild focused on **training AI** and **fully automatic data
capture** for a physical spin wheel (numbers `1, 2, 5, 8, 10, 15, 20, 30, 40`).

No manual input, no Selenium. You play; the app watches the screen, reads each
winning number, logs it, and learns from it.

> **Honest framing.** A real spin wheel is *i.i.d.*: the next number is
> independent of the past, so **no model can predict the next spin**. This
> project is for (1) **practicing AI/ML training** on a real, clean pipeline and
> (2) **measuring** the wheel's true distribution + detecting genuine bias. If a
> bet ever looks +EV, it is only because the physical wheel is measurably
> biased -- otherwise the honest answer is **SKIP**.

## What's new (statistical & ML upgrades)

Ten upgrades that make the analysis more rigorous and the pipeline better
tested. Every one degrades gracefully and is covered by a test.

1. **Exact Beta credible intervals** -- the per-number posterior band uses exact
   Beta quantiles instead of a normal approximation (honest at small n / near 0).
2. **Fully Bayesian EV** -- `ev_samples()` Monte-Carlos the Dirichlet posterior
   and reports `P(EV > 0)`, not just a point bound.
3. **Multiple-testing correction** -- searching all 9 numbers at once is Sidak/
   Bonferroni corrected (`config.MULTIPLE_TEST_CORRECTION`) so edges aren't
   false positives.
4. **Power analysis** -- `core/power.py` + `scripts/power_check.py` simulate
   *how many spins* you'd need to detect a bias of a given size.
5. **Optional OCR verification** -- `ResultReader(ocr_verify=True)` cross-checks
   a detected number with pytesseract (`ocr_strict` vetoes unconfirmed reads);
   fully optional and cached.
6. **Auto-calibration** -- `vision/calibrate.py` + `auto_watch.py --auto-calibrate`
   detect the 9 result cells automatically instead of manual config nudging.
7. **Synthetic vision regression** -- `vision/synthetic.py` renders deterministic
   result-row frames so detection is regression-tested without a bundled video.
8. **Richer ML features** -- running frequency, recency gap, fair base rate and a
   repeat flag are fed alongside the embedding (all computed causally).
9. **Multiple baselines** -- the backtest compares the LSTM against most-frequent,
   moving-frequency, persistence and design-argmax baselines in one table.
10. **Probabilistic calibration** -- log-loss and Brier score (`ai/metrics.py`)
    grade the full probability vector, not just top-1 accuracy.

## Layout

```
config.py              # wheel layout, payouts, all knobs
core/
  wheel.py             # design distribution, payouts, EV, chi-square
  bias_tracker.py      # OnlineBiasTracker -- the Bayesian "brain"
  physics.py           # why a heavy wheel is unpredictable (sensitivity)
vision/
  capture.py           # mss screen capture + monitor listing
  result_reader.py     # reads the winning number off the result grid
ai/
  dataset.py           # sliding-window data + synthetic practice generators
  device.py            # PyTorch/CUDA detection for the GREEN/RED indicator
  lstm.py              # clean PyTorch LSTM + HONEST walk-forward backtest
app/
  observation_log.py   # append/load results CSV (training data)
  panel.py             # tiny always-on-top live panel + capture loop
scripts/
  wheel_cam.py         # screen helper (keeps `--list-monitors`)
  auto_watch.py        # run the live panel
  train.py             # train + backtest the AI
  gpu_check.py         # diagnose CPU vs GPU (PyTorch/CUDA)
tests/                 # fast, no-GUI checks
run_tests.py           # cross-platform test runner (use this on Windows)
run_tests.bat          # Windows double-click wrapper
run_tests.sh           # Linux/macOS wrapper
```

## Install (Windows 11 -- PowerShell or cmd)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt            :: core (numpy)
pip install -r requirements-vision.txt     :: screen capture (opencv, mss)
pip install -r requirements-ai.txt         :: AI training (PyTorch) - optional
```

(macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, same pips.)

> **GPU note (important):** the AI trainer runs on **PyTorch**. Plain
> `pip install torch` gives a **CPU-only** build. To use an NVIDIA GPU, install
> the CUDA build instead:
> ```bat
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> Unlike TensorFlow, PyTorch ships native-Windows CUDA wheels, so you do **not**
> need WSL. This LSTM is tiny, though, so CPU trains it in seconds -- GPU is
> purely optional here.

## 1) Capture results automatically (no typing)

```bat
python scripts\wheel_cam.py --source screen --list-monitors   :: find your monitor
python scripts\auto_watch.py --monitor 1                       :: small live panel
python scripts\auto_watch.py --region 60,40,1912,974           :: just the game area
python scripts\auto_watch.py --monitor 1 --no-ui               :: text mode
```

The panel shows a **compute indicator** (🟢 GREEN dot = GPU, 🔴 RED dot = CPU),
the last number, spin count, top-3 distribution, a chi-square bias p-value, and
a BET/SKIP advice.

### How results are saved (nothing is ever lost)
Every detected spin is appended to `runtime/observations.csv` **the instant it
happens** (one row per spin: timestamp, number, spin index, spike, layout). So
even a hard `Ctrl+C` keeps all recorded results -- the file is always up to date.
That CSV is your AI training data (`scripts/train.py` reads it automatically).

### How to close cleanly
- Click **"Save & Close"** in the panel, or close the window with the **X**.
- Or press **Ctrl+C once** in the terminal (works in both panel and `--no-ui` mode).

Either way you get a short session summary (spins observed, current advice, and
the CSV path). No forced kill needed.

### ⚠️ Detection accuracy comes first (calibration)
Everything downstream (stats, bias test, AI) is only as good as the numbers the
reader records. If the detected distribution looks wrong (e.g. it never records
a `1`, which should be the most common number), the cell boxes are misaligned.
Fix it in seconds:
```bat
python scripts\auto_watch.py --snapshot --monitor 1
```
This saves `runtime\calibration.png` with green boxes drawn where the reader is
looking. If the boxes are not sitting on the game's result-row numbers, edit
`RESULT_LANDSCAPE` (or `RESULT_PORTRAIT`) in `config.py` -- `fx_start` = center
of `1`, `fx_end` = center of `40`, `fy` = row height -- then snapshot again
until they line up. Accurate boxes = trustworthy data.

### GPU vs CPU indicator
The indicator follows the **active training backend, PyTorch**:
- 🟢 **GREEN = GPU**: PyTorch sees a CUDA GPU (`torch.cuda.is_available()`).
- 🔴 **RED = CPU**: CPU-only build, no CUDA GPU visible, or PyTorch not installed.

The same status is printed by `python scripts\train.py` on the `[device]` line,
and in `--no-ui` mode at startup.

**Indicator stuck on RED even with PyTorch installed?** Run the diagnostic:
```bat
python scripts\gpu_check.py
```
The most common cause is installing the **CPU-only wheel** (plain
`pip install torch`). Reinstall the CUDA build:
```bat
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Also make sure your **NVIDIA driver is up to date**. If you have no NVIDIA GPU,
RED is expected and totally fine -- this LSTM is small enough that CPU is fast.

## 2) Train the AI (the playground)

```bat
python scripts\train.py                       :: train on your logged results
python scripts\train.py --demo fair --n 1500  :: fair wheel: model should NOT beat baseline
python scripts\train.py --demo markov --n 1500:: sequential pattern: model SHOULD beat baseline
python scripts\train.py --demo biased --n 1500:: biased wheel: bias tracker flags an edge
```

`train.py` runs three honest checks in order:
1. **physics** -- how unpredictable the wheel is in principle,
2. **bias tracker** -- is there a genuine +EV edge? (usually SKIP),
3. **walk-forward backtest** -- does the LSTM actually beat a trivial baseline?

The `--demo` modes are the teaching tool: compare `fair` (no learnable order)
vs `markov` (learnable order) and watch the backtest verdict flip. That contrast
is the whole ML lesson -- it shows *when deep learning helps and when it can't*.
The physics + bias checks run even if PyTorch is not installed.

> **Note on the saved model.** `train.py` writes the trained weights to
> `runtime/lstm_spinwheel.pt`, but the live panel intentionally does **not** load
> them to predict spins -- prediction on a fair wheel is impossible, and the
> Bayesian bias tracker is the real decision-maker. The saved `.pt` file exists
> so you can inspect/re-use the model as an ML exercise.

## 3) Run the tests

```bat
python run_tests.py      :: Windows / macOS / Linux
```

## Notes
- Missing `mss` / `opencv` / PyTorch / Tk never crash the app -- you get a
  clear message and an install hint instead.
- The LSTM is an honest training playground with a real walk-forward backtest,
  NOT a "next-spin predictor". The Bayesian `OnlineBiasTracker` is the primary
  brain (correct for i.i.d. data). On a fair wheel the honest answer is SKIP.
