"""PROMPT 13: deep analytics charts (matplotlib, GUI-agnostic).

All functions here return bare matplotlib ``Figure`` objects built ONLY from
recorded history. They never import tkinter, so they are unit-testable in a
headless environment:

    * GUI embeds them via FigureCanvasTkAgg.
    * Export renders them via FigureCanvasAgg / PdfPages.

Four audit visuals are provided:
    1. Reliability diagram (calibration) per engine - are stated confidences
       honest? A perfectly calibrated model sits on the diagonal.
    2. Per-number deviation heatmap - observed minus predicted probability for
       each number x engine. Large deviations = miscalibration / wheel bias.
    3. Equity curve with drawdown shading - cumulative P/L and how deep the
       worst valleys go.
    4. Rolling win-rate (default 50-spin window) vs the chance baseline.

HONEST NOTE: pretty charts do not create edge. They are diagnostics to SEE
whether any apparent performance is real signal or just variance.
"""

from matplotlib.figure import Figure

# Default palette (matches config.settings.UI_COLORS). The GUI passes its own
# dict so embedded charts match the dark theme; tests rely on these defaults.
_DEFAULT_COLORS = {
    "background": "#0D0D0D",
    "panel": "#1A1A2E",
    "card": "#16213E",
    "primary": "#39FF14",
    "secondary": "#FFD700",
    "error": "#FF4444",
    "info": "#00D4FF",
    "text": "#FFFFFF",
    "text_secondary": "#9AA0B4",
}

# Distinct line colors cycled per engine in multi-series charts.
_SERIES = ["#39FF14", "#00D4FF", "#FFD700", "#FF4444", "#B388FF", "#FF9E80"]


def _colors(colors=None):
    c = dict(_DEFAULT_COLORS)
    if colors:
        c.update({k: v for k, v in colors.items() if v})
    return c


def _history(data):
    """Accept either a tracker data dict or a raw history list."""
    if data is None:
        return []
    if isinstance(data, dict):
        return list(data.get("history", []) or [])
    return list(data)


def _style(ax, c):
    ax.set_facecolor(c["panel"])
    ax.tick_params(colors=c["text_secondary"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(c["text_secondary"])
        spine.set_alpha(0.3)
    ax.grid(color=c["text_secondary"], linestyle="--", alpha=0.15)
    ax.title.set_color(c["text"])
    ax.xaxis.label.set_color(c["text_secondary"])
    ax.yaxis.label.set_color(c["text_secondary"])


def _empty(fig, c, msg="Belum ada data yang cukup"):
    ax = fig.add_subplot(111)
    ax.set_facecolor(c["panel"])
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color=c["text_secondary"], fontsize=11, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# --------------------------------------------------------------------------- #
# Data reducers (pure, also useful for tests / the markdown report)
# --------------------------------------------------------------------------- #
def _engine_of(rec):
    return rec.get("engine_used") or "unknown"


def reliability_bins(history, n_bins=10):
    """Per engine: list of (mean_confidence, hit_rate, count) over confidence bins.

    Each individual bet contributes one sample: predicted = its confidence,
    outcome = 1 if that number actually hit, else 0.
    """
    history = _history(history)
    per_engine = {}
    for rec in history:
        eng = _engine_of(rec)
        actual = rec.get("actual_number")
        for bet in rec.get("bets", []) or []:
            conf = bet.get("confidence")
            if conf is None:
                continue
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            hit = 1.0 if bet.get("number") == actual else 0.0
            per_engine.setdefault(eng, []).append((conf, hit))
    out = {}
    width = 1.0 / n_bins
    for eng, samples in per_engine.items():
        bins = []
        for b in range(n_bins):
            lo, hi = b * width, (b + 1) * width
            pts = [s for s in samples
                   if (s[0] >= lo and (s[0] < hi or (b == n_bins - 1 and s[0] <= hi)))]
            if not pts:
                continue
            mc = sum(p[0] for p in pts) / len(pts)
            hr = sum(p[1] for p in pts) / len(pts)
            bins.append((mc, hr, len(pts)))
        if bins:
            out[eng] = bins
    return out


def per_number_deviation(history, valid_numbers):
    """{engine: {number: (predicted_prob, observed_prob, deviation)}}.

    predicted = mean confidence the engine assigned to that number across the
    rounds it was active; observed = how often that number was the actual result
    in those rounds. deviation = observed - predicted.
    """
    history = _history(history)
    per_engine = {}
    for rec in history:
        eng = _engine_of(rec)
        d = per_engine.setdefault(eng, {"n": 0, "conf": {}, "actual": {}})
        d["n"] += 1
        actual = rec.get("actual_number")
        if actual is not None:
            d["actual"][actual] = d["actual"].get(actual, 0) + 1
        for bet in rec.get("bets", []) or []:
            num = bet.get("number")
            conf = bet.get("confidence")
            if num is None or conf is None:
                continue
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            d["conf"].setdefault(num, []).append(conf)
    out = {}
    for eng, d in per_engine.items():
        n = max(1, d["n"])
        row = {}
        for num in valid_numbers:
            confs = d["conf"].get(num, [])
            predicted = (sum(confs) / n) if confs else 0.0
            observed = d["actual"].get(num, 0) / n
            row[num] = (predicted, observed, observed - predicted)
        out[eng] = row
    return out


def equity_series(history):
    """Return (equity, running_peak, drawdown) lists from profit_change."""
    history = _history(history)
    equity, peaks, dd = [], [], []
    cum = 0.0
    peak = 0.0
    for rec in history:
        cum += float(rec.get("profit_change", 0) or 0)
        peak = max(peak, cum)
        equity.append(cum)
        peaks.append(peak)
        dd.append(peak - cum)
    return equity, peaks, dd


def rolling_winrate(history, window=50):
    """Rolling win-rate (%). Expanding mean until `window` samples exist."""
    history = _history(history)
    wins = [1.0 if rec.get("is_win") else 0.0 for rec in history]
    out = []
    for i in range(len(wins)):
        lo = max(0, i - window + 1)
        seg = wins[lo:i + 1]
        out.append(100.0 * sum(seg) / len(seg))
    return out


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def build_reliability_figure(history, colors=None, n_bins=10):
    c = _colors(colors)
    fig = Figure(figsize=(5.2, 3.6), dpi=100, facecolor=c["card"])
    bins = reliability_bins(history, n_bins=n_bins)
    if not bins:
        return _empty(fig, c, "Belum ada taruhan untuk kalibrasi")
    ax = fig.add_subplot(111)
    _style(ax, c)
    ax.plot([0, 1], [0, 1], linestyle="--", color=c["text_secondary"],
            alpha=0.6, label="Kalibrasi sempurna")
    for i, (eng, pts) in enumerate(sorted(bins.items())):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=4,
                color=_SERIES[i % len(_SERIES)], label=str(eng))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence yang diklaim")
    ax.set_ylabel("Hit-rate sebenarnya")
    ax.set_title("Reliability / kalibrasi per engine")
    ax.legend(facecolor=c["panel"], labelcolor=c["text"], fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def build_deviation_heatmap_figure(history, valid_numbers, colors=None):
    c = _colors(colors)
    fig = Figure(figsize=(5.2, 3.6), dpi=100, facecolor=c["card"])
    dev = per_number_deviation(history, valid_numbers)
    if not dev:
        return _empty(fig, c, "Belum ada data prediksi per-angka")
    engines = sorted(dev.keys())
    matrix = [[dev[eng][num][2] for eng in engines] for num in valid_numbers]
    ax = fig.add_subplot(111)
    ax.set_facecolor(c["panel"])
    vmax = max((abs(v) for row in matrix for v in row), default=0.0) or 0.01
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(engines, color=c["text"], fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(valid_numbers)))
    ax.set_yticklabels([str(n) for n in valid_numbers], color=c["text"], fontsize=8)
    ax.set_title("Deviasi observed - predicted per angka", color=c["text"])
    ax.set_xlabel("Engine", color=c["text_secondary"])
    ax.set_ylabel("Angka", color=c["text_secondary"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=c["text_secondary"], labelsize=7)
    fig.tight_layout()
    return fig


def build_equity_figure(history, colors=None):
    c = _colors(colors)
    fig = Figure(figsize=(5.2, 3.6), dpi=100, facecolor=c["card"])
    equity, peaks, dd = equity_series(history)
    if not equity:
        return _empty(fig, c, "Belum ada riwayat profit")
    ax = fig.add_subplot(111)
    _style(ax, c)
    x = range(len(equity))
    ax.plot(x, equity, color=c["primary"], linewidth=1.6, label="P/L kumulatif")
    ax.plot(x, peaks, color=c["secondary"], linewidth=0.8, alpha=0.5, label="Puncak")
    ax.fill_between(x, equity, peaks, color=c["error"], alpha=0.18, label="Drawdown")
    ax.axhline(0, color=c["text_secondary"], linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Ronde")
    ax.set_ylabel("Token (P/L kumulatif)")
    ax.set_title("Kurva ekuitas + drawdown")
    ax.legend(facecolor=c["panel"], labelcolor=c["text"], fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def build_rolling_winrate_figure(history, window=50, colors=None, baseline=None):
    c = _colors(colors)
    fig = Figure(figsize=(5.2, 3.6), dpi=100, facecolor=c["card"])
    series = rolling_winrate(history, window=window)
    if not series:
        return _empty(fig, c, "Belum ada riwayat menang/kalah")
    ax = fig.add_subplot(111)
    _style(ax, c)
    ax.plot(range(len(series)), series, color=c["info"], linewidth=1.6,
            label=f"Win-rate (window {window})")
    if baseline is not None:
        ax.axhline(baseline, color=c["secondary"], linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"Baseline acak ~{baseline:.1f}%")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Ronde")
    ax.set_ylabel("Win-rate (%)")
    ax.set_title("Win-rate bergulir")
    ax.legend(facecolor=c["panel"], labelcolor=c["text"], fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def build_all_figures(data, valid_numbers, colors=None, window=50, baseline=None):
    """Return an ordered dict {name: Figure} of the four audit charts."""
    history = _history(data)
    return {
        "reliability": build_reliability_figure(history, colors=colors),
        "deviation_heatmap": build_deviation_heatmap_figure(history, valid_numbers, colors=colors),
        "equity_curve": build_equity_figure(history, colors=colors),
        "rolling_winrate": build_rolling_winrate_figure(history, window=window, colors=colors, baseline=baseline),
    }


# --------------------------------------------------------------------------- #
# Export (headless-safe: attaches an Agg canvas, never touches the GUI backend)
# --------------------------------------------------------------------------- #
def export_charts(figures, base_path, fmt="png"):
    """Save each figure to ``<base_path>_<name>.<fmt>`` (png) OR a single PDF.

    Returns the list of file paths written.
    """
    import os
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    base, ext = os.path.splitext(base_path)
    fmt = (fmt or "png").lower().lstrip(".")
    written = []
    if fmt == "pdf":
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = base + ".pdf"
        with PdfPages(pdf_path) as pdf:
            for fig in figures.values():
                FigureCanvasAgg(fig)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
        written.append(pdf_path)
        return written
    for name, fig in figures.items():
        FigureCanvasAgg(fig)
        path = f"{base}_{name}.{fmt}"
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=120)
        written.append(path)
    return written


def export_audit_charts(data, base_path, valid_numbers, colors=None,
                        window=50, baseline=None, fmt="png"):
    """Convenience: build all four figures from data and export them."""
    figs = build_all_figures(data, valid_numbers, colors=colors,
                             window=window, baseline=baseline)
    return export_charts(figs, base_path, fmt=fmt)


# --------------------------------------------------------------------------- #
# PROMPT 15: bankroll visuals (daily P/L bars + GitHub-style calendar heatmap)
# --------------------------------------------------------------------------- #
def build_daily_pnl_figure(data, colors=None):
    """Bar chart of realized P/L per calendar day (green up, red down) with the
    cumulative equity overlaid as a line."""
    from core.bankroll import daily_report
    c = _colors(colors)
    fig = Figure(figsize=(7.0, 3.4), dpi=100, facecolor=c["background"])
    days = daily_report(_history(data))
    if not days:
        return _empty(fig, c, "Belum ada P/L harian")
    ax = fig.add_subplot(111)
    _style(ax, c)
    labels = [d["date"][5:] for d in days]  # MM-DD
    profits = [d["profit"] for d in days]
    cum = [d["cum_profit"] for d in days]
    x = list(range(len(days)))
    bar_colors = [c["primary"] if p >= 0 else c["error"] for p in profits]
    ax.bar(x, profits, color=bar_colors, alpha=0.85, label="P/L harian")
    ax.axhline(0, color=c["text_secondary"], linewidth=0.8, alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(x, cum, color=c["info"], linewidth=1.8, marker="o",
             markersize=3, label="Kumulatif")
    ax2.tick_params(colors=c["info"], labelsize=8)
    ax2.spines["right"].set_color(c["info"])
    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
    ax.set_title("P/L Harian + Ekuitas Kumulatif", fontsize=11)
    ax.set_ylabel("Token / hari")
    ax2.set_ylabel("Kumulatif", color=c["info"])
    fig.tight_layout()
    return fig


def build_calendar_heatmap_figure(data, colors=None):
    """GitHub-style calendar heatmap: weeks as columns, weekday as rows, each
    cell shaded by that day's net P/L (red=loss, green=profit, neutral=0)."""
    from datetime import datetime, timedelta
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    from core.bankroll import daily_report

    c = _colors(colors)
    fig = Figure(figsize=(7.0, 3.0), dpi=100, facecolor=c["background"])
    days = daily_report(_history(data))
    if not days:
        return _empty(fig, c, "Belum ada data kalender")
    ax = fig.add_subplot(111)
    ax.set_facecolor(c["panel"])

    dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in days]
    pnl = {d["date"]: d["profit"] for d in days}
    start = min(dates)
    end = max(dates)
    start_monday = start - timedelta(days=start.weekday())
    n_weeks = ((end - start_monday).days) // 7 + 1
    grid = np.full((7, n_weeks), np.nan)
    for d in days:
        dt = datetime.strptime(d["date"], "%Y-%m-%d").date()
        wk = ((dt - start_monday).days) // 7
        grid[dt.weekday(), wk] = pnl[d["date"]]

    finite = grid[np.isfinite(grid)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0
    cmap = LinearSegmentedColormap.from_list(
        "pnl", [c["error"], c["card"], c["primary"]])
    cmap.set_bad(c["panel"])
    im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"],
                       fontsize=8, color=c["text_secondary"])
    week_starts = [start_monday + timedelta(weeks=w) for w in range(n_weeks)]
    step = max(1, n_weeks // 10)
    ax.set_xticks(list(range(0, n_weeks, step)))
    ax.set_xticklabels([week_starts[w].strftime("%d/%m")
                        for w in range(0, n_weeks, step)],
                       fontsize=7, rotation=45, ha="right",
                       color=c["text_secondary"])
    ax.set_title("Kalender Heatmap P/L Harian", fontsize=11, color=c["text"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors=c["text_secondary"], labelsize=7)
    fig.tight_layout()
    return fig


def build_bankroll_figures(data, colors=None):
    """Ordered dict of the two bankroll figures."""
    return {
        "daily_pnl": build_daily_pnl_figure(data, colors=colors),
        "calendar_heatmap": build_calendar_heatmap_figure(data, colors=colors),
    }


def export_bankroll_charts(data, base_path, colors=None, fmt="png"):
    """Build + export the bankroll figures (PNG = 2 files, PDF = 1 file)."""
    return export_charts(build_bankroll_figures(data, colors=colors),
                         base_path, fmt=fmt)
