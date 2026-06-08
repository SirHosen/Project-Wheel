# -*- coding: utf-8 -*-
"""
gui/views/main_window.py - Main application window (View layer).

Clean, professional dark dashboard. No emoji glyphs are used anywhere in the
UI: Tkinter cannot reliably render colour emoji, so they appear as empty boxes.
Status and accents are conveyed with typography, colour and simple shapes.
"""

import customtkinter as ctk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter

from config import settings
from gui.viewmodels.main_viewmodel import MainViewModel

# Shorthands for the theme.
C = settings.UI_COLORS
FAMILY = settings.FONTS["body"][0]


def font(size, weight="normal"):
    """Convenience builder for a themed font tuple."""
    return (FAMILY, size, weight)


class MainWindow(ctk.CTk):
    def __init__(self, viewmodel: MainViewModel):
        super().__init__()
        self.vm = viewmodel

        ctk.set_appearance_mode("dark")
        self.title("Spin Wheel Predictor")
        self.geometry("1180x820")
        self.minsize(900, 640)
        self.resizable(True, True)
        self.configure(fg_color=C["background"])

        # Root layout: header (fixed) / content (flex) / stats (flex).
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=2)
        self.grid_columnconfigure(0, weight=1)

        self.current_predictions = []

        self._build_header()
        self._build_content()
        self._build_stats_zone()

        self._refresh_stats()
        self._refresh_header()

    # ------------------------------------------------------------------ #
    # Small reusable helpers
    # ------------------------------------------------------------------ #
    def _section_label(self, parent, text):
        return ctk.CTkLabel(
            parent,
            text=text.upper(),
            font=font(12, "bold"),
            text_color=C["text_secondary"],
            anchor="w",
        )

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    def _build_header(self):
        # A 2px gold underline is created by padding the panel inside a gold frame.
        header_outer = ctk.CTkFrame(self, fg_color=C["secondary"], corner_radius=0, height=74)
        header_outer.grid(row=0, column=0, sticky="ew")
        header_outer.grid_propagate(False)

        header = ctk.CTkFrame(header_outer, fg_color=C["panel"], corner_radius=0)
        header.pack(fill="both", expand=True, pady=(0, 2))

        # Brand + capital (left).
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=24)
        ctk.CTkLabel(
            left, text="SPIN WHEEL PREDICTOR", font=font(22, "bold"), text_color=C["text"]
        ).pack(anchor="w")
        self.capital_lbl = ctk.CTkLabel(
            left, text="Modal: - token", font=font(12, "bold"), text_color=C["secondary"]
        )
        self.capital_lbl.pack(anchor="w")

        # Engine selector (right).
        self.engine_var = ctk.StringVar(value=self.vm.selected_engine)
        self.engine_dropdown = ctk.CTkOptionMenu(
            header,
            values=["AI-Optimal", "Ensemble", "Markov", "TF-LSTM", "Heuristic"],
            variable=self.engine_var,
            command=self._on_engine_change,
            width=140,
            font=font(13),
            fg_color=C["card"],
            button_color=C["secondary"],
            button_hover_color=C["info"],
            text_color=C["text"],
            dropdown_fg_color=C["card"],
            dropdown_text_color=C["text"],
            dropdown_hover_color=C["panel"],
        )
        self.engine_dropdown.pack(side="right", padx=(0, 24))
        ctk.CTkLabel(
            header, text="ENGINE", font=font(11, "bold"), text_color=C["text_secondary"]
        ).pack(side="right", padx=(0, 8))

        # Status indicator: a coloured dot + label (no emoji).
        gpu_on = getattr(self.vm, "gpu_available", False)
        dot_color = C["primary"] if gpu_on else C["text_secondary"]
        status_text = "GPU MODE" if gpu_on else "CPU MODE"
        status = ctk.CTkFrame(header, fg_color="transparent")
        status.pack(side="right", padx=24)
        dot = ctk.CTkFrame(status, width=10, height=10, corner_radius=5, fg_color=dot_color)
        dot.pack(side="left")
        dot.pack_propagate(False)
        ctk.CTkLabel(
            status, text=status_text, font=font(12, "bold"), text_color=dot_color
        ).pack(side="left", padx=8)

    # ------------------------------------------------------------------ #
    # Content (input + predictions)
    # ------------------------------------------------------------------ #
    def _build_content(self):
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=38, uniform="cols")
        content.grid_columnconfigure(1, weight=62, uniform="cols")
        content.grid_rowconfigure(0, weight=1)
        self.content_frame = content

        self._build_input_zone()
        self._build_prediction_zone()

    def _build_input_zone(self):
        left = ctk.CTkFrame(self.content_frame, fg_color=C["panel"], corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        # Primary call to action pinned to the bottom.
        self.hitung_btn = ctk.CTkButton(
            left,
            text="HITUNG PREDIKSI",
            font=font(15, "bold"),
            height=48,
            corner_radius=10,
            command=self._on_hitung,
            fg_color=C["secondary"],
            text_color=C["background"],
            hover_color=C["info"],
        )
        self.hitung_btn.pack(side="bottom", fill="x", padx=20, pady=(8, 20))

        # Scrollable controls so nothing is ever clipped on small screens.
        body = ctk.CTkScrollableFrame(
            left,
            fg_color="transparent",
            scrollbar_button_color=C["card"],
            scrollbar_button_hover_color=C["secondary"],
        )
        body.pack(side="top", fill="both", expand=True, padx=10, pady=(16, 0))

        # --- Modal token ---
        self._section_label(body, "Modal Token").pack(fill="x", pady=(2, 4))
        modal_f = ctk.CTkFrame(body, fg_color="transparent")
        modal_f.pack(fill="x")
        self.modal_entry = ctk.CTkEntry(
            modal_f, font=font(13), justify="center",
            fg_color=C["card"], border_color=C["secondary"], border_width=1,
        )
        self.modal_entry.insert(0, str(self.vm.current_capital))
        self.modal_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            modal_f, text="SET", width=64, command=self._on_set_modal, font=font(12, "bold"),
            fg_color=C["card"], hover_color=C["secondary"], text_color=C["text"],
        ).pack(side="left", padx=(8, 0))

        # --- Manual probabilities ---
        self._section_label(body, "Probabilitas Manual (%)").pack(fill="x", pady=(18, 4))
        perc_frame = ctk.CTkFrame(body, fg_color="transparent")
        perc_frame.pack(fill="x")
        for i in range(5):
            perc_frame.grid_columnconfigure(i, weight=1)
        self.perc_entries = {}
        row = col = 0
        for num in settings.VALID_NUMBERS:
            cell = ctk.CTkFrame(perc_frame, fg_color="transparent")
            cell.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            ctk.CTkLabel(
                cell, text=str(num), font=font(11, "bold"), text_color=C["text_secondary"]
            ).pack()
            e = ctk.CTkEntry(
                cell, font=font(12), justify="center", fg_color=C["card"], border_width=0
            )
            e.insert(0, "0")
            e.pack(fill="x")
            e.bind("<KeyRelease>", self._update_perc_total)
            self.perc_entries[num] = e
            col += 1
            if col > 4:
                col = 0
                row += 1

        self.perc_total_lbl = ctk.CTkLabel(
            body, text="Total: 0%", font=font(12, "bold"), text_color=C["error"], anchor="w"
        )
        self.perc_total_lbl.pack(fill="x", pady=(6, 0))
        ctk.CTkButton(
            body, text="AUTO ISI DARI WHEEL", command=self._auto_fill_perc, height=32,
            font=font(12, "bold"), fg_color=C["card"], hover_color=C["secondary"],
            text_color=C["text"],
        ).pack(fill="x", pady=(6, 0))

        # Lock + auto-update: freezes the % as a prior, then refreshes it live
        # after every confirmed spin (self-tuning toward the real wheel).
        self.lock_btn = ctk.CTkButton(
            body, text="KUNCI & AUTO-UPDATE", command=self._on_toggle_lock, height=34,
            font=font(12, "bold"), fg_color=C["primary"], hover_color="#2FE00F",
            text_color=C["background"],
        )
        self.lock_btn.pack(fill="x", pady=(6, 0))
        self.lock_status_lbl = ctk.CTkLabel(
            body, text="Status: manual (statis) - kunci untuk update otomatis tiap putaran.",
            font=font(11), text_color=C["text_secondary"], anchor="w", wraplength=300,
            justify="left",
        )
        self.lock_status_lbl.pack(fill="x", pady=(4, 0))

        # --- History selector ---
        self._section_label(body, "Riwayat Dianalisis").pack(fill="x", pady=(18, 4))
        self.hist_var = ctk.StringVar(value="100 Terakhir")
        self.hist_seg = ctk.CTkSegmentedButton(
            body, values=["10 Terakhir", "50 Terakhir", "100 Terakhir"],
            variable=self.hist_var, command=self._on_hist_change, font=font(12),
            selected_color=C["secondary"], selected_hover_color=C["info"],
            unselected_color=C["card"], fg_color=C["card"], text_color=C["text"],
        )
        self.hist_seg.pack(fill="x")

        # --- Risk allocation ---
        self.risk_lbl = ctk.CTkLabel(
            body, text="Budget Taruhan: 30% dari modal", font=font(12), text_color=C["text"],
            anchor="w",
        )
        self.risk_lbl.pack(fill="x", pady=(18, 2))
        self.risk_slider = ctk.CTkSlider(
            body, from_=10, to=50, number_of_steps=40, command=self._on_risk_change,
            progress_color=C["secondary"], button_color=C["secondary"],
            button_hover_color=C["info"], fg_color=C["card"],
        )
        self.risk_slider.set(30)
        self.risk_slider.pack(fill="x")

        self.ev_lbl = ctk.CTkLabel(
            body, text="EV: -   |   Prob: -%", font=font(12),
            text_color=C["text_secondary"], anchor="w",
        )
        self.ev_lbl.pack(fill="x", pady=(10, 4))

        # --- Live learning panel (continuous Ensemble brain) ---
        # Visualises how much the system currently trusts each signal. The bars
        # and accuracies update live after every confirmed spin.
        self._section_label(body, "Pembelajaran Live (Ensemble)").pack(fill="x", pady=(18, 4))
        self.learn_info_lbl = ctk.CTkLabel(
            body, text="Belajar dari 0 putaran  -  LSTM-GPU: -",
            font=font(11), text_color=C["text_secondary"], anchor="w",
        )
        self.learn_info_lbl.pack(fill="x", pady=(0, 6))
        self.learn_widgets = {}
        _model_labels = {"physics": "Fisika", "bayes": "Bayes", "markov": "Markov", "lstm": "LSTM-GPU"}
        for key in ("physics", "bayes", "markov", "lstm"):
            lr = ctk.CTkFrame(body, fg_color="transparent")
            lr.pack(fill="x", pady=2)
            lr.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                lr, text=_model_labels[key], font=font(11, "bold"),
                text_color=C["text"], width=78, anchor="w",
            ).grid(row=0, column=0, sticky="w")
            bar = ctk.CTkProgressBar(
                lr, progress_color=C["info"], fg_color=C["card"], height=8, corner_radius=4,
            )
            bar.set(0.25)
            bar.grid(row=0, column=1, sticky="ew", padx=8)
            acc = ctk.CTkLabel(
                lr, text="acc -", font=font(10), text_color=C["text_secondary"], width=58,
            )
            acc.grid(row=0, column=2, sticky="e")
            self.learn_widgets[key] = (bar, acc)

    def _build_prediction_zone(self):
        self.right = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.right.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)

        self._section_label(self.right, "Prediksi Teratas").pack(fill="x", pady=(0, 6))

        # EV-aware recommendation banner (BET vs SKIP).
        self.reco_lbl = ctk.CTkLabel(
            self.right, text="Tekan HITUNG PREDIKSI untuk rekomendasi taruhan.",
            font=font(12, "bold"), text_color=C["text_secondary"], anchor="w",
        )
        self.reco_lbl.pack(fill="x", pady=(0, 6))

        # Scrollable so prediction cards never clip when the window shrinks.
        cards_scroll = ctk.CTkScrollableFrame(
            self.right, fg_color="transparent",
            scrollbar_button_color=C["card"], scrollbar_button_hover_color=C["secondary"],
        )
        cards_scroll.pack(fill="both", expand=True)
        # Inner normal frame keeps winfo_children() == the cards (flash/destroy).
        self.cards_frame = ctk.CTkFrame(cards_scroll, fg_color="transparent")
        self.cards_frame.pack(fill="both", expand=True)
        self._show_message_card("Belum ada prediksi", "Tekan HITUNG PREDIKSI untuk memulai.")

        # Result confirmation.
        self.confirm_frame = ctk.CTkFrame(self.right, fg_color=C["panel"], corner_radius=16)
        self.confirm_frame.pack(fill="x", pady=(12, 0))
        self._section_label(self.confirm_frame, "Input Hasil Aktual").pack(
            fill="x", padx=16, pady=(12, 6)
        )
        inp_f = ctk.CTkFrame(self.confirm_frame, fg_color="transparent")
        inp_f.pack(fill="x", padx=16, pady=(0, 14))
        ctk.CTkLabel(
            inp_f, text="Angka keluar:", font=font(13), text_color=C["text_secondary"]
        ).pack(side="left")
        self.result_var = ctk.StringVar(value=str(settings.VALID_NUMBERS[0]))
        ctk.CTkOptionMenu(
            inp_f, values=[str(n) for n in settings.VALID_NUMBERS], variable=self.result_var,
            width=90, font=font(13), fg_color=C["card"], button_color=C["secondary"],
            button_hover_color=C["info"], text_color=C["text"],
            dropdown_fg_color=C["card"], dropdown_text_color=C["text"],
            dropdown_hover_color=C["panel"],
        ).pack(side="left", padx=10)
        self.confirm_btn = ctk.CTkButton(
            inp_f, text="KONFIRMASI HASIL", command=self._on_confirm, font=font(13, "bold"),
            fg_color=C["primary"], text_color=C["background"], hover_color="#2FE00F",
        )
        self.confirm_btn.pack(side="left")

    def _show_message_card(self, title, subtitle):
        for w in self.cards_frame.winfo_children():
            w.destroy()
        ph = ctk.CTkFrame(self.cards_frame, fg_color=C["panel"], corner_radius=16)
        ph.pack(fill="both", expand=True)
        holder = ctk.CTkFrame(ph, fg_color="transparent")
        holder.pack(pady=48)
        ctk.CTkLabel(
            holder, text=title, font=font(18, "bold"), text_color=C["text_secondary"]
        ).pack()
        ctk.CTkLabel(
            holder, text=subtitle, font=font(12), text_color=C["text_secondary"]
        ).pack(pady=(4, 0))

    # ------------------------------------------------------------------ #
    # Stats zone
    # ------------------------------------------------------------------ #
    def _build_stats_zone(self):
        stats_bg = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0)
        stats_bg.grid(row=2, column=0, sticky="nsew")
        stats_bg.grid_columnconfigure(0, weight=1)
        stats_bg.grid_rowconfigure(1, weight=1)

        cards_f = ctk.CTkFrame(stats_bg, fg_color="transparent")
        cards_f.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        for i in range(4):
            cards_f.grid_columnconfigure(i, weight=1, uniform="stat")
        self.stat_winrate = self._make_stat_card(cards_f, "WIN RATE", "0.0%", 0, C["primary"])
        self.stat_total = self._make_stat_card(cards_f, "TOTAL GAME", "0", 1, C["text"])
        self.stat_profit = self._make_stat_card(cards_f, "PROFIT", "0", 2, C["secondary"])
        self.stat_streak = self._make_stat_card(cards_f, "WIN STREAK", "0x", 3, C["info"])

        chart_f = ctk.CTkFrame(stats_bg, fg_color="transparent")
        chart_f.grid(row=1, column=0, sticky="nsew", padx=16)
        self.fig = Figure(figsize=(8, 2), facecolor=C["panel"])
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.18)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_f)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        actions_f = ctk.CTkFrame(stats_bg, fg_color="transparent")
        actions_f.grid(row=2, column=0, sticky="ew", padx=16, pady=12)
        self.adv_lbl = ctk.CTkLabel(
            actions_f,
            text="Best: 0     Worst: 0     Avg/ronde: 0.0     Max streak: 0",
            font=font(12), text_color=C["text_secondary"],
        )
        self.adv_lbl.pack(side="left")
        ctk.CTkButton(
            actions_f, text="RESET SEMUA", command=self._on_reset, width=130, font=font(12, "bold"),
            fg_color="transparent", border_width=1, border_color=C["error"],
            text_color=C["error"], hover_color=C["card"],
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            actions_f, text="EXPORT CSV", command=self._on_export, width=130, font=font(12, "bold"),
            fg_color=C["card"], hover_color=C["info"], text_color=C["text"],
        ).pack(side="right")
        ctk.CTkButton(
            actions_f, text="EXPORT LAPORAN AUDIT", command=self._on_export_audit, width=190,
            font=font(12, "bold"), fg_color=C["secondary"], hover_color=C["info"],
            text_color=C["background"],
        ).pack(side="right", padx=(0, 8))

        # Statistical wheel-bias verdict (is the model beating chance?).
        self.bias_lbl = ctk.CTkLabel(
            stats_bg, text="Bias roda: belum diuji (butuh lebih banyak data).",
            font=font(11), text_color=C["text_secondary"], anchor="w",
        )
        self.bias_lbl.grid(row=3, column=0, sticky="w", padx=16, pady=(0, 10))

    def _make_stat_card(self, parent, title, val, col, accent):
        f = ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=12)
        f.grid(row=0, column=col, padx=6, sticky="ew")
        ctk.CTkLabel(
            f, text=title, font=font(11, "bold"), text_color=C["text_secondary"]
        ).pack(pady=(12, 2))
        lbl = ctk.CTkLabel(f, text=val, font=font(24, "bold"), text_color=accent)
        lbl.pack(pady=(0, 12))
        return lbl

    def _style_axes(self):
        self.ax.set_facecolor(C["panel"])
        self.ax.tick_params(colors=C["text_secondary"], labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.grid(color=C["text_secondary"], linestyle="--", alpha=0.15)

    # ------------------------------------------------------------------ #
    # Input handlers
    # ------------------------------------------------------------------ #
    def _update_perc_total(self, event=None):
        total = 0.0
        for num, e in self.perc_entries.items():
            try:
                val = float(e.get())
                self.vm.manual_percentages[num] = val
                total += val
            except ValueError:
                pass
        color = C["primary"] if abs(total - 100) < 0.01 else C["error"]
        self.perc_total_lbl.configure(text=f"Total: {total:.1f}%", text_color=color)

    def _auto_fill_perc(self):
        if self.vm.manual_locked:
            return
        seq = settings.SPINWHEEL_SEQUENCE
        counts = Counter(seq)
        total_len = len(seq)
        for num in settings.VALID_NUMBERS:
            prob = (counts[num] / total_len) * 100
            self.perc_entries[num].delete(0, "end")
            self.perc_entries[num].insert(0, f"{prob:.2f}")
        self._update_perc_total()

    def _apply_live_percentages(self, live, lock=False):
        """Write live percentages into the entry boxes. When lock=True the
        boxes are made read-only to signal they are auto-managed."""
        for num, e in self.perc_entries.items():
            e.configure(state="normal")
            e.delete(0, "end")
            e.insert(0, f"{live.get(num, 0.0):.2f}")
            if lock:
                e.configure(state="disabled")
        self._update_perc_total()

    def _on_toggle_lock(self):
        if not self.vm.manual_locked:
            # Pull whatever is currently typed, then lock + seed the live prior.
            self._update_perc_total()
            live = self.vm.lock_manual_percentages()
            self._apply_live_percentages(live, lock=True)
            spins = len(self.vm.get_current_history(1000))
            self.lock_btn.configure(text="BUKA KUNCI (mode manual)", fg_color=C["secondary"])
            self.lock_status_lbl.configure(
                text=(
                    f"Status: TERKUNCI - persen update otomatis tiap konfirmasi hasil "
                    f"(prior 54 putaran + {spins} hasil nyata)."
                ),
                text_color=C["primary"],
            )
        else:
            self.vm.unlock_manual_percentages()
            for e in self.perc_entries.values():
                e.configure(state="normal")
            self.lock_btn.configure(text="KUNCI & AUTO-UPDATE", fg_color=C["primary"])
            self.lock_status_lbl.configure(
                text="Status: manual (statis) - kunci untuk update otomatis tiap putaran.",
                text_color=C["text_secondary"],
            )

    def _refresh_live_lock(self):
        """After a confirmed spin, repaint the live percentages if locked."""
        if getattr(self.vm, "manual_locked", False):
            self._apply_live_percentages(self.vm.manual_percentages, lock=True)
            spins = len(self.vm.get_current_history(1000))
            self.lock_status_lbl.configure(
                text=(
                    f"Status: TERKUNCI - persen ter-update ({spins} hasil nyata + prior 54)."
                ),
                text_color=C["primary"],
            )

    def _on_hist_change(self, val):
        self.vm.history_length = int(val.split()[0])

    def _on_risk_change(self, val):
        risk = int(val)
        self.vm.risk_percentage = risk / 100.0
        self.risk_lbl.configure(text=f"Budget Taruhan: {risk}% dari modal")

    def _on_engine_change(self, choice):
        self.vm.selected_engine = choice

    def _on_set_modal(self):
        try:
            val = int(float(self.modal_entry.get()))
            if val <= 0:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror("Modal", "Masukkan angka modal yang valid (lebih dari 0).")
            return
        self.vm.set_initial_capital(val)
        self._refresh_header()
        messagebox.showinfo("Modal", f"Modal diset ke {val} token.")

    # ------------------------------------------------------------------ #
    # Prediction flow
    # ------------------------------------------------------------------ #
    def _on_hitung(self):
        self.hitung_btn.configure(text="MENGHITUNG...", state="disabled")
        self._update_perc_total()
        # Run prediction off the UI thread so the window never freezes.
        self.vm.get_predictions_async(
            lambda allocs: self.after(0, self._display_predictions, allocs)
        )

    def _display_predictions(self, allocs=None):
        self.hitung_btn.configure(text="HITUNG PREDIKSI", state="normal")
        allocs = allocs or []
        self.current_predictions = allocs

        if not allocs:
            self.reco_lbl.configure(
                text="Tekan HITUNG PREDIKSI untuk rekomendasi taruhan.",
                text_color=C["text_secondary"],
            )
            self._show_message_card(
                "Tidak ada prediksi", "Tambah riwayat hasil atau ganti engine, lalu coba lagi."
            )
            return

        # EV-aware verdict: any positive-EV stake recommended?
        has_bet = any(p.get("token_bet", 0) > 0 for p in allocs)
        if has_bet:
            staked = sum(p.get("token_bet", 0) for p in allocs)
            self.reco_lbl.configure(
                text=f"EDGE +EV TERDETEKSI  -  disarankan BET total {staked} token",
                text_color=C["primary"],
            )
        else:
            self.reco_lbl.configure(
                text="TIDAK ADA TARUHAN +EV  -  disarankan SKIP ronde ini (lindungi modal)",
                text_color=C["secondary"],
            )

        for w in self.cards_frame.winfo_children():
            w.destroy()
        for i, p in enumerate(allocs[:3]):
            self._create_pred_card(p, i)

    def _create_pred_card(self, p, index):
        border = ctk.CTkFrame(self.cards_frame, fg_color=C["secondary"], corner_radius=16)
        border.pack(fill="x", pady=6)
        card = ctk.CTkFrame(border, fg_color=C["card"], corner_radius=15)
        card.pack(fill="both", expand=True, padx=2, pady=2)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=18, pady=14)
        inner.grid_columnconfigure(1, weight=1)

        # Left: rank label + large number.
        numbox = ctk.CTkFrame(inner, fg_color="transparent")
        numbox.grid(row=0, column=0, rowspan=3, padx=(0, 20), sticky="w")
        ctk.CTkLabel(
            numbox, text=f"PREDIKSI #{index + 1}", font=font(11, "bold"),
            text_color=C["text_secondary"],
        ).pack(anchor="w")
        num_color = C["primary"] if p.get("is_positive_ev") else C["text_secondary"]
        ctk.CTkLabel(
            numbox, text=str(p["number"]), font=font(46, "bold"), text_color=num_color
        ).pack(anchor="w")

        # Right: confidence header + progress bar + token line.
        conf = max(0.0, min(1.0, float(p["confidence"])))
        conf_row = ctk.CTkFrame(inner, fg_color="transparent")
        conf_row.grid(row=0, column=1, sticky="ew")
        ctk.CTkLabel(
            conf_row, text="CONFIDENCE", font=font(10, "bold"), text_color=C["text_secondary"]
        ).pack(side="left")
        ctk.CTkLabel(
            conf_row, text=f"{conf * 100:.1f}%", font=font(12, "bold"), text_color=C["info"]
        ).pack(side="right")
        pb = ctk.CTkProgressBar(
            inner, progress_color=C["info"], fg_color=C["panel"], height=10, corner_radius=5
        )
        pb.set(conf)
        pb.grid(row=1, column=1, sticky="ew", pady=(3, 10))

        pot = settings.calculate_reward(p["token_bet"], p["number"]) - p["token_bet"]
        ev = p.get("ev_per_token")
        info = ctk.CTkFrame(inner, fg_color="transparent")
        info.grid(row=2, column=1, sticky="ew")
        left_txt = f"Token bet: {p['token_bet']}"
        if ev is not None:
            left_txt += f"   |   EV/token: {ev:+.2f}"
        ctk.CTkLabel(
            info, text=left_txt, font=font(13), text_color=C["text"]
        ).pack(side="left")
        ctk.CTkLabel(
            info, text=f"Potensi: +{pot}", font=font(14, "bold"), text_color=C["primary"]
        ).pack(side="right")

    def _on_confirm(self):
        actual = int(self.result_var.get())
        predicted = None
        profit_change = 0
        card_won = -1

        if self.current_predictions:
            # Total token yang dipertaruhkan di SEMUA angka yang disarankan.
            total_bet = sum(p["token_bet"] for p in self.current_predictions)
            for i, p in enumerate(self.current_predictions):
                if p["number"] == actual:
                    predicted = actual
                    # Payout angka menang sudah termasuk modal taruhan angka itu.
                    # Profit bersih = payout - SELURUH taruhan (termasuk yang kalah).
                    payout = settings.calculate_reward(p["token_bet"], actual)
                    profit_change = payout - total_bet
                    card_won = i
                    break
            if predicted is None:
                # Tidak ada angka yang kena: kehilangan seluruh taruhan.
                profit_change = -total_bet

        self._flash_cards(card_won)
        self.confirm_btn.configure(state="disabled", text="MENYIMPAN...")
        self.vm.process_new_actual(actual, predicted, profit_change, self._on_confirm_done)

    def _flash_cards(self, card_won):
        if not self.current_predictions:
            return
        for i, border in enumerate(self.cards_frame.winfo_children()):
            border.configure(fg_color=C["primary"] if i == card_won else C["error"])

        def reset_colors():
            for border in self.cards_frame.winfo_children():
                border.configure(fg_color=C["secondary"])

        self.after(1000, reset_colors)

    def _on_confirm_done(self):
        self.after(0, self._refresh_stats)
        self.after(0, self._refresh_header)
        self.after(0, self._refresh_live_lock)
        self.after(0, lambda: self.confirm_btn.configure(state="normal", text="KONFIRMASI HASIL"))

    # ------------------------------------------------------------------ #
    # Refresh
    # ------------------------------------------------------------------ #
    def _refresh_learning(self):
        """Repaint the live-learning panel from the continuous ensemble state."""
        if not hasattr(self, "learn_widgets"):
            return
        try:
            st = self.vm.get_learning_status()
        except Exception:
            return
        gpu = "aktif" if st.get("lstm_ready") else "memanas..."
        self.learn_info_lbl.configure(
            text=f"Belajar dari {st.get('n_observed', 0)} putaran  -  LSTM-GPU: {gpu}"
        )
        weights = st.get("weights", {})
        accs = st.get("accuracy", {})
        for key, (bar, acc) in self.learn_widgets.items():
            bar.set(max(0.0, min(1.0, float(weights.get(key, 0.0)))))
            a = accs.get(key)
            acc.configure(text=(f"acc {a*100:.0f}%" if a is not None else "acc -"))

    def _refresh_header(self):
        self.capital_lbl.configure(text=f"Modal: {self.vm.current_capital} token")

    def _refresh_stats(self):
        stats = self.vm.get_stats()
        self.stat_winrate.configure(text=f"{stats['win_rate']:.1f}%")
        self.stat_total.configure(text=f"{stats['total']}")

        profit = stats["profit"]
        sign = "+" if profit > 0 else ""
        if profit > 0:
            profit_color = C["primary"]
        elif profit < 0:
            profit_color = C["error"]
        else:
            profit_color = C["secondary"]
        self.stat_profit.configure(text=f"{sign}{profit}", text_color=profit_color)
        self.stat_streak.configure(text=f"{self.vm.tracker.get_streak()}x")

        adv = self.vm.get_advanced_stats()
        best = adv["best_round"]
        best_txt = f"+{best}" if best > 0 else str(best)
        self.adv_lbl.configure(
            text=(
                f"Best: {best_txt}     Worst: {adv['worst_round']}     "
                f"Avg/ronde: {adv['avg_profit']:.1f}     Max streak: {adv['max_streak']}x"
            )
        )

        if self.vm.latest_ev is not None:
            self.ev_lbl.configure(
                text=f"EV terakhir: {self.vm.latest_ev:.2f}   |   Prob: {self.vm.latest_prob * 100:.2f}%"
            )

        # Update the statistical wheel-bias verdict.
        try:
            bias = self.vm.get_bias_report()
            cmap = {
                "edge": C["primary"],
                "no_edge": C["secondary"],
                "insufficient": C["text_secondary"],
            }
            self.bias_lbl.configure(
                text=bias["message"],
                text_color=cmap.get(bias["status"], C["text_secondary"]),
            )
        except Exception:
            pass

        self._refresh_learning()

        # Cumulative profit chart.
        history = self.vm.tracker.data.get("history", [])
        self.ax.clear()
        self._style_axes()
        if history:
            cum, running = [], 0
            for h in history:
                running += h["profit_change"]
                cum.append(running)
            x = range(len(cum))
            line_color = C["primary"] if running >= 0 else C["error"]
            self.ax.plot(x, cum, color=line_color, linewidth=2)
            self.ax.fill_between(x, cum, color=line_color, alpha=0.18)
            self.ax.axhline(0, color=C["text_secondary"], linewidth=0.8, alpha=0.4)
        self.canvas.draw()

    # ------------------------------------------------------------------ #
    # Data actions
    # ------------------------------------------------------------------ #
    def _on_export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
        )
        if path:
            self.vm.tracker.export_csv(path)
            messagebox.showinfo("Export", f"Data berhasil diekspor ke:\n{path}")

    def _on_export_audit(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md")],
            initialfile="laporan_audit.md",
        )
        if not path:
            return
        try:
            files = self.vm.export_audit_report(path)
        except Exception as exc:
            messagebox.showerror("Export Audit", f"Gagal membuat laporan:\n{exc}")
            return
        listing = "\n".join(files)
        messagebox.showinfo(
            "Export Audit",
            "Laporan audit dibuat (Markdown + JSON + CSV mentah):\n" + listing,
        )

    def _on_reset(self):
        if not messagebox.askyesno(
            "Reset", "Yakin reset semua data? Tindakan ini tidak bisa dibatalkan."
        ):
            return
        self.vm.tracker.reset_data()
        self.current_predictions = []
        self.vm.current_capital = 1000
        self.modal_entry.delete(0, "end")
        self.modal_entry.insert(0, "1000")
        self._show_message_card("Belum ada prediksi", "Tekan HITUNG PREDIKSI untuk memulai.")
        self._refresh_stats()
        self._refresh_header()
