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

class MainWindow(ctk.CTk):
    def __init__(self, viewmodel: MainViewModel):
        super().__init__()
        self.vm = viewmodel
        
        self.title("Spin Wheel Predictor - MVVM Architecture")
        self.geometry("1100x800")
        self.configure(fg_color=settings.UI_COLORS["background"])
        
        # Grid layout: 2 rows (Content, Stats)
        self.grid_rowconfigure(0, weight=3) # Content (Header + Left + Right)
        self.grid_rowconfigure(1, weight=2) # Stats
        self.grid_columnconfigure(0, weight=1)
        
        self.current_predictions = []
        
        self._build_ui()
        self._refresh_stats()
        self._refresh_header()
        
    def _build_ui(self):
        # Header Bar
        self.header_frame = ctk.CTkFrame(self, fg_color=settings.UI_COLORS["panel"], corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="new")
        
        # Split Header
        self.title_lbl = ctk.CTkLabel(self.header_frame, text="🎡 SPIN WHEEL PREDICTOR", font=settings.FONTS["header"], text_color=settings.UI_COLORS["text"])
        self.title_lbl.pack(side="left", padx=20, pady=10)
        
        self.capital_lbl = ctk.CTkLabel(self.header_frame, text="💰 Modal: - token", font=settings.FONTS["subheader"], text_color=settings.UI_COLORS["secondary"])
        self.capital_lbl.pack(side="left", padx=20)
        
        # GPU Indicator
        if getattr(self.vm, 'gpu_available', False):
            gpu_text = "⚡ GPU Mode"
            gpu_color = settings.UI_COLORS["primary"]
        else:
            gpu_text = "🖥️ CPU Mode"
            gpu_color = settings.UI_COLORS["text_secondary"]
        self.gpu_lbl = ctk.CTkLabel(self.header_frame, text=gpu_text, font=settings.FONTS["body"], text_color=gpu_color)
        self.gpu_lbl.pack(side="left", padx=20)
        
        self.engine_var = ctk.StringVar(value="TF-LSTM")
        self.engine_dropdown = ctk.CTkOptionMenu(
            self.header_frame, values=["TF-LSTM", "Heuristic"], 
            variable=self.engine_var,
            command=self._on_engine_change,
            fg_color=settings.UI_COLORS["card"], button_color=settings.UI_COLORS["secondary"], button_hover_color=settings.UI_COLORS["info"]
        )
        self.engine_dropdown.pack(side="right", padx=20)
        
        # Separator line under header
        self.sep = ctk.CTkFrame(self, fg_color=settings.UI_COLORS["secondary"], height=2, corner_radius=0)
        self.sep.grid(row=0, column=0, sticky="new", pady=(50, 0)) 
        
        # Content Frame
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=0, column=0, sticky="nsew", pady=(52, 0))
        self.content_frame.grid_columnconfigure(0, weight=38) # 38% Left
        self.content_frame.grid_columnconfigure(1, weight=62) # 62% Right
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        self._build_input_zone()
        self._build_prediction_zone()
        self._build_stats_zone()

    def _build_input_zone(self):
        left = ctk.CTkFrame(self.content_frame, fg_color=settings.UI_COLORS["panel"], corner_radius=15)
        left.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        
        # Modal Token Input
        ctk.CTkLabel(left, text="💰 Modal Token", font=settings.FONTS["subheader"], text_color=settings.UI_COLORS["text"]).pack(pady=(16, 5))
        modal_f = ctk.CTkFrame(left, fg_color="transparent")
        modal_f.pack()
        self.modal_entry = ctk.CTkEntry(modal_f, width=110, font=settings.FONTS["body"], justify="center")
        self.modal_entry.insert(0, str(self.vm.current_capital))
        self.modal_entry.pack(side="left", padx=5)
        ctk.CTkButton(modal_f, text="💾 Set", width=60, command=self._on_set_modal,
                      fg_color=settings.UI_COLORS["card"], hover_color=settings.UI_COLORS["secondary"],
                      text_color=settings.UI_COLORS["text"]).pack(side="left")
        
        # Manual % Input
        ctk.CTkLabel(left, text="📊 % Kemunculan Manual", font=settings.FONTS["subheader"], text_color=settings.UI_COLORS["text"]).pack(pady=(16, 5))
        
        perc_frame = ctk.CTkFrame(left, fg_color="transparent")
        perc_frame.pack(fill="x", padx=10)
        
        self.perc_entries = {}
        row, col = 0, 0
        for num in settings.VALID_NUMBERS:
            f = ctk.CTkFrame(perc_frame, fg_color="transparent")
            f.grid(row=row, column=col, padx=2, pady=2)
            ctk.CTkLabel(f, text=str(num), font=settings.FONTS["small"]).pack()
            e = ctk.CTkEntry(f, width=45, font=settings.FONTS["small"], justify="center")
            e.insert(0, "0")
            e.pack()
            e.bind("<KeyRelease>", self._update_perc_total)
            self.perc_entries[num] = e
            col += 1
            if col > 4:
                col = 0
                row += 1
                
        self.perc_total_lbl = ctk.CTkLabel(left, text="Total: 0%", text_color=settings.UI_COLORS["error"], font=settings.FONTS["small"])
        self.perc_total_lbl.pack()
        
        btn_auto = ctk.CTkButton(left, text="⚡ Auto dari Wheel", command=self._auto_fill_perc, 
                                 fg_color=settings.UI_COLORS["card"], hover_color=settings.UI_COLORS["secondary"], text_color=settings.UI_COLORS["text"])
        btn_auto.pack(pady=5)
        
        # History Selector
        ctk.CTkLabel(left, text="📜 Riwayat Dianalisis", font=settings.FONTS["subheader"], text_color=settings.UI_COLORS["text"]).pack(pady=(16, 5))
        self.hist_var = ctk.StringVar(value="100 Terakhir")
        self.hist_seg = ctk.CTkSegmentedButton(left, values=["10 Terakhir", "50 Terakhir", "100 Terakhir"], 
                                               variable=self.hist_var, command=self._on_hist_change,
                                               selected_color=settings.UI_COLORS["secondary"], selected_hover_color=settings.UI_COLORS["info"])
        self.hist_seg.pack()
        
        # Risk Slider
        self.risk_lbl = ctk.CTkLabel(left, text="📉 Budget Taruhan: 30% dari modal", font=settings.FONTS["body"])
        self.risk_lbl.pack(pady=(16, 0))
        self.risk_slider = ctk.CTkSlider(left, from_=10, to=50, number_of_steps=40, command=self._on_risk_change,
                                         progress_color=settings.UI_COLORS["secondary"], button_color=settings.UI_COLORS["secondary"], button_hover_color=settings.UI_COLORS["info"])
        self.risk_slider.set(30)
        self.risk_slider.pack(pady=5)
        
        # EV Label
        self.ev_lbl = ctk.CTkLabel(left, text="EV: - | Prob: -%", font=settings.FONTS["small"], text_color=settings.UI_COLORS["text_secondary"])
        self.ev_lbl.pack(pady=5)
        
        # Hitung Button
        self.hitung_btn = ctk.CTkButton(left, text="🔮 HITUNG PREDIKSI", font=settings.FONTS["header"], height=50,
                                        command=self._on_hitung, fg_color=settings.UI_COLORS["secondary"], text_color=settings.UI_COLORS["background"], hover_color=settings.UI_COLORS["info"])
        self.hitung_btn.pack(pady=(10, 16), padx=16, fill="x", side="bottom")

    def _build_prediction_zone(self):
        self.right = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.right.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        
        # Prediction Cards Container
        self.cards_frame = ctk.CTkFrame(self.right, fg_color="transparent")
        self.cards_frame.pack(fill="both", expand=True)
        
        # Confirm Result Zone
        self.confirm_frame = ctk.CTkFrame(self.right, fg_color=settings.UI_COLORS["panel"], corner_radius=15)
        self.confirm_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(self.confirm_frame, text="── Input Hasil ──", font=settings.FONTS["subheader"]).pack(pady=5)
        
        inp_f = ctk.CTkFrame(self.confirm_frame, fg_color="transparent")
        inp_f.pack(pady=5)
        
        self.result_var = ctk.StringVar(value=str(settings.VALID_NUMBERS[0]))
        ctk.CTkOptionMenu(inp_f, values=[str(n) for n in settings.VALID_NUMBERS], variable=self.result_var).pack(side="left", padx=10)
        
        self.confirm_btn = ctk.CTkButton(inp_f, text="✅ KONFIRMASI HASIL", command=self._on_confirm, 
                                         fg_color=settings.UI_COLORS["primary"], text_color=settings.UI_COLORS["background"], hover_color="#32E011")
        self.confirm_btn.pack(side="left", padx=10)

    def _build_stats_zone(self):
        stats_bg = ctk.CTkFrame(self, fg_color=settings.UI_COLORS["panel"], corner_radius=0)
        stats_bg.grid(row=1, column=0, sticky="nsew")
        stats_bg.grid_columnconfigure(0, weight=1)
        stats_bg.grid_rowconfigure(1, weight=1)
        
        # Stat cards row
        cards_f = ctk.CTkFrame(stats_bg, fg_color="transparent")
        cards_f.grid(row=0, column=0, sticky="ew", padx=16, pady=10)
        
        self.stat_winrate = self._make_stat_card(cards_f, "WIN RATE", "0.0%")
        self.stat_total = self._make_stat_card(cards_f, "TOTAL", "0 game")
        self.stat_profit = self._make_stat_card(cards_f, "PROFIT", "0 tok")
        self.stat_streak = self._make_stat_card(cards_f, "STREAK", "🔥 0x")
        
        # Chart
        chart_f = ctk.CTkFrame(stats_bg, fg_color="transparent")
        chart_f.grid(row=1, column=0, sticky="nsew", padx=16)
        
        self.fig = Figure(figsize=(8, 2), facecolor=settings.UI_COLORS["panel"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(settings.UI_COLORS["panel"])
        self.ax.tick_params(colors=settings.UI_COLORS["text_secondary"])
        # Remove borders
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.grid(color=settings.UI_COLORS["text_secondary"], linestyle='--', alpha=0.2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_f)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Bottom Actions
        actions_f = ctk.CTkFrame(stats_bg, fg_color="transparent")
        actions_f.grid(row=2, column=0, sticky="ew", padx=16, pady=10)
        
        self.adv_lbl = ctk.CTkLabel(actions_f, text="Best: 0  |  Worst: 0  |  Avg/ronde: 0.0  |  Max streak: 0",
                                    font=settings.FONTS["small"], text_color=settings.UI_COLORS["text_secondary"])
        self.adv_lbl.pack(side="left", padx=5)
        
        ctk.CTkButton(actions_f, text="🔄 Reset Semua", command=self._on_reset, fg_color=settings.UI_COLORS["error"], hover_color="#CC0000").pack(side="right", padx=5)
        ctk.CTkButton(actions_f, text="📊 Export CSV", command=self._on_export, fg_color=settings.UI_COLORS["card"], hover_color=settings.UI_COLORS["info"]).pack(side="right", padx=5)

    def _make_stat_card(self, parent, title, val):
        f = ctk.CTkFrame(parent, fg_color=settings.UI_COLORS["card"], corner_radius=10)
        f.pack(side="left", expand=True, fill="x", padx=5)
        ctk.CTkLabel(f, text=title, font=settings.FONTS["small"], text_color=settings.UI_COLORS["text_secondary"]).pack(pady=(10, 0))
        lbl = ctk.CTkLabel(f, text=val, font=settings.FONTS["subheader"], text_color=settings.UI_COLORS["text"])
        lbl.pack(pady=(0, 10))
        return lbl

    def _update_perc_total(self, event=None):
        total = 0.0
        for num, e in self.perc_entries.items():
            try:
                val = float(e.get())
                self.vm.manual_percentages[num] = val
                total += val
            except ValueError:
                pass
        
        color = settings.UI_COLORS["primary"] if abs(total - 100) < 0.01 else settings.UI_COLORS["error"]
        self.perc_total_lbl.configure(text=f"Total: {total:.1f}%", text_color=color)

    def _auto_fill_perc(self):
        seq = settings.SPINWHEEL_SEQUENCE
        counts = Counter(seq)
        total_len = len(seq)
        for num in settings.VALID_NUMBERS:
            prob = (counts[num] / total_len) * 100
            self.perc_entries[num].delete(0, "end")
            self.perc_entries[num].insert(0, f"{prob:.2f}")
        self._update_perc_total()

    def _on_hist_change(self, val):
        length = int(val.split()[0])
        self.vm.history_length = length

    def _on_risk_change(self, val):
        risk = int(val)
        self.vm.risk_percentage = risk / 100.0
        self.risk_lbl.configure(text=f"📉 Budget Taruhan: {risk}% dari modal")

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

    def _on_hitung(self):
        self.hitung_btn.configure(text="⏳ Menghitung...", state="disabled")
        # Ensure percentages are updated
        self._update_perc_total()
        
        # Run prediction OFF the UI thread so the window never freezes
        self.vm.get_predictions_async(
            lambda allocs: self.after(0, self._display_predictions, allocs)
        )
        
    def _display_predictions(self, allocs=None):
        self.hitung_btn.configure(text="🔮 HITUNG PREDIKSI", state="normal")
        
        for w in self.cards_frame.winfo_children():
            w.destroy()
            
        if allocs is None:
            allocs = []
        self.current_predictions = allocs
        
        if not allocs:
            ctk.CTkLabel(self.cards_frame, text="Tidak ada data atau error engine.", font=settings.FONTS["body"]).pack(pady=20)
            return
            
        for i, p in enumerate(allocs[:3]): # Show top 3
            self._create_pred_card(p, i)

    def _create_pred_card(self, p, index):
        # Card with gold border simulation
        card_border = ctk.CTkFrame(self.cards_frame, fg_color=settings.UI_COLORS["secondary"], corner_radius=16)
        card_border.pack(fill="x", pady=5, padx=10)
        
        card = ctk.CTkFrame(card_border, fg_color=settings.UI_COLORS["card"], corner_radius=15)
        card.pack(fill="both", expand=True, padx=1, pady=1) # 1px border
        
        # Title
        ctk.CTkLabel(card, text=f"🎯 PREDIKSI #{index+1}", font=settings.FONTS["small"], text_color=settings.UI_COLORS["text_secondary"]).pack(pady=(5,0))
        
        # Number
        ctk.CTkLabel(card, text=str(p['number']), font=("Inter", 48, "bold"), text_color=settings.UI_COLORS["primary"]).pack()
        
        # Progress Bar
        conf = p['confidence']
        pb_f = ctk.CTkFrame(card, fg_color="transparent")
        pb_f.pack(fill="x", padx=20)
        pb = ctk.CTkProgressBar(pb_f, progress_color=settings.UI_COLORS["info"], fg_color=settings.UI_COLORS["panel"], height=10)
        pb.set(conf)
        pb.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(pb_f, text=f"{conf*100:.1f}%", font=settings.FONTS["small"]).pack(side="right", padx=(10, 0))
        
        # Tokens
        pot = settings.calculate_reward(p['token_bet'], p['number']) - p['token_bet']
        info = f"Token: {p['token_bet']} → Potensi: +{pot}"
        ctk.CTkLabel(card, text=info, font=settings.FONTS["body"], text_color=settings.UI_COLORS["text"]).pack(pady=10)

    def _on_confirm(self):
        actual = int(self.result_var.get())
        
        predicted = None
        profit_change = 0
        card_won = -1
        
        if self.current_predictions:
            for i, p in enumerate(self.current_predictions):
                if p["number"] == actual:
                    predicted = actual
                    profit_change = settings.calculate_reward(p["token_bet"], actual) - p["token_bet"]
                    card_won = i
                    break
            
            if predicted is None:
                profit_change = -sum(p["token_bet"] for p in self.current_predictions)
        
        # Animations
        self._flash_cards(card_won)
        
        self.confirm_btn.configure(state="disabled", text="⏳ Menyimpan...")
        self.vm.process_new_actual(actual, predicted, profit_change, self._on_confirm_done)
        
    def _flash_cards(self, card_won):
        children = self.cards_frame.winfo_children()
        for i, card_border in enumerate(children):
            if i == card_won:
                card_border.configure(fg_color=settings.UI_COLORS["primary"])
            else:
                card_border.configure(fg_color=settings.UI_COLORS["error"])
                
        def reset_colors():
            for card_border in self.cards_frame.winfo_children():
                card_border.configure(fg_color=settings.UI_COLORS["secondary"])
                
        self.after(1000, reset_colors)

    def _on_confirm_done(self):
        self.after(0, self._refresh_stats)
        self.after(0, self._refresh_header)
        self.after(0, lambda: self.confirm_btn.configure(state="normal", text="✅ KONFIRMASI HASIL"))

    def _refresh_header(self):
        self.capital_lbl.configure(text=f"💰 Modal: {self.vm.current_capital} token")

    def _refresh_stats(self):
        stats = self.vm.get_stats()
        self.stat_winrate.configure(text=f"{stats['win_rate']:.1f}%")
        self.stat_total.configure(text=f"{stats['total']} game")
        sign = "+" if stats['profit'] > 0 else ""
        self.stat_profit.configure(text=f"{sign}{stats['profit']} tok")
        self.stat_streak.configure(text=f"🔥 {self.vm.tracker.get_streak()}x")
        
        adv = self.vm.get_advanced_stats()
        self.adv_lbl.configure(
            text=f"Best: +{adv['best_round']}  |  Worst: {adv['worst_round']}  |  "
                 f"Avg/ronde: {adv['avg_profit']:.1f}  |  Max streak: {adv['max_streak']}x"
        )
        
        if self.vm.latest_ev is not None:
            self.ev_lbl.configure(text=f"Last Actual EV: {self.vm.latest_ev:.2f} | Probabilitas: {self.vm.latest_prob*100:.2f}%")
        
        history = self.vm.tracker.data.get("history", [])
        self.ax.clear()
        
        # Re-apply chart settings
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.grid(color=settings.UI_COLORS["text_secondary"], linestyle='--', alpha=0.2)
        
        if history:
            profits = [h["profit_change"] for h in history]
            cum_profits = []
            c = 0
            for p in profits:
                c += p
                cum_profits.append(c)
                
            x = range(len(cum_profits))
            self.ax.plot(x, cum_profits, color=settings.UI_COLORS["primary"], linewidth=2)
            self.ax.fill_between(x, cum_profits, color=settings.UI_COLORS["primary"], alpha=0.2)
            
        self.canvas.draw()

    def _on_export(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if path:
            self.vm.tracker.export_csv(path)
            messagebox.showinfo("Export", f"Data berhasil diexport ke:\n{path}")

    def _on_reset(self):
        ans = messagebox.askyesno("Reset", "Yakin reset semua data? Tindakan ini tidak bisa dibatalkan.")
        if ans:
            self.vm.tracker.reset_data()
            self.current_predictions = []
            for w in self.cards_frame.winfo_children():
                w.destroy()
            self.vm.current_capital = 1000
            self.modal_entry.delete(0, "end")
            self.modal_entry.insert(0, "1000")
            self._refresh_stats()
            self._refresh_header()
