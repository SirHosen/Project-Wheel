# Experimental / not wired into the live app

Modul & kelas di bawah ini **sengaja disimpan untuk eksperimen** tapi **tidak
tersambung ke alur aplikasi utama** (audit V3, bagian "dead/redundan"). Mereka
lolos test dan aman dipakai dari script/REPL, tapi jangan dianggap memengaruhi
taruhan di GUI.

| Item | Lokasi | Status |
|------|--------|--------|
| `AdaptiveRiskManager` | `core/risk_manager.py` | Tidak dipanggil ViewModel. Live app pakai `core/betting.py` (`net_kelly_portfolio`). Disimpan untuk eksperimen money-management. |
| `GamblersFallacySimulator` | `core/wheel_math.py` | Simulator edukasi, tidak dipakai di alur app. (`WheelMath` di file yang sama TETAP dipakai untuk EV.) |
| `HigherOrderMarkovEngine` | `predictors/higher_order_markov.py` | **Alat analisis saja.** Tidak diekspos di dropdown engine karena backtest jujur menunjukkan ia tidak lebih baik dari baseline frekuensi pada roda i.i.d. Dipakai di `scripts/run_backtest.py` & diagnostics. |
| `predictors/heuristic_engine.py` | root `predictors/` | Shim deprecated yang re-export `predictors/legacy/`. Dipertahankan untuk kompatibilitas import lama. |

## Kenapa tidak dihapus saja?

Kamu memang ingin terus bereksperimen, jadi modul-modul ini dipertahankan dan
DITANDAI jelas (`[EXPERIMENTAL / NOT WIRED]` di docstring) alih-alih dibuang.
Kalau suatu saat ingin mengaktifkan salah satunya, sambungkan ke
`gui/viewmodels/main_viewmodel.py` dan tambahkan ke dropdown di
`gui/views/main_window.py`.
