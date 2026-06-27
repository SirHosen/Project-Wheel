# Cleanup + Reset History + Bugfix (v1.30.1)

## Apa yang berubah
1. **Satu folder runtime/** untuk SEMUA data yang dihasilkan saat app jalan:
   riwayat spin (`history.json`/`.db`), state belajar (`continuous_state.json`),
   kalibrasi (`calibration_state.json`), risk (`risk_state.json`), model LSTM
   (`lstm_spinwheel.keras`/`.h5`), log vision (`vision_observations.csv`), dan
   hasil export. Semua path default sekarang diarahkan ke `runtime/` lewat
   `config/settings.py`.
2. **`.gitignore` dirapikan** -> seluruh ISI `runtime/` di-ignore (cuma
   `runtime/.gitkeep` yang dilacak), plus nutup sisa noise lama
   (`data/*.json`, `models/`, `history_export.json`, `grafik_audit_*.png`,
   `data/*.premigrate.bak`, `reports/`). Repo selalu bersih.
3. **Bugfix desync di `reset_data()`** -> dulu "Reset Data" cuma mengosongkan
   riwayat spin, tapi state belajar (n_observed dll) ketinggalan -> desync.
   Sekarang reset ikut purge continuous/calibration/risk + model LSTM, jadi
   otak app benar-benar balik ke nol.
4. **Semua artefak lama dihapus** dari working tree (history, model, chart audit,
   reports, backup).

## File yang berubah
- config/settings.py
- data/tracker.py  (reset_data fix + default path + _purge_learning_state)
- core/calibration.py
- core/risk_manager.py
- vision/observation_log.py
- .gitignore
- runtime/.gitkeep (baru)

## Catatan
- 29/29 test tetap PASS (test pakai path eksplisit, jadi ganti default aman).
- Data lama lo TIDAK ikut dipindah otomatis. Kalau mau pertahankan riwayat lama,
  copy manual file lama ke runtime/ sebelum jalan. Kalau mau fresh, biarkan saja.
