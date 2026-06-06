# Changelog — Enhancement Pass oleh Notion AI

## Diagnosis awal
- Tidak ada syntax error; seluruh modul lolos kompilasi.
- Logika inti (wheel_math, heuristic_engine, token allocation, reward) sudah BENAR & sudah dites.
- Versi ini ~90% sesuai spec enhance; bukan "fitur hilang" tapi beberapa perbaikan kualitas.

## Perubahan yang diterapkan
### 1. Prediksi tidak lagi nge-freeze UI (data/viewmodel)
- `MainViewModel.get_predictions_async(callback)`: prediksi (termasuk TF-LSTM) kini jalan di thread terpisah.
- `MainWindow._on_hitung` memanggil versi async; hasil dirender via `after(0, ...)`.
  Sebelumnya prediksi LSTM bisa nge-hang window di tombol "Menghitung...".

### 2. Field Modal Token (fitur baru di UI)
- Input "Modal Token" + tombol Set di zona kiri.
- `MainViewModel.set_initial_capital(amount)` menyimpan modal awal ke history.json.
- Validasi: hanya menerima angka > 0.

### 3. Analitik lebih lengkap (data/tracker.py)
- `get_max_win_streak()`, `get_number_frequency()`, `get_advanced_stats()`.
- Baris analitik baru di dashboard: Best / Worst / Avg-per-ronde / Max streak.

## Catatan penting
- customtkinter & tensorflow TIDAK terpasang di lingkungan build, jadi tampilan GUI
  belum diverifikasi visual. Logika & analitik sudah dites.
- Jalankan lokal: `pip install -r requirements.txt` lalu `python app/main.py`.
