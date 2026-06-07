# Changelog — Enhancement Pass oleh Notion AI

## v1.6.0 — Pembelajaran berkesinambungan + UI live-learning (oleh Mahapatih)
- **Otak Ensemble kontinu (`core/continuous_engine.py`)**: satu mesin yang menyatukan 4 sinyal — Fisika (area-fraction = kebenaran roda adil), Bayes (posterior Dirichlet dari hasil live, di-seed prior fisika), Markov (transisi), dan TF-LSTM (GPU). Menghasilkan SATU distribusi probabilitas terpadu.
- **Belajar bobot tiap putaran**: setiap `observe(actual)` menilai top-pick tiap model, memperbarui akurasi walk-forward (EMA), lalu menata ulang bobot blend via softmax. Sistem benar-benar "belajar harus percaya sinyal mana" secara kontinu.
- **Kontinu antar-sesi**: state belajar (skor + jumlah putaran) dipersist ke `models/continuous_state.json`, jadi tidak mulai dari nol tiap buka app. GPU LSTM dilatih inkremental tiap putaran & disimpan periodik (GPU tetap hangat).
- **Engine default = "Ensemble"** di dropdown UI; viewmodel memanggil `continuous.observe()` di `process_new_actual`.
- **UI**: panel "Pembelajaran Live (Ensemble)" — bar bobot + akurasi tiap model (Fisika/Bayes/Markov/LSTM-GPU) + jumlah putaran dipelajari + status GPU, refresh tiap konfirmasi hasil.
- **Uji**: `_test_continuous.py` (prior fisika, konvergensi Bayes, distribusi ensemble valid, update bobot, persistensi) — semua LULUS.
- **Catatan jujur**: pada roda adil, semua model konvergen ke baseline frekuensi yang sama, jadi bobot tetap seimbang & ensemble ≈ area-fraction. Mesinnya nyata; ia tidak bisa menciptakan edge yang tidak dimiliki roda.

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

## UI Overhaul (rewrite main_window.py)
- HAPUS SEMUA EMOJI dari UI & kode (penyebab "icon tidak ke-load"/kotak tofu di Tkinter).
  Diverifikasi via scan otomatis: 0 emoji tersisa.
- Indikator GPU/CPU kini pakai titik berwarna + label "GPU MODE"/"CPU MODE" (bukan emoji).
- Layout grid dirapikan total: header (tinggi tetap) / konten / statistik jadi 3 baris terpisah,
  hilangkan trik offset piksel yang bikin header numpuk dengan konten.
- Header profesional: judul + garis bawah emas tipis, selector engine berlabel.
- Panel input kiri kini scrollable (kontrol tidak akan terpotong) + tombol CTA dipin di bawah.
- Kartu prediksi didesain ulang: angka besar + bar confidence + baris token/potensi yang rapi.
- Kartu statistik dengan aksen warna; chart profit kumulatif lebih bersih (garis hijau/merah,
  garis nol, area fill halus).
- Teks tombol konsisten huruf kapital tanpa emoji (HITUNG PREDIKSI, KONFIRMASI HASIL, dst).
- Judul window diganti dari "...MVVM Architecture" menjadi "Spin Wheel Predictor".

## Fix: % manual kini berlaku untuk SEMUA engine
- Sebelumnya input % manual hanya dipakai engine Heuristik; padahal default engine = TF-LSTM,
  jadi user yang membiarkan default tidak melihat efek apa pun saat mengisi %.
- Sekarang % manual diperlakukan sebagai PRIOR dan di-blend (bobot 0.35) ke output
  engine mana pun yang aktif. Terbukti mengubah ranking prediksi.

## Fix KRUSIAL: perhitungan profit saat menang
- BUG: saat salah satu angka prediksi menang, kode hanya mengurangi modal taruhan
  ANGKA YANG MENANG, bukan total seluruh taruhan. Token yang kalah di angka lain
  tidak ikut dikurangi -> modal jadi lebih besar dari seharusnya.
- FIX: profit_change = payout(angka menang) - TOTAL seluruh taruhan.
- Contoh: modal 10, bet {1:2, 2:1, 5:1} (total 4), angka 2 keluar.
  payout = 1*2+1 = 3 -> profit = 3-4 = -1 -> modal akhir = 9 token. (BENAR)

## Engine baru: Markov / Transisi (default) - untuk win rate lebih tinggi
- predictors/markov_engine.py: belajar P(next | angka terakhir) dari riwayat hasil.
  Smoothing aditif menarik ke prior frekuensi roda saat data sedikit, jadi:
    * data sedikit  -> setara "angka tersering" (top-3 hit tertinggi)
    * data banyak   -> menangkap korelasi antar-spin (bias posisi) bila ada
- Dijadikan engine DEFAULT (paling robust). Ditambahkan ke dropdown ENGINE di UI.
- Backtest (3000 spin):
    Skenario ACAK murni : Heuristik top3=56.5% | Markov top3=72.7%
    Skenario BERBIAS    : Heuristik top3=67.1% | Markov top3=76.9%

## Update: EV-aware betting + statistical bias detection

### 1. core/betting.py (BARU) - mesin keputusan profit
- `kelly_allocation()`: ubah confidence engine jadi keputusan taruhan yang memaksimalkan EXPECTED PROFIT, bukan asal bet top-3.
  - Hitung EV/token tiap angka: `EV = p*(m+1) - 1` (m = net multiplier; n untuk n>=2, 1 untuk n=1). Break-even = `1/(m+1)`.
  - Cuma stake angka **+EV**; sizing pakai **half-Kelly** (`f = p - (1-p)/m`), di-cap ke risk budget (risk% x modal).
  - Kalau tidak ada angka +EV -> rekomendasi **SKIP** (semua stake 0). Di roda fair, ini memang langkah optimal (semua bet -EV).
- Dipakai di viewmodel.get_predictions() menggantikan get_token_allocation flat-by-confidence.

### 2. Deteksi bias statistik - viewmodel.get_bias_report()
- Walk-forward: bandingkan akurasi top-1 engine Markov vs baseline 'selalu angka tersering' di seluruh riwayat.
- Uji z dua-proporsi; verdict EDGE (z>1.96) / belum terbukti / data kurang (min 40 sampel).
- Memberi tahu user KAPAN model benar-benar mengalahkan peluang (= bias roda terbukti & layak dibet besar).

### 3. UI (main_window.py)
- Banner rekomendasi BET/SKIP di atas kartu prediksi (hijau = ada edge +EV, emas = skip).
- Tiap kartu: angka berwarna hijau kalau +EV / abu-abu kalau -EV, plus baris `EV/token`.
- Label verdict 'Bias roda' di zona statistik, update tiap konfirmasi hasil.

### Cross-check (_test_betting_bias.py) - SEMUA PASSED
- EV/break-even tiap angka konsisten dengan calculate_reward; roda fair -> semua -EV.
- Fair-wheel alloc -> SKIP; bias kuat -> BET half-Kelly dalam budget; longshot +EV terdeteksi.
- Bias walk-forward: data random -> no edge (z=-0.60); data autocorrelated -> EDGE (z=11.43).

## Update: evidence gate (berdasarkan analisis 117 putaran data asli user)

### Temuan dari data nyata (1.csv, 117 putaran, modal 20 -> ~73)
- Distribusi angka ~ identik dgn frekuensi roda (chi2=12.97 < 15.51) -> tidak ada bias frekuensi.
- Lag-1 'angka sama' 23.3% vs harapan acak 22.6% -> tidak ada autocorrelation.
- Markov top-1 35.5% vs baseline 'selalu 1' 36.4% (z=-0.14) -> Markov TIDAK memberi edge; roda tampak fair.
- 52/110 ronde ter-flag +EV tapi hanya 34.6% benar -> confidence over-pede dari sampel kecil = sumber taruhan +EV palsu yang membocorkan modal.
- Profit +53 user berasal dari taruhan kecil/skip + variance, bukan edge.

### Perubahan
- predictors/markov_engine.py: tiap prediksi kini membawa `support` = jumlah transisi teramati dari last-number (0 untuk fallback frekuensi).
- core/betting.py: kelly_allocation menambah `min_support` (default 10) + `ev_margin` (default 0.10).
  - Sebuah angka hanya boleh dipertaruhkan jika EV > margin DAN support >= min_support.
  - Prediksi tanpa field `support` (engine lain) tidak di-gate (kompatibel mundur).
- UI tetap: angka hijau = actionable +EV (lolos gate), abu-abu = belum.

### Simulasi bankroll pada urutan ASLI user (modal 20, auto-bet tiap rekomendasi)
- Lama (tanpa gate): berakhir 3 token (47 ronde bet).
- Baru (gate support>=10): berakhir 10 token (50 ronde bet) -> bleeding lebih lambat.
- Kesimpulan jujur: pada roda fair, mengejar +EV tetap rugi; gate memperlambat kerugian, tidak menjadikan profit. Proteksi terbaik = taruh kecil/skip sampai detektor bias menyatakan EDGE.

### Regression: _test_betting_bias.py tetap ALL PASSED.

## Update: Live Adaptive Prior (persen manual yang update otomatis tiap putaran)

### Permintaan user
Persen manual yang diinput di awal harus auto-update tiap putaran; ada tombol untuk
'mengunci' lalu update otomatis tiap game; pakai konteks 54 angka di roda; UI lebih responsif.

### Logika
- Tombol baru: 'KUNCI & AUTO-UPDATE' (di bawah panel Probabilitas Manual).
- Saat dikunci, persen yang sudah diinput dibekukan jadi PRIOR Bayesian senilai 1 roda penuh
  (54 segmen). Tiap konfirmasi hasil, persen di-recompute:
      live%[n] = (prior_counts[n] + jumlah_kemunculan_nyata[n]) / (54 + total_putaran) * 100
  Jadi awalnya menghormati input user, lalu makin lama makin nyimpang ke frekuensi roda yang
  benar-benar terjadi. Persen di kotak input ikut ter-update otomatis (read-only saat terkunci).
- Jika user mengunci tanpa mengisi apa pun, prior di-seed dari layout fisik roda 54 segmen.
- Tombol bisa di-'BUKA KUNCI' untuk balik ke mode manual statis.
- Persen live tetap di-blend ke engine (bobot 0.35) seperti sebelumnya, jadi prediksi ikut adaptif.

### Implementasi
- gui/viewmodels/main_viewmodel.py: state manual_locked + manual_prior_counts +
  WHEEL_PRIOR_STRENGTH(54); method lock_manual_percentages / unlock_manual_percentages /
  refresh_live_percentages; dipanggil otomatis di process_new_actual tiap hasil baru.
- gui/views/main_window.py: tombol lock + label status; _on_toggle_lock, _apply_live_percentages,
  _refresh_live_lock; auto-refresh entri persen di _on_confirm_done.

### UI lebih responsif
- minsize diturunkan 1040x760 -> 900x640; resizable eksplisit.
- Area kartu prediksi kini dalam CTkScrollableFrame (kartu tidak terpotong di layar kecil),
  dengan inner frame agar logika flash/destroy tetap utuh.

### Catatan kejujuran
Rodanya teruji fair (lihat analisis 117 putaran), jadi fitur ini membuat persen yang ditampilkan
AKURAT mengikuti realita, tapi tidak bisa mendorong winrate jauh di atas ~40% (angka 1 muncul ~37%).
Leverage profit yang nyata tetap di disiplin taruhan (bet kecil/SKIP sampai detektor bias = EDGE).

### Test
- _test_live_prior.py: ALL PASSED (persen konvergen dari input -> frekuensi roda nyata).
- _test_betting_bias.py (regression): tetap ALL PASSED.

## Update: Audit & Diagnostics Engine (laporan akurat untuk upgrade/audit)

### Tujuan
Menyediakan laporan yang jauh lebih akurat & detail dari sekadar export CSV, supaya tiap
permintaan upgrade/update/audit punya dasar bukti yang presisi.

### Baru: core/diagnostics.py (pure stdlib, tanpa TF/CustomTkinter)
Menghitung dari history.json:
- Metadata sesi + APP_VERSION (config/settings.py) untuk ketertelusuran.
- Bankroll: modal awal/sekarang, profit terealisasi, ROI, max drawdown (token & %).
- Performa: win-rate tebakan teratas, win-rate saat bertaruh, rasio skip, profit per ronde bertaruh.
- Statistik per ronde: terbaik/terburuk/rata-rata/standar deviasi.
- Streak: menang sekarang, max menang, max kalah beruntun.
- Tabel per angka: observasi % vs frekuensi teori roda (54 segmen) + deviasi.
- Uji kewajaran roda: chi-square goodness-of-fit + nilai kritis 0.05 + putusan fair/biased.
- Autokorelasi lag-1 (angka berturut sama) vs harapan acak.
- Edge Markov walk-forward: akurasi top-1 vs baseline + z-score + putusan edge/no_edge.
- Flag kualitas data + rekomendasi (mis. log snapshot prediksi untuk audit taruhan per-angka).
Ekspor bundle: <nama>.md (untuk dibaca) + <nama>.json (untuk audit presisi) + <nama>_raw.csv.

### Wiring
- gui/viewmodels/main_viewmodel.py: export_audit_report(md_path) -> memanggil export_audit_bundle.
- gui/views/main_window.py: tombol 'EXPORT LAPORAN AUDIT' di samping EXPORT CSV; dialog simpan .md;
  pesan sukses menampilkan ketiga file yang dibuat.
- config/settings.py: APP_VERSION = "1.3.0".

### Test
- _test_audit.py: ALL PASSED (120 ronde sintetis -> laporan lengkap, struktur tervalidasi).
- _test_live_prior.py & _test_betting_bias.py (regression): tetap ALL PASSED.

## Update v1.4.0: TF-LSTM jadi model GPU sungguhan (RTX 3080)

### Tujuan
Membuat TensorFlow benar-benar dimanfaatkan di PC ber-GPU (Ryzen + RTX 3080),
bukan sekadar model kecil yang tidak terpakai.

### Perubahan predictors/tf_lstm_engine.py (interface dijaga 100% kompatibel)
- Input pakai EMBEDDING kategori (bukan lagi dibagi N) -> representasi angka jauh lebih baik.
- Arsitektur lebih dalam & bisa dikonfigurasi: stacked LSTM [128, 64] + Dense 64 + softmax.
- Mixed precision float16 otomatis aktif kalau ada GPU (Tensor Cores Ampere) -> training ngebut.
- Bulk training: banyak epoch + validation split + EarlyStopping (restore best weights).
- Persistensi model: train sekali di GPU, simpan ke models/lstm_spinwheel.keras, lalu dipakai ulang.
- evaluate_backtest(): uji walk-forward jujur, akurasi top-1 model vs baseline 'angka paling sering'.

### Baru: train_lstm.py (dijalankan di PC kamu)
Memuat history.json (+ opsional CSV), konfirmasi GPU, training penuh dengan validasi,
backtest jujur (model vs baseline), lalu simpan model. App otomatis memuatnya saat dibuka.

### config/settings.py
- APP_VERSION -> 1.4.0.
- Tambah konfigurasi: LSTM_EMBEDDING_DIM, LSTM_UNITS, LSTM_DENSE_UNITS, LSTM_DROPOUT,
  LSTM_BATCH_SIZE, LSTM_BULK_EPOCHS, LSTM_VALIDATION_SPLIT, LSTM_EARLY_STOP_PATIENCE,
  LSTM_USE_MIXED_PRECISION, LSTM_MODEL_PATH.

### gui/viewmodels/main_viewmodel.py
- _initial_train: kalau model tersimpan ada -> refresh ringan; kalau belum -> bulk train + simpan.

### Catatan kejujuran
GPU bikin training cepat & model bisa besar, TAPI tidak bisa menciptakan pola dari roda yang acak.
evaluate_backtest sengaja dibuat untuk menunjukkan apakah model benar-benar mengalahkan baseline
pada DATA kamu. Kalau tidak (kemungkinan besar pada roda fair), itu bukti, bukan tebakan.

### Catatan teknis
Kode TF tidak bisa dijalankan/diuji di lingkungan build (TensorFlow tidak terpasang di sana);
sudah lolos py_compile (cek sintaks). Jalankan di PC ber-GPU kamu untuk eksekusi penuh.

## v1.5.0 — Physics engine + GPU Monte-Carlo (audit ulang full-GPU)

**Audit ulang (data asli `1.csv`, 117 putaran):** chi-square 12.97 < 15.51 (df=8) -> roda FAIR; repeat lag-1 23.3% vs 22.6% acak -> tidak ada autokorelasi; distribusi cocok dengan luas segmen. Kesimpulan audit tidak berubah: roda acak & adil.

**Baru — `core/physics_wheel.py`:** model dinamika rotasi benda tegar untuk roda fisik.
- Disk seragam: I = 1/2 m R^2. Default radius 0.80 m (~1.6 m diameter, "seukuran orang dewasa"), massa 25 kg -> inersia 8.0 kg*m^2, torsi gesek 4.8 N*m.
- Kinematika deselerasi konstan: omega(t)=omega0-alpha t; Theta=omega0^2/(2 alpha); segmen = floor(theta_f / seg_angle).
- `spin()`, `spin_vec()` (numpy tervektor), `monte_carlo()`, `area_fractions()`, `sensitivity()`, `predictability_report()`.

**Baru — `physics_lab.py`:** laboratorium GPU (jalankan: `python physics_lab.py --ml`).
1. Monte-Carlo jutaan spin di GPU (TensorFlow) -> distribusi hasil vs luas teoretis.
2. Batas keterprediksian: presisi pengukuran omega0 yang dibutuhkan vs yang bisa didapat dari video.
3. Demo ML: jaringan saraf belajar fisika torsi->hasil; akurasi runtuh ke baseline begitu ada noise pengukuran realistis.

**Settings:** APP_VERSION 1.4.0 -> 1.5.0; tambah WHEEL_RADIUS_M, WHEEL_MASS_KG, WHEEL_DECEL_RAD_S2, WHEEL_SPIN_OMEGA_MEAN, WHEEL_SPIN_OMEGA_STD.

**Test:** `_test_physics.py` (5 test, semua lulus): inersia/torsi, determinisme & kinematika, vektor==skalar, Monte-Carlo mereproduksi luas segmen (error 0.0004), batas keterprediksian (butuh 0.048% presisi; video ~29% -> ~600 segmen ketidakpastian).

**Catatan jujur (penting):** Modul fisika ini MEMODELKAN dan MEMBUKTIKAN perilaku roda; ia TIDAK bisa memprediksi spin nyata dari hasil masa lalu. Hasil = fungsi deterministik dari kondisi awal (sudut & kecepatan lepas) yang di-set ulang acak tiap spin. Tidak ada hubungan kausal dari urutan angka ke gaya putar, jadi "belajar torsi dari urutan angka" secara fisika tidak valid. GPU di sini dipakai penuh untuk simulasi & pembuktian, bukan untuk meramal yang tak teramalkan.

*Catatan teknis: TF tidak dapat dijalankan di lingkungan build (hanya py_compile + numpy test). physics_lab.py diuji py_compile; dijalankan di mesin GPU Anda.*

### v1.5.0 — koreksi kejujuran (demo ML)
Output nyata di RTX 3050: model mentok di baseline (~37%) BAHKAN dengan input float presisi, bukan "akurasi tinggi lalu runtuh". Sebabnya hasil = fungsi frekuensi-tinggi dari omega0 (~172 segmen per rad/s) yang mustahil dipelajari jaringan saraf kontinu = tanda tangan chaos. Teks kesimpulan physics_lab.py dibuat dinamis & akurat sesuai temuan ini.
