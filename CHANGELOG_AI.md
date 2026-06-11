# v1.28.1 - Audit V4 fixes: regresi path + migrasi statistik jujur

Menindaklanjuti audit eksternal ke-4 (V4) yang memverifikasi v1.28.0 di level
kode (6/7 temcommit lama beres) dan menemukan 1 regresi + beberapa sisa kecil.

1. **[KRITIS] Path data `scripts/train_lstm.py` diperbaiki.** Setelah pindah ke
   `scripts/`, `os.path.dirname(__file__)` resolve ke `scripts/data/history.json`
   (tidak ada) -> training default diam-diam jalan dengan 0 data. Kini resolve
   ke ROOT project (`<root>/data/history.json`).
2. **Migrasi statistik lama -> `scripts/migrate_stats.py`.** Counter `wins`/
   `losses` lama dihitung dengan definisi LAMA (tebakan 0-token = menang).
   Script sekali-jalan ini menilai ulang SETIAP ronde dengan aturan jujur
   (menang = ada stake nyata pada angka yang keluar), mem-backfill `top1_hit`
   (= tebakan #1 benar), lalu membangun ulang `wins/losses/total/profit`.
   Idempoten + bikin backup `history.json.premigrate.bak`. Pada data seed
   sandbox: WIN RATE (taruhan) 67.8% (lama) -> **0.0%** (jujur, memang tak ada
   taruhan), Akurasi top-1 -> **(n=40)**. **Jalankan sekali di mesinmu:**
   `python scripts/migrate_stats.py`.
3. **Contoh `--csv` di docstring** `train_lstm.py` & `run_backtest.py` diperbarui
   jadi `samples/1.csv` (file CSV sudah pindah ke `samples/`).
4. **Penanda `[EXPERIMENTAL / ANALYSIS-ONLY -- NOT WIRED]`** ditambahkan ke
   docstring `predictors/higher_order_markov.py` biar konsisten dengan modul
   eksperimen lain.

5. **Test dimigrasi ke konvensi pytest.** Ke-26 file `tests/_test_*.py` di-rename
   jadi `tests/test_*.py` (bisa di-discover `pytest`), dan harness `check()` kini
   me-`raise AssertionError` saat gagal -- jadi pytest benar-benar mendeteksi
   kegagalan, bukan cuma nge-print. `run_tests.sh` & README disesuaikan. Catatan:
   pytest belum tentu terinstal -- `pip install pytest` lalu `pytest -q` dari root;
   `bash run_tests.sh` tetap jalan tanpa pytest.

Verifikasi: 26/26 test HIJAU (via run_tests.sh), py_compile seluruh tree OK.

> Catatan binary di git: `.gitignore` baru berlaku untuk commit ke depan. Kalau
> `models/*.keras` / state json terlanjur ke-track, lepas dengan
> `git rm --cached <file>` (file tetap ada di disk).

---

# v1.28.0 - Audit V3 fixes: win-rate jujur, perf, & rapikan struktur folder

Menindaklanjuti audit eksternal ke-3 (V3) yang terverifikasi AKURAT. Tidak ada
fitur prediksi baru -- ini soal KEJUJURAN metrik, efisiensi, dan kebersihan repo.

## Perbaikan logika (jujur > enak dilihat)
1. **Win rate kini = kemenangan TARUHAN beneran.** Sebelumnya tebakan benar
   tapi 0 token (SKIP karena evidence-gate / -EV) ikut dihitung "menang",
   membuat win rate terlihat lebih bagus dari kenyataan. Sekarang `is_win`
   hanya true kalau ada token bertaruh pada angka yang keluar. Kartu di GUI
   diberi label **"WIN RATE (TARUHAN)"**.
2. **Metrik baru "Akurasi tebakan teratas" (top-1).** Ditampilkan TERPISAH di
   panel statistik -- ini akurasi tebakan #1 apa adanya (lepas dari staking),
   jadi dua angka itu tidak lagi saling tertukar. Disimpan per ronde
   (`top1_hit`) dan diekspos lewat `get_stats()` (`top1_accuracy`,
   `top1_graded`).
3. **Tulis riwayat inkremental.** `Tracker.record_result()` kini meng-INSERT
   satu baris via `store.append_record()` + upsert metadata, bukan menulis
   ulang SELURUH riwayat (O(n)) tiap spin. Mirror `history.json` tetap dijaga.
4. **`export_csv()` kolom `bets` kini JSON valid** (`json.dumps`), bukan repr
   dict Python -> bisa di-parse ulang.
5. **Buang dead code:** loop `fitted` mati di `calibration._pava()`, import
   `kelly_allocation` & cabang engine `Bayesian` yang tak terjangkau di
   ViewModel.

## Modul eksperimen ditandai jelas (bukan dihapus)
Kamu memang mau terus bereksperimen, jadi modul tak-tersambung DIPERTAHANKAN tapi
ditandai `[EXPERIMENTAL / NOT WIRED]` + didokumentasikan di `experimental/README.md`:
`AdaptiveRiskManager` (core/risk_manager.py), `GamblersFallacySimulator`
(core/wheel_math.py), dan `HigherOrderMarkovEngine` (alat analisis saja --
backtest jujur tidak menunjukkan ia lebih baik dari baseline, jadi sengaja TIDAK
diekspos di dropdown engine).

## Struktur folder dirapikan
Root tadinya berisi 27 file test + script + CSV + artefak campur aduk. Sekarang:
- `tests/` -- 26 file `_test_*.py` (tiap file kini menyisipkan root project ke
  `sys.path` otomatis, jadi tetap bisa dijalankan dari mana saja selama dari root).
- `scripts/` -- `train_lstm.py`, `run_backtest.py`, `physics_lab.py`, `wheel_cam.py`.
- `samples/` -- `1.csv`, `2.csv` (data contoh; jalankan mis. `--csv samples/1.csv`).
- `reports/` -- artefak hasil (`backtest_results.csv`), dengan `reports/arsip/`
  menyimpan `laporan_audit.*` lama (v1.6.1) yang sudah usang.
- `run_tests.sh` -- runner satu perintah; `.gitignore` baru (abaikan `*.keras`,
  `data/*.db`, state json, `reports/`, `__pycache__/`, venv).

**Verifikasi:** 26/26 file test HIJAU dari lokasi `tests/` baru; py_compile
seluruh tree OK. Wheel tetap near-random i.i.d. -- reality-check tetap MENYALA;
perbaikan win-rate ini justru menegaskan kejujuran itu.

---

# v1.27.0 - Experimental camera wheel tracking (OpenCV) (PROMPT 20)

## Audit fix (re-check semua 20 prompt)
Saat audit ulang menyeluruh, ketemu 1 regression dari PROMPT 18: `Tracker`
yang dibuat dengan `history_file` custom tapi tanpa `db_file` diam-diam memakai
`data/history.db` GLOBAL (dan auto-migrate `data/history.json` asli) -> bikin
test saling cemari. Fix: `db_file` kini OTOMATIS jadi sibling dari `history_file`
kalau tidak diberi eksplisit. Default produksi tetap identik
(`data/history.json` -> `data/history.db`), tapi history_file custom kini dapat
storage TERISOLASI. Hasil audit: 26/26 file test HIJAU, py_compile seluruh tree OK.

## DISCLAIMER PALING PENTING
Ini bagian PALING eksperimental & PALING lemah dari semua project. Ia OBSERVASI
roda fisik (estimasi sudut / kapan berhenti / mendarat di segmen mana). Ia
**TIDAK dan TIDAK BISA** memprediksi hasil spin berikutnya -- spin asli itu
proses fisik chaotic. Anggap ini alat REKAM/ANOTASI hasil, bukan dukun.

## Arsitektur (inti murni + capture opsional)
- **`vision/wheel_tracker.py` -- inti murni numpy, TANPA OpenCV.** Semua
  geometri + state ada di sini, jadi 100% bisa di-unit-test di Python+numpy
  polosan:
  - `normalize_angle`, `angular_delta` (tahan wrap 0/360),
  - `angle_from_point(center, point)` (0=kanan, 90=atas, 180=kiri, 270=bawah),
  - `angle_to_segment_index` + `angle_to_number` (petakan sudut -> angka roda
    pakai `SPINWHEEL_SEQUENCE` 54-segmen, plus skor keyakinan = seberapa di
    tengah segmen),
  - `mask_centroid` (centroid pixel dari mask),
  - `AngleTracker`: hitung kecepatan sudut, deteksi BERHENTI (hanya setelah ada
    gerakan nyata + N frame terakhir di bawah ambang -> diam awal tidak salah
    dihitung sebagai selesai), latch sudut istirahat, map ke angka.
- **`vision/camera.py` -- satu-satunya yang menyentuh OpenCV (OPSIONAL).**
  Import-safe walau cv2 tidak terinstall: `opencv_available()` / `opencv_status()`,
  dan `CameraWheelTracker.run()` melempar RuntimeError yang jelas (bukan crash).
  - `build_mask` (HSV inRange, dukung multi-range mis. merah yang wrap hue),
  - `frame_marker_angle(frame, center)` -> sudut marker per-frame,
  - `detect_wheel_center` (Hough circle, fallback ke tengah frame),
  - `CameraWheelTracker(source, ...)`: drive AngleTracker dari webcam/video.
- **`wheel_cam.py` -- CLI.** `--source` (index kamera / file), `--duration`,
  `--max-frames`, `--no-show`. Kalau cv2 tidak ada -> cetak cara install +
  keluar bersih (app utama tetap jalan).

## Cara pakai
```
pip install -r requirements-vision.txt     # opencv-python + numpy
python wheel_cam.py --source 0             # webcam, ada jendela preview
python wheel_cam.py --source clip.mp4 --no-show
```
Tempel marker warna terang (default: hijau) di pinggir roda biar kebaca.

## Tes
`_test_wheel_tracker.py` -> **37/37 LOLOS**: matematika sudut (normalize/delta/
wrap, from_point 4 arah, segment index + offset, number mapping + confidence
tengah vs tepi), mask_centroid (+ kosong->None), AngleTracker (spin melambat
sampai berhenti + latch + map angka; diam tidak dianggap berhenti; kecepatan
benar saat wrap 360), PLUS lapisan OpenCV pakai FRAME SINTETIS (marker hijau di
kanan->~0deg, di atas->~90deg, tanpa marker->None, build_mask). COMPILE_OK:
vision/* + wheel_cam.py. (Loop capture kamera tidak diuji -- tidak ada device --
tapi ia hanya mendelegasikan ke helper yang sudah teruji.)

## Penutup
Ini menutup SEMUA 20 prompt audit (v1.8.0 -> v1.27.0). Dari sisi taruhan: tidak
ada satu pun fitur baru yang mengubah kenyataan bahwa wheel ini house-edge dan
nyaris acak. Yang berubah cuma KEJUJURAN, KEANDALAN DATA, dan PERKAKAS -- bukan
edge.

# v1.26.0 - REST API (FastAPI + stdlib fallback, --serve) (PROMPT 19)

## Kenapa
Biar engine prediksi + riwayat bisa diakses program lain (script, dashboard,
bot, automasi) tanpa harus buka GUI. Sekalian misahin "otak" dari "tampilan".

**Catatan jujur:** API cuma EXPOSE logika yang sudah ada -- nggak nambah akurasi
sama sekali. Prediksi tetap estimasi probabilistik di proses (nyaris) acak,
bukan jaminan hasil.

## Arsitektur (3 lapis, 1 sumber kebenaran)
- **`api/service.py` -- routing core murni stdlib.** Semua route, validasi,
  dan serialisasi di sini. Transport HTTP cuma tipis di atasnya, jadi perilaku
  identik di transport mana pun + 100% bisa di-unit-test tanpa framework web.
  Logika berat (ViewModel/TensorFlow) di-inject lewat objek `backend`.
- **`api/server.py` -- dua transport berbagi core yang sama:**
  - **FastAPI + uvicorn** kalau terinstall -> Swagger docs interaktif di `/docs`,
    skema request pydantic. Pengalaman "enak".
  - **Fallback `http.server` stdlib** kalau belum -> ZERO dependency tambahan,
    jadi API SELALU jalan walau Python polosan. Otomatis pindah, nggak bisa mati.
- **`RealBackend`:** `/stats`, `/history`, `/record` cuma butuh Tracker ringan
  (jalan walau tanpa TF). `/predict` baru bikin ViewModel penuh (lazy) saat
  request pertama datang.

## Endpoint
- `GET  /health`  -> liveness + versi app
- `GET  /stats`   -> modal, win rate, streak, frekuensi (advanced stats)
- `GET  /history?limit=N` -> N record terakhir
- `POST /predict` -> `{engine?, history_length?}` -> daftar prediksi
- `POST /record`  -> `{actual_number, profit_change, predicted_number?, bets?, engine_used?, mode?}` -> stats terbaru
- `GET  /` -> daftar endpoint; FastAPI: `/docs` (Swagger) + `/redoc`

Validasi: limit/angka non-integer, field wajib hilang, bets bukan list, body
bukan objek, JSON rusak -> semua balas 4xx JSON yang rapi (bukan 500/crash).

## Cara pakai
```
python app/main.py --serve                  # http://127.0.0.1:8000 (auto FastAPI kalau ada)
python app/main.py --serve --port 9000      # ganti port
python app/main.py --serve --no-fastapi      # paksa fallback stdlib
pip install -r requirements-api.txt          # unlock FastAPI + Swagger /docs
```
Mode `--serve` melewati GUI + diagnostik TF; TF baru dimuat saat /predict pertama.

## Tes
`_test_api.py` -> **40/40 LOLOS**: routing core (health/stats/history/predict/
record, default & query-string limit, trailing-slash, 404 method/path, 400 untuk
tipe/field salah & body non-objek) PLUS end-to-end transport stdlib BENERAN
(server di thread, di-hit pakai `requests`: health/stats/history/predict/record +
JSON rusak->400 + unknown->404). COMPILE_OK: api/service, api/server, app/main.

## Catatan
- FastAPI/uvicorn TIDAK bisa di-runtime-test di sandbox build ini (tanpa
  network -> nggak bisa pip install). Lapisan FastAPI sengaja dibikin shim tipis
  di atas core yang sudah 100% teruji, dan sudah lolos py_compile. Fallback
  stdlib teruji penuh end-to-end, jadi --serve dijamin jalan apa adanya.

# v1.25.0 - SQLite Storage + Auto-Migration + JSON Import/Export (PROMPT 18)

## Kenapa
`history.json` ditulis ulang penuh tiap ronde. Aman di skala kecil, tapi rapuh:
satu crash/disk-full saat tulis bisa mengorbankan SELURUH riwayat. SQLite
memberi penyimpanan transaksional & crash-safe, tetap ringan (stdlib, 1 file).

**Catatan jujur:** ini murni soal KEANDALAN DATA, bukan akurasi prediksi. Tidak
mengubah satu pun angka taruhan atau edge. Cuma bikin riwayatmu lebih susah hilang.

## Yang berubah
- **Modul baru `data/sqlite_store.py`** (murni stdlib `sqlite3`+`json`):
  - Tabel `meta` (current_capital/total/wins/losses/profit + key tak-standar apa pun)
    dan `history` (kolom bertipe untuk query + blob JSON penuh tiap record sehingga
    SEMUA field round-trip persis, termasuk `bets`, `engine_used`, `mode`, dan key masa depan).
  - `save_all` (replace transaksional), `append_record`, `load`, `is_empty`, `count`.
  - `migrate_from_json`, `import_json(mode="replace"|"append")`, `export_json`.
- **`data/tracker.py` dirombak (tetap kompatibel penuh):**
  - SQLite jadi sumber kebenaran durable. Bentuk `self.data` di memori TIDAK berubah,
    jadi seluruh kode lain (stats, sesi, analytics, bankroll, tilt, consensus) jalan apa adanya.
  - **Auto-migrasi sekali jalan:** kalau DB kosong tapi `history.json` ada, isinya diimpor
    ke SQLite + backup disimpan di `history.json.migrated` (file lama TIDAK dihapus).
  - `save_data()` nulis ke SQLite (utama) DAN mirror ke `history.json` (backup + auto-export,
    format lama dipertahankan -- additive only).
  - Method baru `export_json(path)` & `import_json(path, mode)`.
- **GUI:** dua tombol baru di zona aksi -- **EXPORT JSON** (history+meta) dan **IMPORT JSON**
  (dialog GABUNG vs GANTI). Setelah import, dashboard + modal langsung refresh.

## Keamanan data
- File lama tidak pernah dihapus; ada backup `.migrated`.
- JSON corrupt saat migrasi -> di-backup ke `.corrupt.json`, app tetap jalan dari nol.
- `RESET SEMUA` kini juga mengosongkan SQLite (lewat replace transaksional).

## Tes
`_test_sqlite_store.py` -> **55/55 LOLOS** (shape; fresh empty; save/load round-trip
termasuk bets/None/engine/mode/key-masa-depan; replace; append_record + meta upsert;
migrasi (sukses/file-hilang/non-dict); import replace & append (meta dihitung ulang);
export->import; persistensi buka-ulang; integrasi Tracker end-to-end: auto-migrate +
backup + reopen-from-sqlite + reset). COMPILE_OK: sqlite_store, tracker, main_window.

# v1.24.0 - Multi-Engine Consensus Filter (PROMPT 17)

**Ringkasan jujur:** filter RISIKO, bukan generator edge. Di roda adil & tanpa memori,
kesepakatan banyak model TIDAK bikin angka "sudah waktunya". Yang dilakukan filter ini:
mengurangi overconfidence/noise satu model dengan hanya membiarkan taruhan berdiri kalau
>= K model INDEPENDEN sepakat. Ia tidak bisa menciptakan keuntungan yang tak diberikan
pembayaran roda.

## Baru
- **`core/consensus.py`** (modul murni, stdlib saja, headless-testable):
  - `top_numbers_from_dist(dist, top_n, min_prob)` / `top_numbers_from_preds(...)` - ambil
    top-N "vote" dari distribusi {angka: prob} atau list prediksi.
  - `tally_votes(engine_top)` - hitung berapa engine vote tiap angka (+ daftar voter).
  - `consensus_numbers(votes, min_agree)` - himpunan angka yang didukung >= K engine.
  - `build_votes(engine_distributions, ...)` - rakit votes + jumlah engine aktif (skip yang kosong).
  - `apply_consensus_filter(predictions, engine_distributions, min_agree, top_n, ...)` - nol-kan
    `token_bet` pada angka di bawah ambang konsensus; anotasi tiap prediksi dengan
    `consensus_votes`/`consensus_voters` (+ `consensus_blocked`). **FAIL OPEN** (tidak
    memblok apa pun) saat engine < min_agree atau filter dimatikan.
- **`ContinuousLearningEngine.model_distributions(history)`** - akses publik ke distribusi
  tiap model penyusun (physics / bayes / markov / lstm), buang yang tak tersedia. Dipakai
  sebagai voter independen.
- **Konfigurasi (`config/settings.py`):** `CONSENSUS_FILTER_ENABLED=True`,
  `CONSENSUS_MIN_AGREE=2`, `CONSENSUS_TOP_N=3`, `CONSENSUS_MIN_PROB=0.0`.
- **Wiring ViewModel:** di `get_predictions`, SETELAH reality-check, taruhan yang lolos
  dicross-check ke vote independen physics/bayes/markov/lstm; hanya angka yang didukung
  >= K engine yang boleh memasang token. Info disimpan di `self.consensus_info`.
  (Jalur AI-Optimal & Harvest tidak melewati filter ini -- AI-Optimal sudah paling
  konservatif, Harvest sengaja bypass semua gate.)

## Uji
- `_test_consensus.py`: **37/37 LOLOS** (top-N dari dist & preds; min_prob/min_confidence;
  tally + dedup vote dalam satu engine; consensus_numbers untuk K=2/3; build_votes skip
  distribusi kosong/None; blokir angka tanpa konsensus + anotasi vote; lolos saat cukup
  vote; FAIL OPEN saat engine kurang; disabled; tidak menyentuh taruhan yang sudah 0;
  min_agree=3).
- `py_compile` OK: consensus, continuous_engine, settings, viewmodel.

# v1.23.0 - Anti-Tilt Guard (TiltDetector) (PROMPT 16)

**Ringkasan jujur:** fitur ini melindungi BANKROLL & DISIPLIN, bukan menambah edge.
Di roda yang adil & tanpa memori, kalah beruntun TIDAK bikin menang "sudah waktunya",
dan menaikkan taruhan cuma menaikkan variance + kerugian harapan. Guard ini berhenti
memprediksi saat mendeteksi pola tilt dan memaksa jeda.

## Baru
- **`core/tilt.py`** (modul murni, stdlib saja, headless-testable):
  - `is_rising(stakes, strict)` - deteksi taruhan menaik (strict = tiap langkah naik).
  - `detect_tilt(events, n_losses=3, window_minutes=5, require_rising=True)` - cek pola
    tilt pada N ronde terakhir: semua kalah, ada taruhan (>0), dalam jendela waktu, dan
    (opsional) taruhan menaik. Mengembalikan `{triggered, reason, span_minutes, ...}`.
  - `TiltDetector` - guard stateful dengan cooldown wajib: `record()`, `status()`,
    `is_in_cooldown()`, `clear_cooldown()`, `reset()`; buffer bergulir; auto-expire.
- **Konfigurasi (`config/settings.py`):** `TILT_DETECT_ENABLED`, `TILT_N_LOSSES=3`,
  `TILT_WINDOW_MINUTES=5.0`, `TILT_COOLDOWN_SECONDS=60`, `TILT_REQUIRE_RISING=True`,
  `TILT_STRICT_RISING=True`.
- **Wiring ViewModel:** tiap ronde dikonfirmasi -> `tilt_detector.record(loss?, total_staked)`
  (total taruhan = jumlah token_bet di snapshot; kalah = profit bersih <= 0).
  Helper baru `tilt_status()` dan `acknowledge_tilt()`.
- **UI:** banner merah anti-tilt di atas tombol "HITUNG PREDIKSI" dengan hitung mundur
  cooldown (~1 dtk). Saat cooldown, tombol prediksi dikunci jadi "TENANG DULU (Ns)".
  Klik banner = "saya sudah tenang" -> cooldown dibersihkan.

## Uji
- `_test_tilt.py`: **49/49 LOLOS** (is_rising; trip/tidak-trip karena jendela/menang/
  taruhan-turun/ronde-tanpa-taruhan/data-kurang; pakai-ekor-saja; timestamp ISO;
  parameter n_losses; cooldown aktif/auto-expire/clear/reset/disabled; buffer cap/min;
  pulih lalu trip lagi).
- `py_compile` OK: tilt, settings, viewmodel, main_window.

# v1.22.0 - Laporan Bankroll Harian/Per-Sesi + Calendar Heatmap (PROMPT 15)

Melihat KAPAN token benar-benar didapat atau hilang, bukan cuma satu angka total.
Pembukuan, BUKAN edge: di roda adil P/L harian didominasi variance.

## Baru
- `core/bankroll.py` (modul murni, tanpa GUI/dep eksternal):
  - `daily_report(data)` - agregasi per tanggal kalender (ronde, win-rate,
    profit, best/worst ronde, max drawdown intra-hari) + ekuitas kumulatif
    berjalan (`cum_profit`).
  - `session_report(data, gap_minutes=30)` - agregasi per sesi main (jeda idle
    >= gap = sesi baru).
  - `overall_summary(data)` - hari hijau/merah, rata-rata harian, hari
    terbaik/terburuk, max drawdown harian, rentet hari rugi terpanjang.
  - `format_report_text(data)` - laporan teks rapi untuk panel GUI.
- `core/analytics_charts.py`:
  - `build_calendar_heatmap_figure` - kalender ala GitHub: kolom = minggu,
    baris = hari (Sen-Min), warna sel = P/L hari itu (merah rugi / hijau
    profit, skala simetris di sekitar 0).
  - `build_daily_pnl_figure` - batang P/L per hari (hijau/merah) + garis
    ekuitas kumulatif (sumbu kanan).
  - `build_bankroll_figures` + `export_bankroll_charts` (PNG x2 / PDF x1,
    headless-safe lewat Agg).
- GUI: tombol **BANKROLL** membuka panel berisi kalender heatmap + grafik
  P/L harian + laporan teks (harian & per-sesi) + tombol export grafik.
- Laporan audit: section baru **## 13. Bankroll harian & per-sesi** (tabel
  per hari + ringkasan hari hijau/merah, hari terbaik/terburuk, rentet rugi).

## Catatan jujur
- Hari hijau BUKAN bukti sistem menang. Fitur ini untuk audit drawdown dan
  apakah profit cuma menumpuk di segelintir sesi/hari beruntung.

## Tes
- `_test_bankroll.py` - 54/54 LULUS (agregasi harian/sesi, ringkasan,
  drawdown, rentet rugi, teks, grafik kalender/batang, export PNG/PDF, input
  list-mentah & timestamp rusak diabaikan). Regresi analytics 20/20 + audit
  diagnostics tetap hijau.

## v1.21.0 - Manual % = Dirichlet prior (strength slider, buang blend 0.35 hardcoded)

Dulu manual % dicampur ke engine dengan bobot TETAP 0.35 (0.65*engine + 0.35*manual) - tebakan dihitung sama beratnya entah kamu baru 3 spin atau 300 spin. Itu salah secara statistik. Sekarang manual % diperlakukan sebagai Dirichlet prior yang JUJUR.

- KONSEP: manual % = prior senilai `strength` pseudo-observasi. Bobot campur vs engine = strength / (strength + jumlah_spin). Jadi:
  - cold start (sedikit data) -> tebakanmu dominan (regularisasi berguna saat data minim).
  - makin banyak spin nyata -> bobot prior MENGECIL otomatis; data akhirnya menang.
  - strength == jumlah spin -> bobot 0.5 (imbang).
  - di default 27 dengan ~50 spin -> bobot ~0.35: kompatibel dengan perilaku lama, TAPI sekarang bergerak mengikuti data.
- SLIDER baru "Kekuatan Prior Manual" (5-100, default 27 pseudo-spin) di panel kiri, di bawah tombol KUNCI. Kecil = cepat percaya data nyata; besar = lama pegang tebakan. Saat terkunci, geser slider langsung me-recompute persen live.
- UNIFIKASI: satu parameter `manual_prior_strength` kini menyetir BOTH (a) bobot blend ke engine DAN (b) seberapa lama prior terkunci bertahan terhadap frekuensi live (dulu hardcoded 54). Default turun 54 -> 27 (konsisten dengan slider).
- Blend 0.35/0.65 hardcoded DIHAPUS dari ketiga jalur prediksi (AI-Optimal, engine umum, Variance Harvest) - semua lewat `core/priors.apply_manual_prior`.
- Logika prior dipisah ke modul murni `core/priors.py` (dirichlet_prior_weight, blend_confidence, apply_manual_prior, clamp_strength) -> bisa diuji headless.
- TES `_test_manual_prior.py` (23 cek, SEMUA LOLOS): limit (strength 0 -> bobot 0; data 0 -> bobot 1; strength==data -> 0.5), monotonik (naik dgn strength, turun dgn data), kompat-mundur (27@50 ~= 0.35), blend konveks dalam batas, has_manual_input, mutasi confidence benar, no-op saat manual kosong, clamp slider.
- HONEST NOTE: prior TIDAK menciptakan sinyal di roda adil tanpa memori. Manfaatnya murni regularisasi cold-start + cara mengkodekan keyakinan yang otomatis memudar saat data nyata menumpuk.

## v1.20.0 - Analytics Deep: panel audit visual + export grafik

Panel diagnostik visual baru. INGAT: grafik bagus TIDAK menciptakan edge - ini alat untuk MELIHAT apakah performa yang kelihatan itu sinyal nyata atau cuma variance.

- TOMBOL UI baru di zona statistik: "ANALYTICS" (buka panel 2x2) + "EXPORT GRAFIK AUDIT".
- PANEL "Analytics Deep" (jendela 2x2) berisi 4 grafik matplotlib:
  1. Reliability / kalibrasi per engine - confidence yang diklaim vs hit-rate sebenarnya, dengan garis diagonal "kalibrasi sempurna". Makin dekat diagonal = confidence makin jujur.
  2. Heatmap deviasi per-angka x engine - observed minus predicted probability (hijau/merah diverging). Deviasi besar = miskalibrasi atau bias roda.
  3. Kurva ekuitas + drawdown - P/L kumulatif, garis puncak, dan area drawdown ter-shading merah.
  4. Win-rate bergulir (window 50 ronde) dengan baseline acak opsional.
- EXPORT GRAFIK: simpan 4 PNG terpisah ATAU 1 PDF gabungan (pilih ekstensi di dialog). Headless-safe (pakai Agg canvas, tidak ganggu backend GUI).
- Logika chart dipisah ke modul GUI-agnostic `core/analytics_charts.py` (tanpa tkinter) -> reducer murni (reliability_bins, per_number_deviation, equity_series, rolling_winrate) + figure builder + export, semuanya bisa diuji headless.
- TES `_test_analytics_charts.py` (20 cek, SEMUA LOLOS): reducer benar (deviation = observed - predicted; drawdown = peak - equity; win-rate 0-100; peak monoton), 4 figure terbangun, data kosong degrade mulus, export PNG (4 file) + PDF (1 file) menghasilkan file non-kosong.

## v1.19.0 - Variance Harvest mode (OPT-IN, default OFF)

> PERINGATAN JUJUR: mode ini SENGAJA -EV jangka panjang. Tujuannya bukan profit konsisten, tapi memberi eksposur variance terkontrol (tiket lotre kecil) di angka multiplier tinggi. Default MATI; reality-check & EV-gate engine biasa tetap utuh saat mode ini OFF.

- TOGGLE UI baru di header: switch merah "VARIANCE HARVEST" (default OFF). Saat ON, muncul BANNER merah full-width: "VARIANCE HARVEST MODE - lottery tickets only on high-multiplier numbers. Long-run -EV but high-variance potential."
- LOGIC harvest (`core/harvest.py`, murni Python, teruji): bypass EV-gate; hanya pasang angka dengan multiplier >= 5 (skip #1/#2); confidence >= HARVEST_MIN_CONFIDENCE (5%); stake TETAP kecil = round(modal * 2%), min 1 token; maksimal 2 angka per ronde; ranking berdasarkan nilai-lotre (confidence x multiplier).
- AUTO-SKIP ronde: 20% ronde dilewati acak untuk hemat modal (`should_skip_round`).
- AUDIT terpisah: tiap ronde harvest ditandai `mode="harvest"` di history (additive, backward-compatible). Laporan audit dapat section baru "## 12. Variance Harvest" - jumlah ronde, profit, avg/ronde, win-rate, big-win (mult>=10), max drawdown, dan Sharpe-equivalent (mean/stddev per ronde).
- SIMULATOR + tes (`_test_variance_harvest.py`, 23 cek, SEMUA LOLOS): default OFF terverifikasi; skip #1/#2; cap 2 pick; stake tetap 2%; gate confidence 5%; floor stake >=1; payout multiplier benar; skip-rate ~20% statistik; simulasi 500 spin -> modal TIDAK habis (bleed kecil, sisa >40%); big-multiplier hits tertangkap.
- Settings baru: HARVEST_MODE_DEFAULT=False, HARVEST_MIN_CONFIDENCE=0.05, HARVEST_TARGET_MULTIPLIERS=[5,8,10,15,20,30,40], HARVEST_TOKEN_PCT=0.02, HARVEST_MAX_PICKS=2, HARVEST_SKIP_RATE=0.20.
- KONTEKS: data nyata kamu (+53 profit dari 217 spin walau semua -EV) itu murni variance/lottery effect. Mode ini memformalkan eksposur itu dengan disiplin (stake kecil, capped, skip), BUKAN mengubahnya jadi edge. Sistem konservatif yang selalu SKIP itu benar secara matematis; harvest hanyalah pilihan sadar untuk "beli tiket lotre kecil".

## v1.18.0 - LSTM upgrade: konteks lebih panjang + attention + feature engineering

- KONTEKS lebih panjang: `LSTM_SEQUENCE_LENGTH` default 5 -> 10 (configurable). Untuk dataset 200+ spin, window 10 lebih pas untuk menangkap dependency jarak jauh.
- SELF-ATTENTION: arsitektur baru (functional API) Embedding -> LSTM(128, return_sequences) -> MultiHeadAttention(4 head, dim 32) + residual + LayerNorm -> LSTM(64) -> Dense(64, relu) -> softmax. Toggle `LSTM_USE_ATTENTION` (default True); kalau False, layer attention dilewati (fallback arsitektur biasa) -> kompatibel mundur.
- FEATURE ENGINEERING (BARU `predictors/lstm_features.py`, murni numpy, CAUSAL): tiap timestep dapat fitur tambahan di samping embedding angka - one-hot angka, frekuensi window-5 terakhir, time-since-last-same (cap /50), plus session_position relatif window. Toggle `LSTM_USE_FEATURES`.
- AUGMENTASI saat training (bulk): `augment_drop_random_prefix` - sebagian window dipotong prefix acak (diganti PAD) supaya model tahan konteks pendek. Toggle `LSTM_AUGMENT`. Embedding pakai id geser +1 (0 = PAD).
- KOMPAT MODEL LAMA: model .keras dari <=v1.17.0 (seq_len 5, tanpa fitur) otomatis ditolak oleh `try_load()` karena input-shape mismatch -> model baru dilatih ulang. (Saran: hapus `models/lstm_spinwheel.keras` lama saat clean start.)
- PEMBANDING JUJUR: `compare_attention_vs_vanilla()` + flag `python train_lstm.py --compare-attention` -> backtest walk-forward attention vs vanilla pada DATA KAMU, laporkan lift. (Wajib dijalankan di mesin GPU; TF tidak ada di lingkungan build ini.)
- Setting baru: LSTM_USE_ATTENTION=True, LSTM_ATTENTION_HEADS=4, LSTM_ATTENTION_DIM=32, LSTM_USE_FEATURES=True, LSTM_AUGMENT=True, LSTM_AUGMENT_PROB=0.5, LSTM_AUGMENT_MAX_FRAC=0.5.
- Tes `_test_lstm_attention.py`: 29 cek lapisan numpy (bentuk fitur, one-hot, freq-5 ternormalisasi, time-since-same, geser +1, label di ruang kelas, session_position 0->1, augmentasi nambah baris + PAD prefix, setting ada). SEMUA LOLOS. Lapisan TF (build/train 1 epoch/predict shape, attention & vanilla) di-SKIP otomatis bila TF tak terpasang - jalan beneran di mesin GPU kamu.
- CATATAN JUJUR: angka lift attention-vs-vanilla pada data nyata BELUM bisa dilaporkan dari sini karena TF tidak terpasang di sandbox build. Pada 235 spin yang ada, ekspektasi realistis tetap di sekitar baseline (~37-40%) - attention memperbaiki kapasitas model, bukan keacakan roda. Ukur sendiri dengan `--compare-attention`.

## v1.17.0 - Heuristic dipensiunkan ke legacy (gambler's fallacy)

- DIHAPUS dari dropdown utama: "Heuristic" tidak lagi jadi pilihan engine. Dropdown sekarang hanya ["AI-Optimal", "Ensemble", "Markov", "TF-LSTM"] - alasan: heuristik "overdue/proximity" adalah gambler's fallacy dan menyesatkan untuk roda adil.
- PINDAH lokasi: `predictors/heuristic_engine.py` -> `predictors/legacy/heuristic_engine.py` (paket legacy baru + `__init__.py`). Path lama tetap bekerja sebagai shim back-compat, tapi memunculkan DeprecationWarning.
- Import internal (viewmodel, `core/diagnostics`, `run_backtest`) sekarang menunjuk langsung ke `predictors.legacy.heuristic_engine` - tetap dipakai di backtest/diagnostics sebagai pembanding, bukan prediktor primer.
- BARU Lab Mode tersembunyi (Ctrl+Shift+L): jendela edukatif yang menampilkan keyakinan heuristik 'overdue' berdampingan dengan frekuensi roda sebenarnya, plus penjelasan kenapa gambler's fallacy salah. Murni edukasi.
- Ensemble/continuous engine dikonfirmasi TIDAK pernah mereferensikan heuristik sebagai model -> tidak ada perubahan diperlukan di sana.
- Tes `_test_legacy_heuristic.py`: engine legacy berfungsi & ternormalisasi, modul fisik ada di predictors/legacy/, shim re-export kelas yang sama + memicu DeprecationWarning, dropdown bebas "Heuristic", handler Lab Mode ada, continuous engine bebas heuristik. SEMUA LOLOS (9/9).

## v1.16.0 - Session detection & temporal-drift (KS)

- BARU `core/sessions.py`: deteksi sesi & statistik drift TANPA scipy. chi-square goodness-of-fit, KS 2-sample (statistik D + p-value asimtotik distribusi Kolmogorov), `detect_session_drift` (pool separuh-awal vs separuh-akhir sesi), `recency_weights` (exponential decay, half-life).
- `data/tracker.py`: `get_sessions(gap_minutes=30)` (pisah history per jeda idle), `per_session_stats()` (n_spins, angka dominan, win rate, profit, chi^2 per sesi), `session_drift()` (putusan KS).
- `core/diagnostics.py`: section baru "## 11. Analisis per-sesi" - tabel per sesi + putusan "TERDETEKSI DRIFT ANTAR-SESI" bila KS p < alpha. Di data nyata sekarang: 1 sesi, abstain (butuh >=2 sesi) - jujur, tidak mengarang drift.
- Konstanta settings: SESSION_GAP_MINUTES=30, SESSION_DRIFT_ALPHA=0.05, RECENCY_HALF_LIFE=50.
- Tes `_test_sessions.py`: split per jeda, field per-sesi, chi^2 mendeteksi skew, KS membedakan distribusi identik vs disjoint, putusan drift + abstain 1-sesi, recency_weights monoton, render section 11. SEMUA LOLOS.
- CATATAN JUJUR: chart profit per-sesi (small multiples) di UI ditunda ke panel matplotlib PROMPT 13 (v1.20.0) supaya tidak menambah dependency GUI setengah jadi sekarang. Logika & data per-sesi sudah lengkap dan teruji.

## v1.15.0 - Confidence interval & support visualization

- BARU `core/bootstrap_ci.py`: CI 95% non-parametrik via bootstrap (resample history 200x dgn replacement, recompute confidence, percentile 2.5/97.5) untuk engine tanpa CI native (Heuristic/LSTM). Bayesian pakai CI analitik dari posterior Dirichlet (tidak ditimpa).
- `attach_confidence_intervals()` mengisi ci_low/ci_high untuk top-3 pick yg belum punya, dan support fallback = jumlah observasi.
- `support_label()` badge kekuatan-evidence: <25 = RAW (cold start)/merah, 25-100 = WARMING UP/emas, >100 = STABLE/hijau.
- Kartu prediksi (gui/views/main_window.py): baris baru "CI 95%: [low%, high%]" + badge support berkode-warna + mini-bar inline (track penuh = sumbu 0-100%, segmen ter-shade = pita CI, marker = titik estimasi).
- viewmodel: lampirkan CI ke preds lalu propagasi ke alokasi taruhan; bungkus try/except supaya gagal-bootstrap tidak mematikan prediksi.
- Tes `_test_ci_visualization.py`: batas CI valid (0<=low<=high<=1), reproducible per-seed, attach mengisi top-3 + support fallback, CI native Bayesian/Markov tidak ditimpa, tier label benar. SEMUA LOLOS.

## v1.14.0 - Correlation-aware net-Kelly portfolio

- BARU `core/betting.net_kelly_portfolio()`: alokasi taruhan sadar-korelasi. Karena tiap spin cuma SATU angka menang, stake antar-angka saling eksklusif (negatif berkorelasi) - token di A hangus tiap B menang. `kelly_allocation` lama menyize tiap angka independen lalu rescale ke budget, mengabaikan korelasi ini.
- net-Kelly memaksimalkan expected LOG-GROWTH atas SELURUH distribusi outcome via alokasi integer greedy (marginal-gain). Greedy berhenti otomatis di optimum Kelly -> mustahil over-bet meski budget besar.
- Fractional Kelly via penskalaan bankroll efektif (W = capital * kelly_fraction); di limit taruhan-tunggal cocok persis dengan half-Kelly lama.
- Gate EV + evidence dipertahankan (ev>margin DAN support>=min_support); kalau tak ada yang lolos -> SKIP.
- viewmodel kini pakai net_kelly_portfolio sebagai allocator (reality-check Bayesian tetap jalan di hilir).
- Tes `_test_net_kelly.py`: bukti log-growth net-Kelly (0.1308) >= sizing independen (0.1163) pada model terkorelasi, knob fractional, budget dihormati, SKIP no-edge, evidence gate, single-edge.

## v1.13.0 - Ensemble Bayesian Model Averaging (BMA)

- `core/continuous_engine.weights()` ditulis ulang: dari softmax-EMA-akurasi menjadi BMA sejati. Bobot tiap model = posterior Bayesian ~ exp(log-evidence prediktif terdiskon). Tiap spin, log-evidence model += log p_model(hasil_nyata), dengan forgetting factor (ENSEMBLE_BMA_DISCOUNT=0.98) supaya adaptif.
- BARU `stacking_weights()`: stacking optimal-log-loss via EM mixture-weight (opsional, ENSEMBLE_USE_STACKING=False default -> blend 50/50 dengan BMA bila aktif).
- Maturity-gate dipertahankan: markov/lstm hanya dapat bobot penuh setelah cukup spin; physics+bayes tetap jangkar cold-start.
- prediction_log kini simpan p_model(hasil) untuk stacking; state simpan log_evidence (backward-compatible, default 0).
- learning_status() tampilkan method + log_evidence.
- Tes `_test_continuous_bma.py`: cold-start 0.5/0.5 + gating, BMA hadiahi model adaptif (bayes>physics di realita berat-2), stacking valid, blend mode, predict/status utuh.

## v1.12.0 - Manajemen risiko adaptif (AdaptiveRiskManager)

- BARU `core/risk_manager.py`: menyetel risk_pct dari kondisi bankroll objektif.
  - Drawdown lunak 20% -> potong stake linear; drawdown keras 35% -> STOP.
  - Rem losing-streak 5 kalah -> stake separuh.
  - Boost winning-streak 8 menang -> +25% (dibatasi, tidak liar).
  - Stop harian: rugi >= 25% modal awal hari -> berhenti hari itu.
- `effective_risk_pct(base)`, `risk_multiplier()`, `should_stop()`, `status()`. Persist `models/risk_state.json` (additive/drop-in).
- Filosofi: kalau ragu, taruhan LEBIH KECIL. Ruin biasanya dari chasing, bukan dari model jelek.
- Tes `_test_risk_manager.py`: sehat->penuh, drawdown keras->stop, lunak->skala turun, rem 5-kalah, boost 8-menang, stop harian, persistence.

## v1.11.0 - Kalibrasi probabilitas (ReliabilityTracker)

- BARU `core/calibration.py`: `ReliabilityTracker` mencatat (confidence, benar?) per-engine; hitung Brier, log-loss, Expected Calibration Error, reliability bins.
- Isotonic regression p_raw -> p_terkalibrasi: pakai scikit-learn bila ada, fallback PAVA (Pool-Adjacent-Violators) murni Python -> jalan di mana saja tanpa dependensi keras.
- `calibrate_predictions()` me-remap + renormalisasi confidence agar cocok dengan hit-rate nyata (lawan over-confidence).
- Persist ke `models/calibration_state.json` (additive, drop-in, round-trip teruji).
- Tes `_test_calibration.py`: PAVA monoton, Brier/log-loss, isotonic mengempiskan 0.8->~0.4 (over-confidence), persistence, renormalisasi.

## v1.10.0 - Higher-order Markov (variable-order)

- BARU `predictors/higher_order_markov.py`: `HigherOrderMarkovEngine` (order 1-4). Auto-pilih order via cross-validation walk-forward (log-loss + margin parsimoni), BACKOFF ke order lebih pendek bila context jarang, Laplace smoothing ke prior frekuensi roda. `support` = jumlah observasi context terpilih (gate bukti tetap jalan).
- Terdaftar di `run_backtest.py` dan laporan audit sebagai "Markov-HO".
- Tes `_test_higher_order_markov.py`: bentuk/normalisasi output, prior saat kosong, deteksi pola siklik (-> edge), data acak (-> no_edge).

## v1.9.0 - Backtest walk-forward per-engine

- BARU `core/backtest.py`: `WalkForwardBacktester` menilai engine apa pun secara walk-forward (top-1/top-3, Brier, log-loss, baseline most-frequent, two-proportion z, verdict edge/no_edge, simulasi profit unit-bet + Sharpe, kurva kalibrasi). Verdict "edge" HANYA jika z>1.96 mengalahkan baseline.
- BARU `run_backtest.py`: CLI backtest data nyata (history.json + CSV). Loader CSV diperbaiki: baca KOLOM `actual_number` saja (tidak lagi dobel-hitung `predicted_number`).
- Laporan audit kini memuat tabel "Backtest walk-forward semua engine" (guarded/lazy import, aman walau TF tidak ada).
- Hasil data nyata (235 spin): AI-Optimal 40.2% = baseline 40.2% (z=0), Markov 38.7% (z=-0.31), Heuristic 32.5% -> SEMUA no_edge. Mengonfirmasi roda fair, tidak ada edge statistik.
- Tes: `_test_backtest.py` (pola tersemat -> edge; acak -> no_edge; data kurang; kurva kalibrasi).

# Changelog — Enhancement Pass oleh Notion AI

## v1.8.0 — Logging snapshot taruhan penuh (fondasi audit) (oleh Mahapatih)
- **Masalah yang diperbaiki**: `data/tracker.py` dulu hanya menyimpan `predicted_number` saat MENANG; saat kalah field-nya null, dan angka yang dipertaruhkan, confidence, EV, serta support TIDAK pernah dilog. Akibatnya audit akurasi/ROI per-angka mustahil dihitung jujur.
- **Fix (snapshot penuh tiap ronde)**: `record_result()` kini menerima `bet_snapshot` (daftar `{number, token_bet, confidence, ev_per_token, is_positive_ev, support}`) + `engine_used`, dan menyimpannya sebagai field `bets` di SETIAP record — menang maupun kalah. Backward-compatible: record lama otomatis dianggap `bets: []`.
- **Analitik baru**: `Tracker.get_per_number_bet_stats()` -> per-angka {#bet, menang, hit-rate, token dipasang, profit bersih, ROI}; `Tracker.get_engine_bet_distribution()` -> performa per-engine. Atribusi profit per-angka dijamin **menjumlah persis** ke profit terealisasi (diverifikasi di test).
- **Audit report**: bagian baru "9. Taruhan per-angka" + tabel "Performa per-engine" di `laporan_audit.md`. Flag kualitas data kini cerdas: tahu kalau snapshot penuh sudah aktif.
- **ViewModel/UI**: `get_predictions()` menyimpan `current_predictions`; `process_new_actual()` + `_on_confirm` meneruskan snapshot & nama engine yang aktif.
- **Verifikasi (`_test_logging_snapshot.py`, semua lulus)**: (1) snapshot tersimpan saat menang DAN kalah; (2) backward-compat record lama; (3) stats per-angka benar + konsisten dengan profit total.

## v1.7.1 — Reality-check taruhan: tidak ada engine yang bisa bertaruh "ngawur" lagi (oleh Mahapatih)
- **Bug**: saat engine **TF-LSTM dipilih langsung**, output mentahnya (yang sering "kolaps" pede ke satu angka langka, mis. 30 atau 40) lolos ke penyizing taruhan TANPA pengaman — karena evidence-gate di `kelly_allocation` hanya aktif untuk engine yang punya field `support` (Markov/Bayes). LSTM & Heuristic tidak punya, jadi kepedean palsunya langsung jadi taruhan "bet 1" pada angka langka -> kalah terus.
- **Akar masalah**: confidence yang dilaporkan SATU model BUKAN probabilitas nyata. Model overfit bisa "yakin" pada angka yang sebenarnya cuma ~1.7%.
- **Fix (reality-check universal)**: sebelum token dipertaruhkan oleh engine APAPUN, tiap pick kini dicek-silang ke posterior Bayesian (dari frekuensi NYATA yang diamati). Stake hanya dipertahankan bila angka itu edge +EV yang robust (batas bawah kredibel melewati break-even); kalau tidak -> di-nol-kan (SKIP). Engine tetap boleh MENEBAK angkanya untuk ditampilkan, tapi tidak boleh mempertaruhkan token tanpa bukti statistik. Hasilnya: SEMUA engine kini seaman AI-Optimal soal risiko token.
- **Catatan jujur**: di roda adil, efeknya semua engine akan sering SKIP (token_bet 0) — itu BENAR & sehat. Taruhan baru muncul kalau ada bias nyata. Untuk prediksi terbaik, tetap pakai **AI-Optimal**.

## v1.7.0 — Engine "AI-Optimal": predictor terbaik secara matematis (oleh Mahapatih)
- **Kenapa ini "terbaik" — dan kenapa BUKAN neural net lebih besar**: hasil roda adalah tarikan i.i.d. dari distribusi kategori tetap. Untuk proses seperti itu, predictor Bayes-optimal berbentuk *closed-form*: posterior-predictive **Dirichlet-Multinomial**. Tidak ada LSTM/Markov/"pola" yang bisa mengalahkannya karena tidak ada sinyal urutan untuk dieksploitasi — log 235 spin milik user membuktikan LSTM (27%) kalah telak dari baseline frekuensi (44%). Model besar di sini cuma overfit noise.
- **`predictors/bayesian_optimal.py` (BayesianOptimalEngine)**: estimator frekuensi dengan prior Dirichlet di-seed dari layout fisik roda (sehat sejak spin #1, konvergen ke frekuensi nyata yang diamati), plus **interval kredibel Beta** per angka (tahu seberapa yakin).
- **Mesin EV/edge ber-gate statistik (inti "grinding")**: tiap angka punya payout; taruhan untung hanya jika `p*(payout+1)-1 > 0`. Sebuah angka ditandai edge NYATA & dipertaruhkan HANYA bila **batas bawah** interval kredibelnya melewati break-even (estimasi pesimistis pun masih +EV). Sizing pakai fractional-Kelly konservatif (pakai `prob_low`). Di roda adil semua -EV → otomatis **SKIP** (tidak menggerus token); begitu bias nyata muncul, langsung pasang.
- **Integrasi**: engine baru "AI-Optimal" bisa dipilih di dropdown UI (default tetap aman). `config/settings.py`: `BAYES_OPT_PRIOR_STRENGTH`, `BAYES_OPT_CI_Z`, `BAYES_OPT_EV_MARGIN`, `BAYES_OPT_MIN_OBS`.
- **Verifikasi (`_test_bayesian_optimal.py`, semua lulus)**: (1) posterior konvergen ke frekuensi sejati; (2) urutan CI benar; (3) data roda adil → SKIP; (4) bias nyata yang disuntik ke angka payout-tinggi (20) → TERDETEKSI & dipasang; (5) spike dini sebelum `MIN_OBS` → tidak pasang.
- **Backtest di 176 spin nyatamu**: verdict **SKIP** (tidak ada edge robust). Penting: angka 20 tampak EV(mean)=+0.19 yang menggoda, TAPI EV-konservatif=−0.44 → engine menolak (perlindungan anti-noise bekerja). Akurasi top-1 AI-Optimal = 38.0% = persis baseline optimal. Tidak ada model yang bisa lebih tinggi di roda adil.

## v1.6.1 — Perbaikan bug "hasil 40 terus" (oleh Mahapatih)
- **Bug**: di awal pemakaian, keempat model dapat bobot kepercayaan sama rata (25%). LSTM yang baru dilatih dari data sedikit "kolaps" menembak satu angka langka (40) dengan pede tinggi, lalu membajak prediksi ensemble selama belasan spin pertama (sebelum EMA sempat mengoreksi).
- **Fix (maturity gate)**: model yang butuh latihan (Markov & LSTM) kini harus MEMBUKTIKAN akurasinya dulu sebelum dapat bobot penuh — bobotnya naik bertahap selama `ENSEMBLE_WARMUP_SPINS` (30) putaran. Fisika & Bayes (valid secara statistik sejak spin #1) jadi jangkar, jadi prediksi awal SELALU mencerminkan roda asli dan tidak bisa dibajak ke angka langka.
- **Verifikasi**: prediksi cold-start pada 117 spin nyata (1.csv) kini = [1, 2, 5] (angka tersering), bukan 40. Regression test `test_coldstart_ignores_overconfident_lstm` ditambahkan & lulus.
- **Catatan**: "saran bet 1 token" itu BENAR & jujur — di roda adil semua taruhan ber-EV negatif (mis. taruh "1" bayar 1:1 -> EV = 0.37x2-1 = -0.26), jadi sistem memang menyarankan taruhan minimum. Itu fitur kejujuran, bukan bug.

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
