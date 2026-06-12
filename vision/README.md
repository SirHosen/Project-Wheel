# Vision learning loop

Kamera **MENGAMATI** roda fisik (sudut berhenti -> segmen -> angka). Ini
**observasi, bukan prediksi** -- spin tunggal itu kacau (chaotic) dan tak bisa
diramal. Tapi banyak observasi bisa mengungkap **bias fisik jangka panjang**
(misal poros tak seimbang), dan itulah yang dipelajari loop ini.

## Alur 3 langkah

```
[1] Tangkap            [2] Pelajari                 [3] Upgrade
wheel_cam.py    -->    learn_from_vision.py    -->  BayesianOptimalEngine
  | log tiap             | chi-square vs desain       | observed counts dilipat
  | spin ke CSV          | tulis report .md           | ke posterior Dirichlet
  v                      | tulis wheel_prior.json     v
reports/vision_          v                          rekomendasi taruhan jadi
observations.csv      reports/vision_learning_       grounded ke roda fisik LO
                      report.md                      (restart app)
```

### 1. Tangkap observasi
```bash
# amati 50 spin berturut-turut (tekan Enter antar spin), tiap spin di-log:
python scripts/wheel_cam.py --rounds 50
# dari file video:
python scripts/wheel_cam.py --source clip.mp4
# jangan nge-log (cuma intip):
python scripts/wheel_cam.py --no-log
```
Log ditulis ke `reports/vision_observations.csv`.

### 2. Pelajari (uji bias + report)
```bash
python scripts/learn_from_vision.py                       # report + update prior
python scripts/learn_from_vision.py --no-update            # report saja
python scripts/learn_from_vision.py --stopped-only --min-confidence 0.3
```
- **chi-square goodness-of-fit** terhadap layout desain roda (54 segmen).
- `p >= 0.05` -> tak ada bias signifikan (roda "adil" sesuai desainnya).
- `p < 0.05` -> ada penyimpangan fisik nyata; lihat kolom **residu baku** untuk
  angka yang menyimpang (|z| > 2).
- Report ditulis ke `reports/vision_learning_report.md`.

### 3. Upgrade sistem
`learn_from_vision.py` menulis `models/wheel_prior.json` (observed counts +
statistik). Saat app start, `BayesianOptimalEngine` membaca file ini dan
melipat observasi kamera ke posterior Dirichlet-nya sebagai **spin nyata
tambahan**. Jadi rekomendasi taruhan jadi berdasar roda fisik LO, bukan cuma
layout teoretis.

**Penting:**
- Observasi kamera TIDAK masuk `data/history.json` dan TIDAK mengubah win-rate
  taruhan. Mereka cuma jadi evidence prior. Pemisahan ini disengaja & jujur.
- `models/wheel_prior.json` adalah state lokal (di-gitignore, tidak ikut
  dibagikan). Hapus file itu untuk kembali ke prior layout desain murni.
- Roda adil = engine tetap SKIP semua taruhan (-EV). Tidak ada feature yang bisa
  mengubah itu. Loop ini cuma mendeteksi bias kalau memang ADA.
