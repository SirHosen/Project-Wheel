# Laporan Audit - Spin Wheel Predictor

- Dibuat: 2026-06-08T03:23:17
- Versi app: 1.6.1
- Total putaran tercatat: 59
- Rentang data: 2026-06-08T02:36:58.756412 s/d 2026-06-08T03:23:03.404186

## 1. Bankroll
- Modal awal: 10 token
- Modal sekarang: 10 token
- Profit terealisasi: +0 token
- ROI: 0.0%
- Max drawdown: 8 token (53.3%)

## 2. Performa
- Win-rate tebakan teratas: 67.8% (40/59)
- Ronde bertaruh: 12 | Ronde skip: 47
- Win-rate saat bertaruh: 16.7%
- Rata-rata profit per ronde bertaruh: +0.00 token

## 3. Statistik per ronde
- Terbaik: +5 | Terburuk: -1 | Rata2: +0.00 | Std: 1.02
- Streak menang sekarang: 0 | Max menang: 7 | Max kalah beruntun: 3

## 4. Distribusi angka vs roda (54 segmen)
| Angka | Muncul | Observasi % | Teori roda % | Deviasi |
|------:|-------:|------------:|-------------:|--------:|
| 1 | 26 | 44.1% | 37.0% | +7.0% |
| 2 | 15 | 25.4% | 24.1% | +1.3% |
| 5 | 8 | 13.6% | 13.0% | +0.6% |
| 8 | 1 | 1.7% | 7.4% | -5.7% |
| 10 | 5 | 8.5% | 7.4% | +1.1% |
| 15 | 2 | 3.4% | 3.7% | -0.3% |
| 20 | 0 | 0.0% | 3.7% | -3.7% |
| 30 | 1 | 1.7% | 1.9% | -0.2% |
| 40 | 1 | 1.7% | 1.9% | -0.2% |

## 5. Uji kewajaran roda (chi-square)
- chi^2 = 5.75 | df = 8 | kritis(0.05) = 15.507
- Putusan: RODA WAJAR (acak) - tidak ada bias frekuensi signifikan.

## 6. Autokorelasi (angka berturut sama)
- Rate ulang lag-1: 31.0% vs harapan acak 28.6%
- Putusan: random

## 7. Edge Markov (walk-forward)
- Akurasi Markov top-1: 38.5% vs baseline 42.3% atas 52 ronde
- z = -0.40 -> Belum ada edge: model tidak mengalahkan tebakan paling sering.

## 8. Kualitas data & rekomendasi audit
- predicted_number hanya tersimpan saat MENANG; angka yang dipertaruhkan, confidence, support, dan EV per ronde belum dilog -> audit akurasi taruhan per-angka belum bisa penuh. Rekomendasi: simpan snapshot prediksi tiap ronde.
