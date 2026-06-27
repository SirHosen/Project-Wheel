# Menjalankan Spin Wheel Predictor di Windows NATIVE

Ringkasan jujur dari migrasi WSL -> Windows native.

## TL;DR

| Tujuan | Status di Windows native |
|---|---|
| App + GUI + LSTM jalan native | OK, langsung bisa (mode CPU) |
| Log berisik (oneDNN/absl) hilang | OK, sudah diredam di `app/main.py` |
| OpenCV (webcam / capture layar) | OK, JADI aktif di Windows (di WSL mati) |
| GPU dipakai TensorFlow | TIDAK dengan TF 2.21 (lihat di bawah) |

---

## 1. Kenapa GPU tidak terpakai (ini BUKAN bug)

TensorFlow **>= 2.11 tidak mendukung GPU di Windows native** sama sekali --
ini keputusan upstream Google, bukan masalah kode kamu. Kamu pakai TF 2.21,
jadi walau CUDA/cuDNN terpasang, GPU tetap tidak akan dipakai. TF sendiri yang
mencetak peringatan itu.

> Realita penting: model LSTM proyek ini KECIL (seq=10, 9 kelas, data puluhan
> baris). GPU nyaris tidak mempercepat -- overhead transfer ke GPU sering
> membuat LEBIH lambat dari CPU. Untuk beban kerja ini, **CPU sudah optimal.**

## 2. Jalan paling mulus (REKOMENDASI): Windows native, CPU

```powershell
cd C:\Users\Hosea\Documents\PYTHON-projek\Project-Wheel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-vision.txt   # supaya OpenCV aktif
python app\main.py
```

Log oneDNN/absl sudah hilang (diredam di `app/main.py`), dan diagnostik
sekarang sadar-OS: dia tidak lagi menyuruh kamu balik ke WSL.

## 3. Kalau MEMANG ingin GPU menyala di Windows native

Pilih salah satu. Keduanya mengunci TensorFlow ke **2.10** (model otomatis
disimpan sebagai `.h5`, sudah ditangani kode).

### Opsi A - DirectML (GPU apa pun: NVIDIA / AMD / Intel)

```powershell
python -m venv .venv-dml
.venv-dml\Scripts\activate
pip install -r requirements-windows-gpu.txt
python app\main.py
```

Verifikasi:
```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Opsi B - CUDA native (khusus NVIDIA)

- `pip install tensorflow==2.10`
- Pasang **CUDA Toolkit 11.2** + **cuDNN 8.1** (versi persis ini; TF 2.10 hanya
  cocok dengan kombinasi tsb).
- Jalankan `python app\main.py`.

## 4. OpenCV di Windows (kabar baik)

Di WSL, OpenCV mati karena WSL tidak punya akses webcam (`/dev/video0`) dan
tidak ada layar nyata untuk capture. Di **Windows native keduanya jalan**:

```powershell
pip install -r requirements-vision.txt
python scripts\wheel_cam.py --source 0                 # webcam
python scripts\wheel_cam.py --source screen --list-monitors
```

> Ingat disclaimer modul vision: ini MENGAMATI roda (sudut/segmen berhenti),
> BUKAN meramal spin berikutnya.

## 5. Yang berubah di kode (untuk migrasi ini)

- `app/main.py` - redam log sebelum import TF; diagnostik sadar-OS (berhenti
  menyuruh pakai WSL di Windows); cetak status OpenCV; deteksi DirectML.
- `predictors/tf_lstm_engine.py` - `_model_path()` otomatis pakai `.h5` di
  TF < 2.11 supaya save/load model tetap jalan di jalur GPU Windows.
- `requirements-windows-gpu.txt` - paket jalur DirectML (baru).

Tidak ada perubahan logika taruhan/statistik. Test tetap 29/29 PASS.
