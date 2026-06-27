# -*- coding: utf-8 -*-
"""
app/main.py - Entry point for the Spin Wheel Predictor Application.

Lintas-platform: jalan di Windows native, WSL2, dan Linux. Diagnostik startup
menjelaskan kondisi GPU/CPU + status OpenCV secara jujur sesuai OS yang dipakai.
"""

import sys
import os

# ----------------------------------------------------------------------------
# Redam log berisik SEBELUM TensorFlow di-import. Env var ini HARUS di-set lebih
# dulu; kalau setelah `import tensorflow` efeknya tidak berlaku. Ini yang
# menghilangkan spam "oneDNN custom operations are on" + pesan absl/STDERR.
# ----------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 0=semua .. 3=error saja
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # matikan notice oneDNN
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

# Pastikan root project ada di PYTHONPATH supaya import absolut jalan.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.viewmodels.main_viewmodel import MainViewModel
from gui.views.main_window import MainWindow


def _opencv_status_line():
    """Status OpenCV untuk diagnostik (aman walau cv2 tidak terpasang)."""
    try:
        from vision.camera import opencv_status
        return opencv_status()
    except Exception as e:
        return f"status tidak diketahui ({type(e).__name__})"


def init_tf_gpu():
    """Inisialisasi TensorFlow + diagnostik GPU/CPU yang jelas & sadar-OS.

    Mencetak versi TF, dukungan CUDA, daftar GPU, dan status OpenCV. Kalau jatuh
    ke CPU, langsung menjelaskan KENAPA sesuai OS + cara mengaktifkan GPU yang
    benar (di Windows native: DirectML atau TF 2.10), jadi tidak menebak lagi.
    """
    import platform
    import tensorflow as tf
    try:
        from config import settings as _settings
        _app_ver = getattr(_settings, "APP_VERSION", "?")
    except Exception:
        _app_ver = "?"

    is_windows = platform.system() == "Windows"
    try:
        ver = tuple(int(x) for x in tf.__version__.split(".")[:2])
    except Exception:
        ver = (0, 0)

    print("=" * 64)
    print(" Spin Wheel Predictor - Diagnostik")
    print(f"  App version  : {_app_ver}   (terbaru = 1.30.1)")
    print(f"  TF version   : {tf.__version__}")
    print(f"  OS           : {platform.system()} {platform.release()}")
    try:
        built_cuda = tf.test.is_built_with_cuda()
    except Exception:
        built_cuda = None
    print(f"  Build CUDA?  : {built_cuda}")

    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0

    if gpu_available:
        # DirectML & CUDA sama-sama muncul sebagai device 'GPU' di sini.
        print(f"  Mode         : GPU  [OK]  ({len(gpus)} device)")
        for i, gpu in enumerate(gpus):
            print(f"    [{i}] {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                # DirectML tidak selalu mendukung memory growth -> abaikan halus.
                print(f"        set_memory_growth: {e}")
    else:
        print("  Mode         : CPU  (GPU TIDAK terdeteksi)")
        if is_windows and ver >= (2, 11):
            print("  Penyebab     : TensorFlow >= 2.11 TIDAK mendukung GPU di")
            print("                 Windows native (keputusan upstream Google),")
            print("                 walau CUDA/cuDNN sudah terpasang.")
            print("  Aktifkan GPU di Windows native (pilih salah satu):")
            print("    1) DirectML : pip install tensorflow-cpu==2.10.0 "
                  "tensorflow-directml-plugin")
            print("                  (GPU apa pun: NVIDIA/AMD/Intel, lewat DirectX12)")
            print("    2) CUDA 2.10: pip install tensorflow==2.10  + CUDA 11.2 "
                  "+ cuDNN 8.1  (khusus NVIDIA)")
            print("                  lihat requirements-windows-gpu.txt")
            print("  Catatan jujur: model LSTM proyek ini KECIL; GPU nyaris tidak")
            print("                 mempercepat -- CPU sudah cukup & sering lebih cepat.")
        elif built_cuda is False:
            print("    - TensorFlow ini build CPU-only.")
            print("      Di Windows native pakai DirectML/TF2.10; di WSL/Linux")
            print("      pakai requirements-gpu.txt (tensorflow[and-cuda]).")
        else:
            print("    - venv GPU belum di-activate / lib CUDA tidak ketemu")
            print("      (TF tidak menemukan libcudnn/libcublas).")
            print("    - cek driver: `nvidia-smi`.")
    print(f"  OpenCV       : {_opencv_status_line()}")
    print("=" * 64)
    return gpu_available


def main():
    # Initialize TF GPU
    gpu_available = init_tf_gpu()

    # Initialize ViewModel (Model Layer)
    vm = MainViewModel()
    vm.gpu_available = gpu_available

    # Initialize View (GUI Layer)
    app = MainWindow(vm)

    # Run the application
    app.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spin Wheel Predictor")
    parser.add_argument(
        "--serve", action="store_true",
        help="Run the REST API server instead of the GUI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="API bind host.")
    parser.add_argument("--port", type=int, default=8000, help="API bind port.")
    parser.add_argument(
        "--no-fastapi", action="store_true",
        help="Force the stdlib http.server fallback even if FastAPI is installed.",
    )
    args = parser.parse_args()

    if args.serve:
        # REST mode: skip the GUI + TF GPU diagnostic; the ViewModel (and TF)
        # is built lazily only when the first /predict request arrives.
        from api.server import serve
        serve(host=args.host, port=args.port, prefer_fastapi=not args.no_fastapi)
    else:
        main()
