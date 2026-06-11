# -*- coding: utf-8 -*-
"""
app/main.py — Entry point for the Spin Wheel Predictor Application.
"""

import sys
import os

# Ensure the root directory is in PYTHONPATH so absolute imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.viewmodels.main_viewmodel import MainViewModel
from gui.views.main_window import MainWindow

def init_tf_gpu():
    """Inisialisasi TensorFlow + diagnostik GPU/CPU yang jelas saat startup.

    Mencetak versi TF, apakah build-nya mendukung CUDA, dan daftar GPU yang
    terdeteksi. Kalau jatuh ke CPU, langsung menjelaskan KENAPA + cara
    perbaikannya, jadi tidak perlu menebak-nebak lagi.
    """
    import tensorflow as tf
    try:
        from config import settings as _settings
        _app_ver = getattr(_settings, "APP_VERSION", "?")
    except Exception:
        _app_ver = "?"

    print("=" * 64)
    print(" Spin Wheel Predictor - Diagnostik")
    print(f"  App version  : {_app_ver}   (terbaru = 1.28.1)")
    print(f"  TF version   : {tf.__version__}")
    try:
        built_cuda = tf.test.is_built_with_cuda()
    except Exception:
        built_cuda = None
    print(f"  Build CUDA?  : {built_cuda}")

    # Cek ketersediaan GPU saat app pertama dibuka
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0

    if gpu_available:
        print(f"  Mode         : GPU  [OK]  ({len(gpus)} device)")
        for i, gpu in enumerate(gpus):
            print(f"    [{i}] {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"        set_memory_growth gagal: {e}")
    else:
        print("  Mode         : CPU  (GPU TIDAK terdeteksi)")
        print("  Kemungkinan penyebab:")
        if built_cuda is False:
            print("    - TensorFlow ini build CPU-only (mis. venv Windows).")
            print("      Solusi: jalankan dari WSL dgn venv GPU yg sudah di-`source`.")
        else:
            print("    - venv GPU belum di-activate / LD_LIBRARY_PATH belum di-set")
            print("      (TF tidak menemukan libcudnn/libcublas).")
            print("    - Driver NVIDIA WSL belum kebaca -> cek `nvidia-smi`.")
        print("  Cek manual:")
        print("    python -c \"import tensorflow as tf; "
              "print(tf.config.list_physical_devices('GPU'))\"")
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
