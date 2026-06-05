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
    import tensorflow as tf
    # Cek ketersediaan GPU saat app pertama dibuka
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    if gpu_available:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
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
    main()
