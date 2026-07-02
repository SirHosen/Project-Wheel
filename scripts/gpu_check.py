# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Diagnose why the AI runs on CPU vs GPU.

    python scripts/gpu_check.py

Prints your OS, Python, TensorFlow + PyTorch versions, the devices each can see,
and a plain-language explanation + the realistic ways to get GPU on your
platform. On NVIDIA + native Windows, PyTorch-CUDA is the clean path (no WSL).
"""
import platform
import sys

from ai.device import detect, detect_torch, explain


def main():
    d = detect()
    t = detect_torch()
    print("=" * 62)
    print(" COMPUTE DIAGNOSTIC")
    print("=" * 62)
    print(f"OS                 : {platform.system()} {platform.release()}")
    print(f"Python             : {sys.version.split()[0]}")
    print(f"TensorFlow         : {d['tf_version'] or 'not installed'}")
    print(f"  panel backend    : {d['backend']}  ({'GREEN' if d['backend']=='GPU' else 'RED'} indicator)")
    print(f"  GPUs visible     : {d['gpus'] if d['gpus'] else 'none'}")
    print(f"PyTorch            : {t['torch_version'] or 'not installed'}")
    if t["has_torch"]:
        print(f"  CUDA available   : {t['cuda']}")
        print(f"  CUDA device      : {t['device_name'] or 'none'}")
    print("-" * 62)
    print("WHY (TensorFlow / panel indicator):")
    print("  " + explain(d))
    if t["has_torch"] and t["cuda"]:
        print("GOOD: PyTorch CAN use your GPU (" + str(t["device_name"]) + ").")
    print("=" * 62)

    if d["has_tf"]:
        try:
            import tensorflow as tf
            print("tf.config.list_physical_devices():")
            for dev in tf.config.list_physical_devices():
                print(f"   {dev.device_type:4}  {dev.name}")
        except Exception as e:
            print(f"(could not list TF devices: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
