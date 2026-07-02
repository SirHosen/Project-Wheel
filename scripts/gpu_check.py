# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Diagnose whether the AI trains on CPU or GPU.

    python scripts/gpu_check.py

The trainer uses PyTorch, so the panel indicator follows PyTorch/CUDA. This does
NOT import TensorFlow (the project is fully on PyTorch), so it stays quiet.
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
    print(f"Trainer backend    : PyTorch")
    print(f"  panel indicator  : {d['backend']}  ({'GREEN' if d['backend']=='GPU' else 'RED'})")
    print(f"  PyTorch          : {t['torch_version'] or 'not installed'}")
    if t["has_torch"]:
        print(f"  CUDA available   : {t['cuda']}")
        print(f"  CUDA device      : {t['device_name'] or 'none'}")
    print("-" * 62)
    print("WHY:")
    print("  " + explain(d))
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
