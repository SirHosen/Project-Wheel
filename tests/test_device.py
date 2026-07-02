# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Compute detector: must always return a well-formed dict (GPU or CPU) and
never raise, whether or not PyTorch is installed. The panel indicator follows
the PyTorch/CUDA backend.
"""
from ai import device


def test_detect_shape():
    d = device.detect()
    assert d["backend"] in ("GPU", "CPU"), d
    assert isinstance(d["gpus"], list)
    assert isinstance(d["has_torch"], bool)
    assert isinstance(d.get("label", ""), str) and d["label"]
    # The GREEN/RED rule the UI relies on: GPU only if torch + a visible GPU.
    if d["backend"] == "GPU":
        assert d["has_torch"] and d["gpus"]
    print("OK device.detect:", d["backend"], "-", d["label"])


def test_torch_and_tf_probes():
    t = device.detect_torch()
    assert isinstance(t["has_torch"], bool) and isinstance(t["cuda"], bool)
    tf = device.detect_tf()
    assert isinstance(tf["has_tf"], bool)
    print("OK detect_torch/detect_tf shapes")


def test_status_line():
    s = device.status_line()
    assert isinstance(s, str) and ("[GPU]" in s or "[CPU]" in s)
    print("OK device.status_line:", s)


if __name__ == "__main__":
    test_detect_shape()
    test_torch_and_tf_probes()
    test_status_line()
    print("ALL CHECKS PASSED")
