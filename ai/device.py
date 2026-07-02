# -*- coding: utf-8 -*-
"""Detect whether AI work runs on GPU or CPU, and explain WHY when it's CPU.

The training backend is PyTorch, so the GREEN/RED panel indicator reflects
PyTorch/CUDA:
    GREEN = a CUDA GPU is visible to PyTorch.
    RED   = CPU only (no CUDA, or PyTorch not installed).

PyTorch (unlike TensorFlow) ships native-Windows CUDA builds, so on an NVIDIA
machine it uses the GPU without WSL. TensorFlow is intentionally NOT touched
here -- this project is fully on PyTorch.
"""
import platform


def detect_torch():
    """PyTorch/CUDA availability. Keys: has_torch, cuda, torch_version,
    device_name."""
    out = {"has_torch": False, "cuda": False, "torch_version": None,
           "device_name": None}
    try:
        import torch
    except Exception:
        return out
    out["has_torch"] = True
    out["torch_version"] = getattr(torch, "__version__", "?")
    try:
        if torch.cuda.is_available():
            out["cuda"] = True
            out["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def detect():
    """The panel indicator, based on the ACTIVE training backend (PyTorch).

    Keys: backend ('GPU'|'CPU'), framework (str|None), has_torch (bool),
          gpus (list[str]), label (str), detail (str), os (str).
    """
    t = detect_torch()
    info = {"backend": "CPU", "framework": None, "has_torch": t["has_torch"],
            "gpus": [], "label": "CPU (PyTorch not installed)", "detail": "",
            "os": platform.system()}
    if not t["has_torch"]:
        return info
    info["framework"] = "pytorch"
    info["detail"] = f"pytorch {t['torch_version']}"
    if t["cuda"]:
        info["backend"] = "GPU"
        info["gpus"] = [t["device_name"]] if t["device_name"] else []
        info["label"] = f"GPU ({t['device_name']})"
    else:
        info["backend"] = "CPU"
        info["label"] = "CPU (PyTorch, no CUDA GPU visible)"
    return info


def explain(info=None):
    """Human-readable reason for the current backend + how to change it."""
    d = info or detect()
    if d["backend"] == "GPU":
        return (f"GPU is active via PyTorch ({d['gpus'][0] if d['gpus'] else 'cuda'}). "
                "Nice. (Note: this model is tiny, so CPU is fine too.)")
    if not d["has_torch"]:
        return ("PyTorch is not installed. Install the CUDA build for your NVIDIA "
                "GPU: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121")
    hint = ("You likely installed the CPU-only wheel. Reinstall the CUDA build: "
            "pip uninstall torch -y && pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121")
    if d["os"].lower().startswith("win"):
        return ("PyTorch is installed but no CUDA GPU is visible. " + hint
                + "  (Also check your NVIDIA driver is up to date.)")
    return "PyTorch is installed but sees no CUDA GPU. " + hint


def status_line():
    d = detect()
    tag = "[GPU]" if d["backend"] == "GPU" else "[CPU]"
    base = f"{tag} {d['label']}"
    return f"{base}  ({d['detail']})" if d["detail"] else base
