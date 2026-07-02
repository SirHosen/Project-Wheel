# -*- coding: utf-8 -*-
"""Detect whether AI work runs on GPU or CPU, and explain WHY when it's CPU.

Used for the green/red compute indicator in the panel and for a status line in
the trainer. Works even when TensorFlow is not installed (reports CPU + a hint).

Green  = a GPU is visible to TensorFlow.
Red    = CPU only (no GPU visible, or TensorFlow not installed).

Important Windows reality: native-Windows TensorFlow is CPU-ONLY from 2.11
onward. GPU on Windows needs either the old TF 2.10 + DirectML plugin, or a
different backend (PyTorch), or WSL2 + CUDA. See explain().

detect_torch() reports PyTorch/CUDA separately -- PyTorch (unlike TensorFlow)
still ships native-Windows CUDA builds, so on an NVIDIA machine it can use the
GPU without WSL.
"""
import platform


def _tf_version_tuple(ver):
    parts = []
    for p in str(ver).split(".")[:2]:
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


def detect():
    """Return a dict describing the TensorFlow compute backend.

    Keys: backend ('GPU'|'CPU'), has_tf (bool), gpus (list[str]),
          label (str), detail (str), tf_version (str|None), os (str).
    """
    info = {"backend": "CPU", "has_tf": False, "gpus": [],
            "label": "CPU (TensorFlow not installed)", "detail": "",
            "tf_version": None, "os": platform.system()}
    try:
        import tensorflow as tf
    except Exception as e:  # pragma: no cover - env dependent
        info["detail"] = f"{type(e).__name__}"
        return info

    info["has_tf"] = True
    info["tf_version"] = getattr(tf, "__version__", "?")
    info["detail"] = f"tensorflow {info['tf_version']}"
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as e:  # pragma: no cover
        gpus = []
        info["detail"] += f"; gpu query failed ({type(e).__name__})"
    names = [getattr(g, "name", str(g)) for g in gpus]
    info["gpus"] = names
    if gpus:
        info["backend"] = "GPU"
        info["label"] = f"GPU x{len(gpus)}" + (f"  ({names[0]})" if names else "")
    else:
        info["backend"] = "CPU"
        info["label"] = "CPU (no GPU visible to TensorFlow)"
    return info


def detect_torch():
    """Report PyTorch/CUDA availability (separate from the TF indicator).

    Keys: has_torch (bool), cuda (bool), torch_version (str|None),
          device_name (str|None).
    """
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


def explain(info=None):
    """Human-readable reason for the current backend + how to change it."""
    d = info or detect()
    if d["backend"] == "GPU":
        return "GPU is active. Nice. (Note: this model is tiny, so CPU is fine too.)"
    if not d["has_tf"]:
        return ("TensorFlow is not installed in this environment. "
                "Install with: pip install -r requirements-ai.txt")
    ver = _tf_version_tuple(d["tf_version"])
    is_windows = d["os"].lower().startswith("win")
    if is_windows and ver >= (2, 11):
        return (
            f"TensorFlow {d['tf_version']} on native Windows is CPU-ONLY. "
            "Google dropped Windows GPU support after TF 2.10, so NO amount of "
            "config will make TF use the GPU here. On an NVIDIA GPU the cleanest "
            "native-Windows path is PyTorch + CUDA (no WSL needed); alternatively "
            "use WSL2 + tensorflow[and-cuda]. Or just keep CPU -- this LSTM is "
            "tiny and trains in seconds either way.")
    if is_windows:
        return (
            f"TensorFlow {d['tf_version']} detected but no GPU is visible. "
            "For GPU on Windows with this TF you would need the DirectML plugin "
            "(tensorflow-cpu==2.10 + tensorflow-directml-plugin, Python <=3.10).")
    return (
        f"TensorFlow {d['tf_version']} sees no GPU. On Linux install "
        "tensorflow[and-cuda] and matching NVIDIA drivers/CUDA, or use WSL2.")


def status_line():
    d = detect()
    tag = "[GPU]" if d["backend"] == "GPU" else "[CPU]"
    base = f"{tag} {d['label']}"
    return f"{base}  ({d['detail']})" if d["detail"] else base
