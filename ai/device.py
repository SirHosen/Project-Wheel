# -*- coding: utf-8 -*-
"""Detect whether AI work runs on GPU or CPU.

Used for the green/red compute indicator in the panel and for a status line in
the trainer. Works even when TensorFlow is not installed (reports CPU + a hint).

Green  = a GPU is visible to TensorFlow (incl. DirectML on Windows).
Red    = CPU only (no GPU visible, or TensorFlow not installed).
"""


def detect():
    """Return a dict describing the compute backend.

    Keys: backend ('GPU'|'CPU'), has_tf (bool), gpus (list[str]),
          label (str, human readable), detail (str).
    """
    info = {"backend": "CPU", "has_tf": False, "gpus": [],
            "label": "CPU (TensorFlow not installed)", "detail": ""}
    try:
        import tensorflow as tf
    except Exception as e:  # pragma: no cover - env dependent
        info["detail"] = f"{type(e).__name__}"
        return info

    info["has_tf"] = True
    info["detail"] = f"tensorflow {getattr(tf, '__version__', '?')}"
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


def status_line():
    d = detect()
    tag = "[GPU]" if d["backend"] == "GPU" else "[CPU]"
    return f"{tag} {d['label']}  ({d['detail']})" if d["detail"] else f"{tag} {d['label']}"
