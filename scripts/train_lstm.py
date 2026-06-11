# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
train_lstm.py - Heavy GPU training + honest backtest for the TF-LSTM engine.

Jalankan ini di PC kamu (Ryzen + RTX 3080) yang sudah punya TensorFlow + CUDA.
Script ini akan:
  1. Memuat riwayat spin asli kamu (data/history.json, opsional + CSV).
  2. Memastikan GPU benar-benar dipakai.
  3. Melatih LSTM dalam (deep) dengan validation split + early stopping
     (memanfaatkan Tensor Cores RTX 3080).
  4. Menjalankan backtest walk-forward: akurasi top-1 model vs baseline
     'angka paling sering' -> uji jujur apakah model punya edge nyata
     pada DATA kamu sendiri.
  5. Menyimpan model terlatih supaya app langsung memuatnya saat dibuka.

Pemakaian:
    python train_lstm.py                 # pakai data/history.json
    python train_lstm.py --csv 1.csv     # tambah actuals dari sebuah CSV
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings


def load_history(csv_path=None):
    nums = []
    hist_path = os.path.join(os.path.dirname(__file__), "data", "history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data.get("history", []):
            n = rec.get("actual_number")
            if n in settings.VALID_NUMBERS:
                nums.append(n)
    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        for col in ("actual_number", "actual", "winning_number", "number", "result"):
            if col in df.columns:
                for v in df[col].tolist():
                    try:
                        iv = int(v)
                    except (ValueError, TypeError):
                        continue
                    if iv in settings.VALID_NUMBERS:
                        nums.append(iv)
                break
    return nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Optional CSV of past results")
    ap.add_argument("--compare-attention", action="store_true",
                    help="Backtest attention vs vanilla LSTM and report the lift")
    args = ap.parse_args()

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}")
    print(f"GPU terdeteksi: {[g.name for g in gpus] if gpus else 'TIDAK ADA (pakai CPU)'}")

    from predictors.tf_lstm_engine import TfLstmEngine

    history = load_history(args.csv)
    print(f"Total data riwayat: {len(history)} putaran")
    if len(history) < 60:
        print("PERINGATAN: data < 60 putaran. Hasil backtest belum bisa dipercaya.")

    engine = TfLstmEngine()

    print("\n== Backtest jujur (walk-forward) ==")
    report = engine.evaluate_backtest(history)
    if report["verdict"] == "insufficient":
        print("Data belum cukup untuk backtest yang valid.")
    else:
        print(f"Akurasi model (top-1): {report['model_acc'] * 100:.1f}%")
        print(f"Akurasi baseline      : {report['baseline_acc'] * 100:.1f}%")
        print(f"Selisih (lift)        : {report['lift'] * 100:+.1f}%")
        print(f"Putusan               : {report['verdict']}")

    if args.compare_attention:
        from predictors.tf_lstm_engine import compare_attention_vs_vanilla
        print("\n== Perbandingan: attention vs vanilla (walk-forward) ==")
        cmp = compare_attention_vs_vanilla(history)
        a, v = cmp["attention"], cmp["vanilla"]
        if a["model_acc"] is None or v["model_acc"] is None:
            print("Data belum cukup untuk perbandingan yang valid.")
        else:
            print(f"Attention top-1 : {a['model_acc'] * 100:.1f}%")
            print(f"Vanilla   top-1 : {v['model_acc'] * 100:.1f}%")
            lift = cmp["attention_minus_vanilla"]
            print(f"Selisih (att-van): {lift * 100:+.1f}%")
            print(f"Putusan          : {cmp['verdict']}")

    print("\n== Training penuh + simpan model ==")
    engine.train(history, bulk=True)
    if engine.save():
        print(f"Model tersimpan di: {engine._model_path()}")
    print("Selesai. App akan otomatis memuat model ini saat dibuka.")


if __name__ == "__main__":
    main()
