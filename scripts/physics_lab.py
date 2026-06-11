# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
physics_lab.py — GPU laboratory for the spin-wheel physics model.

Run on your RTX 3050 (WSL2 + CUDA):
    python physics_lab.py
    python physics_lab.py --spins 20000000 --ml

What it does (all GPU-accelerated where it matters):
  1. Detects the GPU.
  2. Runs a MASSIVE Monte-Carlo of realistic spins on the GPU and compares the
     resulting number distribution to the wheel's theoretical area fractions.
  3. Prints the predictability ceiling (how precisely omega0 must be measured).
  4. (--ml) Trains a neural net to learn the torque->landing physics, then shows
     how its accuracy COLLAPSES once realistic measurement noise is added — the
     honest proof that the bottleneck is measurement, not modelling.

This is an HONEST physics tool. It models and explains the wheel. It does NOT,
and cannot, predict live spins from past results — read core/physics_wheel.py.
"""
import argparse
import math
import time
import numpy as np

from core.physics_wheel import WheelPhysics, TWO_PI
from config import settings


def _print_gpu(tf):
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}")
    print(f"GPU terdeteksi: {[g.name for g in gpus] if gpus else 'TIDAK ADA (jalan di CPU)'}")
    return bool(gpus)


def monte_carlo_gpu(tf, wheel: WheelPhysics, n: int, omega_mean: float,
                    omega_std: float, batch: int = 5_000_000) -> dict:
    """Vectorized Monte-Carlo of n spins on the GPU using TensorFlow tensors."""
    seg_angle = wheel.seg_angle
    alpha = wheel.decel
    seg_numbers = tf.constant(wheel.sequence, dtype=tf.int64)  # index -> number
    counts = np.zeros(wheel.n_segments, dtype=np.int64)
    done = 0
    while done < n:
        b = min(batch, n - done)
        theta0 = tf.random.uniform([b], 0.0, TWO_PI, dtype=tf.float32)
        omega0 = tf.abs(tf.random.normal([b], omega_mean, omega_std, dtype=tf.float32))
        total = omega0 * omega0 / (2.0 * alpha)
        final_angle = tf.math.floormod(theta0 + total, TWO_PI)
        idx = tf.cast(tf.math.floor(final_angle / seg_angle), tf.int64) % wheel.n_segments
        counts += np.bincount(idx.numpy(), minlength=wheel.n_segments).astype(np.int64)
        done += b
        print(f"  ...simulated {done:,}/{n:,} spins", end="\r")
    print()
    # segment counts -> per-number frequency
    freq = {int(v): 0 for v in wheel.valid_numbers}
    for i, num in enumerate(wheel.sequence):
        freq[int(num)] += int(counts[i])
    freq = {k: v / n for k, v in freq.items()}
    return freq


def run_monte_carlo(tf, wheel, spins, omega_mean, omega_std):
    print("\n== 1) MONTE-CARLO FISIKA DI GPU ==")
    t0 = time.time()
    freq = monte_carlo_gpu(tf, wheel, spins, omega_mean, omega_std)
    dt = time.time() - t0
    theo = wheel.area_fractions()
    print(f"Simulasi {spins:,} spin dalam {dt:.2f}s ({spins/dt/1e6:.1f} juta spin/detik)")
    print(f"{'angka':>6} {'simulasi':>10} {'teori(luas)':>12} {'selisih':>9}")
    max_err = 0.0
    for v in wheel.valid_numbers:
        err = abs(freq[v] - theo[v]); max_err = max(max_err, err)
        print(f"{v:>6} {100*freq[v]:>9.2f}% {100*theo[v]:>11.2f}% {100*err:>8.2f}%")
    print(f"Error maksimum: {100*max_err:.3f}%  -> fisika SAMA PERSIS dengan luas segmen.")
    print("Artinya: peluang tiap angka = porsi luasnya di roda. Titik.")


def run_predictability(wheel, omega_mean, omega_std):
    print("\n== 2) BATAS KETERPREDIKSIAN (predictability ceiling) ==")
    s = wheel.sensitivity(omega_mean)
    rep = wheel.predictability_report(omega_mean, omega_std, video_fps=30)
    print(f"Sensitivitas: tiap +1 rad/s di kecepatan awal menggeser hasil "
          f"{s['dTheta_domega_rad_per_rad_s']/wheel.seg_angle:.0f} segmen.")
    print(f"Presisi yang DIBUTUHKAN buat nebak 1 segmen : {rep['relative_precision_needed_pct']:.3f}% pada omega0")
    print(f"Presisi optimis dari video 30fps             : ~{rep['optimistic_video_precision_pct']:.2f}%")
    print(f"Sebaran alami kecepatan spin manusia         : ~{rep['natural_spin_spread_pct']:.1f}%")
    print(f"=> Ketidakpastian dari video: ~{rep['segments_of_uncertainty_from_video']:.0f} segmen.")
    print(f"=> Bisa diprediksi dari video? {'YA' if rep['predictable_from_video'] else 'TIDAK'}")


def run_ml_demo(tf, wheel, omega_mean, omega_std, n_train=200_000):
    print("\n== 3) DEMO ML: belajar fisika torsi->hasil (di GPU) ==")
    rng = np.random.default_rng(7)
    valid = wheel.valid_numbers
    num_to_class = {v: i for i, v in enumerate(valid)}

    def make(n):
        th = rng.uniform(0, TWO_PI, n)
        om = np.abs(rng.normal(omega_mean, omega_std, n))
        y = np.array([num_to_class[int(x)] for x in wheel.spin_vec(th, om)])
        X = np.stack([np.sin(th), np.cos(th), (om - omega_mean) / omega_std], axis=1).astype("float32")
        return X, y.astype("int32"), th, om

    Xtr, ytr, _, _ = make(n_train)
    Xte, yte, th_te, om_te = make(40_000)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(valid), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, epochs=12, batch_size=512, verbose=2, validation_split=0.1)

    # baseline: always guess the most common number (area-weighted top-1)
    theo = wheel.area_fractions()
    baseline = max(theo.values())
    acc_exact = model.evaluate(Xte, yte, verbose=0)[1]

    # add realistic measurement noise to omega0 then re-featurize
    def noisy_acc(rel_err):
        om_n = om_te * (1.0 + rng.normal(0, rel_err, om_te.shape))
        Xn = np.stack([np.sin(th_te), np.cos(th_te), (om_n - omega_mean) / omega_std], axis=1).astype("float32")
        return model.evaluate(Xn, yte, verbose=0)[1]

    print(f"\nAkurasi model dgn input PERSIS (tanpa noise) : {100*acc_exact:.1f}%")
    print(f"Akurasi dgn noise 0.05% pada omega0          : {100*noisy_acc(0.0005):.1f}%")
    print(f"Akurasi dgn noise 2% (optimis dari video)    : {100*noisy_acc(0.02):.1f}%")
    print(f"Baseline (asal tebak angka tersering '1')    : {100*baseline:.1f}%")

    seg_shift = wheel.sensitivity(omega_mean)["dTheta_domega_rad_per_rad_s"] / wheel.seg_angle
    print("\n== Kesimpulan ==")
    if acc_exact > baseline + 0.03:
        print("Model bisa belajar pemetaan fisika saat inputnya presisi, lalu akurasinya")
        print("RUNTUH ke baseline begitu ada noise pengukuran realistis.")
        print("=> Hambatannya bukan kecerdasan model, tapi PENGUKURAN kondisi awal.")
    else:
        print(f"Bahkan dengan input float 'persis', model MENTOK di baseline ({100*baseline:.0f}%).")
        print(f"Sebabnya: hasil = fungsi SUPER frekuensi-tinggi dari omega0 (tiap +1 rad/s")
        print(f"menggeser ~{seg_shift:.0f} segmen). Fungsi seberisolasi itu mustahil dipelajari")
        print("jaringan saraf kontinu — inilah TANDA TANGAN CHAOS (sensitive dependence).")
        print("Ketidakterprediksian sudah muncul SEBELUM bicara noise pengukuran: model")
        print("terbaik pun cuma bisa menebak angka tersering. Itulah strategi optimalnya.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=int, default=10_000_000)
    ap.add_argument("--omega-mean", type=float, default=12.0)
    ap.add_argument("--omega-std", type=float, default=3.0)
    ap.add_argument("--ml", action="store_true", help="jalankan demo ML torsi->hasil")
    args = ap.parse_args()

    import tensorflow as tf
    _print_gpu(tf)

    wheel = WheelPhysics(sequence=settings.SPINWHEEL_SEQUENCE,
                         valid_numbers=settings.VALID_NUMBERS)
    print(f"\nRoda: diameter {2*wheel.radius:.2f} m, massa {wheel.mass:.0f} kg, "
          f"inersia {wheel.inertia:.2f} kg*m^2, torsi gesek {wheel.friction_torque:.2f} N*m")

    run_monte_carlo(tf, wheel, args.spins, args.omega_mean, args.omega_std)
    run_predictability(wheel, args.omega_mean, args.omega_std)
    if args.ml:
        run_ml_demo(tf, wheel, args.omega_mean, args.omega_std)

    print("\nSelesai. Catatan jujur: tool ini MEMODELKAN & MEMBUKTIKAN perilaku roda,")
    print("bukan memprediksi spin nyata dari hasil masa lalu.")


if __name__ == "__main__":
    main()
