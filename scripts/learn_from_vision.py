# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""scripts/learn_from_vision.py - ubah observasi kamera jadi evidence.

Membaca reports/vision_observations.csv (ditulis scripts/wheel_cam.py), lalu:
  1. Menguji apakah roda FISIK menyimpang dari layout DESAIN-nya pakai uji
     chi-square goodness-of-fit (harapan = n * porsi segmen desain).
  2. Menulis report buat manusia -> reports/vision_learning_report.md.
  3. Meng-update models/wheel_prior.json dengan observed counts supaya
     BayesianOptimalEngine melipatnya ke posterior sebagai spin NYATA tambahan
     (restart app untuk dipakai). Statistik win-rate taruhan TIDAK tersentuh.

FRAMING JUJUR: roda yang adil akan kasih p-value tinggi (tak ada bias terdeteksi)
dan engine akan terus SKIP taruhan -EV. Cuma bias fisik yang nyata secara
statistik (p kecil + sebuah angka yang batas bawah kredibelnya melewati titik
impas) yang bikin engine bertaruh. Ini mencari bias; BUKAN meramal spin berikut.

    python scripts/learn_from_vision.py
    python scripts/learn_from_vision.py --no-update             # report saja
    python scripts/learn_from_vision.py --stopped-only --min-confidence 0.3
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

from config import settings
from core.wheel_bias import chi_square_gof, design_distribution, standardized_residual
from vision.observation_log import OBSERVATIONS_PATH, load_observations

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEEL_PRIOR_PATH = os.path.join(_ROOT, "models", "wheel_prior.json")
REPORT_PATH = os.path.join(_ROOT, "reports", "vision_learning_report.md")


def build_report(usable, counts, design, expected, chi2, dof, p, biased, stopped_only, no_update, min_spins):
    n = len(usable)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = []
    L.append("# Laporan Pembelajaran Vision\n")
    L.append(f"_Dihasilkan: {ts} | App {getattr(settings, 'APP_VERSION', '?')}_\n")
    L.append("\n## Ringkasan\n")
    extra = " (hanya spin yang benar-benar berhenti)" if stopped_only else ""
    L.append(f"- Total spin teramati (usable): **{n}**{extra}")
    L.append(f"- Chi-square goodness-of-fit vs layout desain: **chi2 = {chi2:.2f}**, dof = {dof}, **p = {p:.4f}**")
    if n < min_spins:
        L.append(f"- WARNING: data masih sedikit (<{min_spins} spin). Hasil belum bisa dipercaya; kumpulkan lebih banyak dulu.")
    if biased:
        L.append("- RODA FISIK MENYIMPANG SIGNIFIKAN dari desain (p < 0.05). Ada kandidat bias yang bisa diuji untuk +EV.")
    else:
        L.append("- Tidak ada bukti bias signifikan (p >= 0.05). Roda berperilaku sesuai desain; engine akan tetap SKIP taruhan -EV (itu jujur & benar).")
    L.append("\n## Frekuensi per angka\n")
    L.append("| Angka | Teramati | Harapan (desain) | Porsi desain | Residu baku |")
    L.append("|---|---|---|---|---|")
    for k in settings.VALID_NUMBERS:
        o = counts.get(k, 0)
        e = expected[k]
        resid = standardized_residual(o, e)
        flag = "  [!]" if abs(resid) > 2 else ""
        L.append(f"| {k} | {o} | {e:.1f} | {design[k] * 100:.1f}% | {resid:+.2f}{flag} |")
    L.append("\n_Residu baku |z|>2 menandai angka yang muncul jauh lebih / kurang sering dari porsi desainnya._\n")
    L.append("\n## Arti untuk sistem\n")
    if no_update:
        L.append("- Mode `--no-update`: prior sistem TIDAK diubah (report saja).")
    else:
        L.append(f"- {n} observasi ini ditulis ke `models/wheel_prior.json` dan akan **dilipat ke posterior BayesianOptimalEngine** sebagai spin nyata tambahan saat app di-restart.")
    L.append("- Engine hanya bertaruh kalau batas bawah kredibel sebuah angka melewati titik impas (lower-bound +EV). Roda adil = tidak ada taruhan. Itu desain yang benar.\n")
    return "\n".join(L), ts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pelajari bias roda dari observasi kamera")
    parser.add_argument("--no-update", action="store_true",
                        help="Hanya tulis report; jangan update models/wheel_prior.json.")
    parser.add_argument("--stopped-only", action="store_true",
                        help="Pakai hanya observasi saat roda benar-benar berhenti.")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Abaikan bacaan di bawah confidence centering ini.")
    parser.add_argument("--min-spins", type=int, default=30,
                        help="Beri peringatan (tetap jalan) di bawah jumlah spin ini.")
    args = parser.parse_args(argv)

    obs = load_observations()
    usable = [
        o for o in obs
        if o["number"] in settings.VALID_NUMBERS
        and (not args.stopped_only or o["stopped"])
        and (o["confidence"] is None or o["confidence"] >= args.min_confidence)
    ]
    n = len(usable)
    if n == 0:
        print(f"[learn] Belum ada observasi usable di {OBSERVATIONS_PATH}.")
        print("[learn] Jalankan dulu: python scripts/wheel_cam.py --rounds 50")
        return 1

    counts = Counter(o["number"] for o in usable)
    design = design_distribution(settings.SPINWHEEL_SEQUENCE, settings.VALID_NUMBERS)
    expected = {k: n * design[k] for k in settings.VALID_NUMBERS}
    chi2, dof, p = chi_square_gof(counts, expected)
    biased = (p < 0.05) and (n >= args.min_spins)

    report, ts = build_report(
        usable, counts, design, expected, chi2, dof, p, biased,
        args.stopped_only, args.no_update, args.min_spins,
    )
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    if not args.no_update:
        payload = {
            "counts": {str(k): int(counts.get(k, 0)) for k in settings.VALID_NUMBERS},
            "n_obs": n,
            "chi_square": round(chi2, 4),
            "dof": dof,
            "p_value": round(p, 6),
            "biased": bool(biased),
            "stopped_only": bool(args.stopped_only),
            "min_confidence": args.min_confidence,
            "updated_at": ts,
            "source": "vision camera observations (scripts/learn_from_vision.py)",
        }
        os.makedirs(os.path.dirname(WHEEL_PRIOR_PATH), exist_ok=True)
        with open(WHEEL_PRIOR_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=== learn_from_vision ===")
    print(f"usable spins   : {n}")
    print(f"chi-square     : {chi2:.2f} (dof {dof}), p = {p:.4f}")
    print(f"verdict        : {'BIAS terdeteksi (p<0.05)' if biased else 'tidak ada bias signifikan'}")
    print(f"report         : {REPORT_PATH}")
    if not args.no_update:
        print(f"prior updated  : {WHEEL_PRIOR_PATH}  (restart app untuk dipakai)")
    else:
        print("prior          : TIDAK diubah (--no-update)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
