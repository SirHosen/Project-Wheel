# -*- coding: utf-8 -*-
"""Statistik untuk vision learning loop: apakah roda FISIK bias?

Pure-math (tanpa numpy/scipy) supaya bisa di-unit-test di mana saja. Dipakai
scripts/learn_from_vision.py buat mengubah observasi kamera jadi uji kecocokan
chi-square yang jujur terhadap layout DESAIN roda.

Ini mendeteksi bias jangka panjang; ini BUKAN prediksi spin berikutnya.
"""
import math
from collections import Counter


def gammq(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).

    Memberi fungsi survival chi-square: p = gammq(dof/2, chi2/2).
    Pure-Python (gaya Numerical Recipes: deret + continued fraction).
    """
    if a <= 0 or x < 0:
        return float("nan")
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        term = total
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - gln)


def design_distribution(sequence, valid_numbers):
    """Return {number: porsi desain} dari layout segmen fisik."""
    seg = Counter(sequence)
    total = float(len(sequence)) or 1.0
    return {n: seg.get(n, 0) / total for n in valid_numbers}


def chi_square_gof(observed_counts, expected_counts):
    """Pearson chi-square goodness-of-fit. Return (chi2, dof, p_value)."""
    chi2 = 0.0
    cats = 0
    for k, exp in expected_counts.items():
        if exp <= 0:
            continue
        obs = observed_counts.get(k, 0)
        chi2 += (obs - exp) ** 2 / exp
        cats += 1
    dof = max(1, cats - 1)
    return chi2, dof, gammq(dof / 2.0, chi2 / 2.0)


def standardized_residual(observed, expected):
    """(obs - exp) / sqrt(exp): |z|>2 ~ angka menyimpang nyata dari desain."""
    return (observed - expected) / math.sqrt(expected) if expected > 0 else 0.0
