# -*- coding: utf-8 -*-
import os as _os, sys as _sys  # path bootstrap: project root importable from subfolder
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""scripts/migrate_stats.py - One-shot honest re-grading of legacy history (audit V4 #3).

Ronde lama direkam dengan definisi LAMA: tebakan benar dengan 0 token tetap
dihitung "menang", sehingga win rate di dashboard menggelembung. Script ini
menilai ulang SETIAP ronde tersimpan dengan aturan jujur lalu membangun ulang
counter agregat (wins/losses/total/profit) supaya GUI mencerminkan kenyataan,
bukan hanya ronde baru.

Aturan jujur:
  * is_win  -> True HANYA jika ada stake nyata (token_bet > 0) pada angka yang
               keluar. Ronde tanpa snapshot taruhan (log tebakan lama) BUKAN
               kemenangan taruhan.
  * top1_hit -> TIDAK di-backfill untuk ronde lama. predicted_number lama hanya
               tersimpan saat tebakan benar (definisi menang lama), jadi backfill
               = (predicted == actual) bikin "Akurasi tebakan teratas" palsu
               ~100% (bias seleksi). Ronde lama ditandai legacy & TIDAK dinilai;
               metrik top-1 mulai bersih dari ronde baru (audit V5 #1).

Aman dijalankan berulang (idempoten). Jalankan dari ROOT project:
    python scripts/migrate_stats.py
"""
import shutil

from data.tracker import Tracker


def _is_betting_win(rec):
    """A real betting win = a stake (token_bet > 0) landed on the winning number."""
    actual = rec.get("actual_number")
    for b in (rec.get("bets") or []):
        if b.get("number") == actual and (b.get("token_bet") or 0) > 0:
            return True
    return False


def main():
    t = Tracker()
    data = t.data
    history = data.get("history", [])

    before = {
        "wins": data.get("wins"),
        "losses": data.get("losses"),
        "total": data.get("total_predictions"),
        "top1_graded": sum(1 for r in history if "top1_hit" in r),
    }

    # Backup the JSON mirror before mutating (in addition to Tracker's own
    # .migrated backup) so the re-grade is fully reversible.
    try:
        shutil.copy2(t.history_file, t.history_file + ".premigrate.bak")
    except Exception:
        pass

    regraded = 0
    backfilled = 0
    for rec in history:
        new_win = _is_betting_win(rec)
        if bool(rec.get("is_win")) != new_win:
            regraded += 1
        rec["is_win"] = new_win

        # Audit V5 #1: do NOT backfill top1_hit from (predicted == actual).
        # Legacy rounds stored predicted_number ONLY when the guess was right
        # (old win definition), so backfilling yields a selection-biased ~100%
        # top-1 accuracy. Instead strip any previously-backfilled grade and mark
        # the round legacy/un-evaluated so the honest LIVE metric starts clean.
        if not rec.get("top1_graded_live"):
            if "top1_hit" in rec:
                rec.pop("top1_hit", None)
                backfilled += 1
            rec["top1_legacy_unevaluated"] = True

    wins = sum(1 for r in history if r.get("is_win"))
    losses = len(history) - wins
    data["wins"] = wins
    data["losses"] = losses
    data["total_predictions"] = len(history)
    data["profit"] = sum(r.get("profit_change", 0) for r in history)

    t.save_data()

    top1_graded = sum(1 for r in history if r.get("top1_graded_live"))
    top1_hits = sum(1 for r in history if r.get("top1_graded_live") and r.get("top1_hit"))
    wr = (wins / len(history) * 100) if history else 0.0
    t1 = (top1_hits / top1_graded * 100) if top1_graded else 0.0

    print("=== migrate_stats (audit V4 #3) ===")
    print(f"records              : {len(history)}")
    print(f"is_win regraded      : {regraded}")
    print(f"top1 legacy cleaned  : {backfilled}")
    print(f"BEFORE  wins/losses  : {before['wins']}/{before['losses']} (total {before['total']}, top1_graded {before['top1_graded']})")
    print(f"AFTER   wins/losses  : {wins}/{losses} (total {len(history)}, top1_graded {top1_graded})")
    print(f"WIN RATE (taruhan)   : {wr:.1f}%")
    print(f"Akurasi top-1        : {t1:.1f}%  (n={top1_graded})")
    print("Selesai. SQLite + history.json sudah disinkronkan.")
    print(f"Backup pra-migrasi   : {t.history_file}.premigrate.bak")


if __name__ == "__main__":
    main()
