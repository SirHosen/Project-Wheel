#!/usr/bin/env bash
# Run every internal test from the project root with the package importable.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.

# Cari interpreter Python yang tersedia. Di Windows (Git Bash) seringkali
# 'python' tidak terlihat di PATH bash walau venv aktif di PowerShell, jadi
# coba beberapa nama umum: python, python3, py.
PYBIN=""
for cand in python python3 py; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYBIN="$cand"; break
    fi
done
if [ -z "$PYBIN" ]; then
    echo "ERROR: Python tidak ditemukan di PATH (coba: python / python3 / py)."
    echo "Di Windows native, jalankan lewat PowerShell: .\\run_tests.ps1"
    exit 127
fi

pass=0; fail=0; failed=""
for f in tests/test_*.py; do
    if "$PYBIN" "$f" >/tmp/_pw_test.log 2>&1; then
        pass=$((pass+1)); echo "PASS  $f"
    else
        fail=$((fail+1)); failed="$failed $f"; echo "FAIL  $f"; tail -8 /tmp/_pw_test.log
    fi
done
echo "============================="
echo "PASS=$pass FAIL=$fail"
[ -n "$failed" ] && echo "FAILED:$failed"
exit $fail
