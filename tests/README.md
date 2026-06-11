# Tests

26 test internal yang mengecek matematika inti, predictor, data layer, dan
logika taruhan. Semuanya berjalan tanpa GUI; yang butuh TensorFlow akan skip
otomatis kalau TF tidak terpasang.

## Cara menjalankan

Dari **root project**:

```bash
# semua test sekaligus
./run_tests.sh

# atau manual
PYTHONPATH=. python tests/test_logging_snapshot.py
```

Setiap file test sudah menyisipkan root project ke `sys.path`, jadi bisa juga
dijalankan langsung (`python tests/test_xxx.py`) selama dijalankan dari root.

> Catatan: test memakai konvensi sendiri (`print("OK ...")`), bukan pytest.
> Kalau mau migrasi ke pytest, ganti pengecekan jadi `assert` dan rename fungsi
> menjadi `def test_*()`.
