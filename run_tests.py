# -*- coding: utf-8 -*-
"""Cross-platform test runner (Windows / macOS / Linux). No pytest, no bash.

    python run_tests.py

A test counts as PASS when it exits 0 and prints "ALL CHECKS PASSED".
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    if not files:
        print("No tests found under tests/test_*.py")
        return 1
    passed = failed = 0
    for f in files:
        rel = os.path.relpath(f, ROOT)
        proc = subprocess.run([sys.executable, f], cwd=ROOT, env=env,
                              capture_output=True, text=True)
        ok = proc.returncode == 0 and "ALL CHECKS PASSED" in proc.stdout
        if ok:
            print(f"PASS  {rel}")
            passed += 1
        else:
            print(f"FAIL  {rel}")
            for line in (proc.stdout + proc.stderr).splitlines():
                print("      " + line)
            failed += 1
    print("=============================")
    print(f"PASS={passed} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
