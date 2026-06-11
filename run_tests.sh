#!/usr/bin/env bash
# Run every internal test from the project root with the package importable.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.
pass=0; fail=0; failed=""
for f in tests/_test_*.py; do
    if python "$f" >/tmp/_pw_test.log 2>&1; then
        pass=$((pass+1)); echo "PASS  $f"
    else
        fail=$((fail+1)); failed="$failed $f"; echo "FAIL  $f"; tail -8 /tmp/_pw_test.log
    fi
done
echo "============================="
echo "PASS=$pass FAIL=$fail"
[ -n "$failed" ] && echo "FAILED:$failed"
exit $fail
