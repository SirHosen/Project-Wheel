#!/usr/bin/env bash
# Linux/macOS convenience wrapper. On Windows use: python run_tests.py
cd "$(dirname "$0")" && exec python3 run_tests.py "$@"
