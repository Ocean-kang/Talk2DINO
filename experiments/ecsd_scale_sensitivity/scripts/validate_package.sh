#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
python -m compileall -q experiments/ecsd_scale_sensitivity
for script in experiments/ecsd_scale_sensitivity/scripts/*.sh; do bash -n "$script"; done
python -m unittest discover -s experiments/ecsd_scale_sensitivity/tests -p 'test_*.py' -v
echo "[PASS] package Python compilation, shell syntax, and unit tests."
