#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
python -m compileall -q experiments/5k_ecsd_v3
for script in experiments/5k_ecsd_v3/scripts/*.sh; do bash -n "$script"; done
python -m unittest discover -s experiments/5k_ecsd_v3/tests -p 'test_*.py' -v
printf '\n[PASS] Python compile, v3 unit tests, and shell syntax.\n'
