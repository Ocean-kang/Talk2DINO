#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

python -m compileall -q experiments/5k_ecsd_next
for script in experiments/5k_ecsd_next/scripts/*.sh experiments/5k_ecsd_next/multiseed/*.sh experiments/5k_ecsd_next/patches/*.sh; do
  bash -n "$script"
done
python -m unittest discover -s experiments/5k_ecsd_next/tests -p 'test_*.py' -v

echo "[PASS] Python compile, unit tests, and shell syntax."
echo "[INFO] Full JSON/data/feature validation runs on the experiment server:"
echo "  python experiments/5k_ecsd_next/tests/validate_json_schema.py --input data/coco2014_subset_50k/5k_ECSD_v2.json --require-ecsd-v2"
echo "  python experiments/5k_ecsd_next/multiseed/prepare_multiseed.py audit --work outputs/5k_ecsd_next/multiseed/work"
