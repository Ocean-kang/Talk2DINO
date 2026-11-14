#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bash experiments/5k_ecsd/run_preprocess.sh
bash experiments/5k_ecsd/run_train.sh
bash experiments/5k_ecsd/run_seg_eval.sh

echo "[DONE] Full 5K ECSD experiment completed."
