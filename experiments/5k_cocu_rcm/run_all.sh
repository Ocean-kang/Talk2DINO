#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
bash experiments/5k_cocu_rcm/run_preprocess.sh
bash experiments/5k_cocu_rcm/run_smoke_test.sh
bash experiments/5k_cocu_rcm/run_train.sh
bash experiments/5k_cocu_rcm/run_seg_eval.sh
python experiments/5k_cocu_rcm/collect_results.py \
  --root outputs/5k_cocu_rcm --weights-dir weights \
  --seg-results-dir segmentation_results
