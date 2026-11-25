#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
bash experiments/full_mscoco14_cocu_rcm/scripts/run_preprocess.sh
bash experiments/full_mscoco14_cocu_rcm/scripts/run_smoke_test.sh
if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
  bash experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh
  bash experiments/full_mscoco14_cocu_rcm/scripts/run_seg_eval.sh
else
  echo "[STOP] Preprocessing and smoke tests are complete. Set RUN_TRAIN=1 to start model training/evaluation."
fi
