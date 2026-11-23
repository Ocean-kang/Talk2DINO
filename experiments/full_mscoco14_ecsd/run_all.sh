#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_preprocess.sh"
bash "$SCRIPT_DIR/run_train.sh"
bash "$SCRIPT_DIR/run_seg_eval.sh"
echo "[DONE] Full MSCOCO14 ECSD experiment pipeline completed."
