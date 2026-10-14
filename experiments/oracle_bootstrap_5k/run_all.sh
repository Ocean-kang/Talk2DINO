#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/oracle_bootstrap_5k/run_all.sh [GPU] [WORK_DIR]
# Four sequential runs: Raw/Rich x alpha=0/1. Each uses original Talk2DINO settings.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
cd "$ROOT"
GPU="${1:-1}"
WORK="${2:-experiments/raw_rich_5k/work}"

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  python "$HERE/check_inputs.py" --work "$WORK"
fi

for ARM in raw rich; do
  for ALPHA in 0.0 1.0; do
    bash "$HERE/run_train.sh" "$GPU" "$ARM" "$ALPHA" "$WORK"
  done
done

echo '[DONE] Completed four 5K Oracle Bootstrap training arms.'
