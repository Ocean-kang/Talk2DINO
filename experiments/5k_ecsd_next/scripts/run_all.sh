#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ECSD_CONFIG="${ECSD_CONFIG:-experiments/5k_ecsd_next/configs/default.json}"
CFG_DATA_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["data_dir"])' "$ECSD_CONFIG")"
CFG_OUT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$ECSD_CONFIG")"
DATA_DIR="${DATA_DIR:-$CFG_DATA_DIR}"
OUT_DIR="${OUT_DIR:-$CFG_OUT}"

if [[ ! -f "$OUT_DIR/candidates/ecsd_v2_candidates.jsonl" || ! -f "$DATA_DIR/5k_ECSD_v2.json" ]]; then
  bash experiments/5k_ecsd_next/scripts/run_experiment_a.sh
else
  echo "[REUSE] Stage A candidate pool and ECSD v2 already exist."
fi

if [[ ! -f "$OUT_DIR/ablation_metrics/summary.csv" ]]; then
  bash experiments/5k_ecsd_next/scripts/run_experiment_b.sh
else
  echo "[REUSE] Stage B metrics summary already exists."
fi

if [[ "${CONFIRM_FULL_RUN:-NO}" != "YES" ]]; then
  echo "[STOP] Stage A/B are ready. Stage C launches 12 training runs and up to 48 OVSS evaluations."
  echo "To continue without regenerating A/B, run:"
  echo "  CONFIRM_FULL_RUN=YES bash experiments/5k_ecsd_next/scripts/run_all.sh"
  exit 0
fi
bash experiments/5k_ecsd_next/scripts/run_experiment_c.sh
