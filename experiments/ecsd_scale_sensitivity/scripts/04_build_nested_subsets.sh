#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
WORK_DIR="${WORK_DIR:-$OUT_ROOT/full_features}"
SUBSET_DIR="${SUBSET_DIR:-$OUT_ROOT/work}"
ARGS=()
if [[ "${OVERWRITE_SUBSETS:-0}" == "1" ]]; then ARGS+=(--overwrite); fi
python experiments/ecsd_scale_sensitivity/data/build_nested_subsets.py \
  --raw-full "$WORK_DIR/full_raw_train.pth" \
  --ecsd-full "$WORK_DIR/full_ecsd_train.pth" \
  --val-full "$WORK_DIR/val_shared.pth" \
  --out-dir "$SUBSET_DIR" \
  --expected-full "${EXPECTED_TRAIN_IMAGES:-82783}" \
  --sizes "${SUBSET_SIZES:-5000,10000,20000,40000,82783}" \
  --seed "${SUBSET_SEED:-2026}" "${ARGS[@]}"
python experiments/ecsd_scale_sensitivity/data/validate_scale_data.py \
  --work-dir "$SUBSET_DIR" --expected-val "${EXPECTED_VAL_IMAGES:-40504}"
