#!/usr/bin/env bash
set -euo pipefail

# Run from any directory; paths passed to train.py are interpreted from repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-so}"  # so | baseline | both
MODEL_CONFIG="${MODEL_CONFIG:-configs/vitb_mlp_infonce.yaml}"
TRAIN_DATASET="${TRAIN_DATASET:-outputs/coco2014_b14/train.pth}"
VAL_DATASET="${VAL_DATASET:-outputs/coco2014_b14/val.pth}"
TEST_DATASET="${TEST_DATASET:-outputs/coco2014_b14/test.pth}"
FEATURE_NAME="${FEATURE_NAME:-disentangled_self_attn}"
TEXT_FEATURES="${TEXT_FEATURES:-ann_feats}"

SEEDS=(123 42 2024 2025 3407)

run_one() {
  local seed="$1"
  local enabled="$2"
  local tag="$3"

  echo "============================================================"
  echo "mode=${enabled}, seed=${seed}, tag=${tag}"
  echo "============================================================"

  T2DINO_SO_SEED="${seed}" T2DINO_SO_ENABLE="${enabled}" \
    python experiments/projector_semi_orthogonal/train_semi_orthogonal.py \
      --model_config "${MODEL_CONFIG}" \
      --train_dataset "${TRAIN_DATASET}" \
      --val_dataset "${VAL_DATASET}" \
      --test_dataset "${TEST_DATASET}" \
      --feature_name "${FEATURE_NAME}" \
      --text_features "${TEXT_FEATURES}" \
      --name_pedix "${tag}"
}

case "${MODE}" in
  so)
    for seed in "${SEEDS[@]}"; do
      run_one "${seed}" "1" "so_seed${seed}"
    done
    ;;
  baseline)
    for seed in "${SEEDS[@]}"; do
      run_one "${seed}" "0" "baseline_seed${seed}"
    done
    ;;
  both)
    for seed in "${SEEDS[@]}"; do
      run_one "${seed}" "0" "baseline_seed${seed}"
      run_one "${seed}" "1" "so_seed${seed}"
    done
    ;;
  *)
    echo "Usage: bash experiments/projector_semi_orthogonal/run_seeds.sh [so|baseline|both]" >&2
    exit 2
    ;;
esac
