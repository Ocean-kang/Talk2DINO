#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WEIGHT_STEM="${1:?Usage: bash run_retrieval_eval.sh WEIGHT_STEM [TEST_DATASET]}"
TEST_DATASET="${2:-../coco2014_b14/test.pth}"
IMG_FEATURES="${IMG_FEATURES:-avg_self_attn_out}"
TEXT_FEATURES="${TEXT_FEATURES:-ann_feats}"

if [[ ! -f "weights/${WEIGHT_STEM}.pth" ]]; then
  echo "Checkpoint not found: weights/${WEIGHT_STEM}.pth" >&2
  exit 1
fi

python src/metrics.py \
  --config configs/vitb_mlp_infonce.yaml \
  --weights "weights/${WEIGHT_STEM}.pth" \
  --test_data "${TEST_DATASET}" \
  --img_features "${IMG_FEATURES}" \
  --text_features "${TEXT_FEATURES}"
