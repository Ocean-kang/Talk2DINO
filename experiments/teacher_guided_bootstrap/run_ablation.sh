#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/teacher_guided_bootstrap/run_ablation.sh [GPU] [TEACHER] [TRAIN] [VAL]
# Example:
#   bash experiments/teacher_guided_bootstrap/run_ablation.sh 0 weights/vitb_mlp_infonce.pth

GPU="${1:-0}"
TEACHER="${2:-weights/vitb_mlp_infonce.pth}"
TRAIN="${3:-outputs/coco2014_b14/train.pth}"
VAL="${4:-outputs/coco2014_b14/val.pth}"

for ALPHA in 0.0 0.1 0.3 0.5 0.7 1.0; do
  echo "============================================================"
  echo "Teacher-guided bootstrap alpha=${ALPHA}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python experiments/teacher_guided_bootstrap/train_bootstrap.py \
    --teacher-weights "${TEACHER}" \
    --train-dataset "${TRAIN}" \
    --val-dataset "${VAL}" \
    --model-config configs/vitb_mlp_infonce.yaml \
    --alpha "${ALPHA}"
done
