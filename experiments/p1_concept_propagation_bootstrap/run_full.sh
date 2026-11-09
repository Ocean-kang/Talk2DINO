#!/usr/bin/env bash
set -euo pipefail

# Run from Talk2DINO repository root.
GPU="${1:-0}"
COCO_ROOT="${COCO_ROOT:-data/coco2014}"
BASE="outputs/p1_concept_propagation_bootstrap"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-123}"

export CUDA_VISIBLE_DEVICES="$GPU"

python -m experiments.p1_concept_propagation_bootstrap.prepare_features \
  --split train \
  --features outputs/coco2014_b14/train.pth \
  --image-root "$COCO_ROOT" \
  --teacher-weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --output "$BASE/cache/train" \
  --device "$DEVICE" \
  --limit 0

python -m experiments.p1_concept_propagation_bootstrap.prepare_features \
  --split val \
  --features outputs/coco2014_b14/val.pth \
  --image-root "$COCO_ROOT" \
  --teacher-weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --output "$BASE/cache/val" \
  --device "$DEVICE" \
  --limit 0

python -m experiments.p1_concept_propagation_bootstrap.train_student \
  --train-features outputs/coco2014_b14/train.pth \
  --val-features outputs/coco2014_b14/val.pth \
  --train-cache "$BASE/cache/train" \
  --val-cache "$BASE/cache/val" \
  --config configs/vitb_mlp_infonce.yaml \
  --output "$BASE/student_seed${SEED}" \
  --device "$DEVICE" \
  --seed "$SEED"
