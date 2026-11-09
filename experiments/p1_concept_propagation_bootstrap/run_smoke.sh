#!/usr/bin/env bash
set -euo pipefail

# Run from Talk2DINO repository root.
GPU="${1:-0}"
COCO_ROOT="${COCO_ROOT:-data/coco2014}"
BASE="outputs/p1_concept_propagation_bootstrap/smoke_cache"
DEVICE="${DEVICE:-cuda}"

export CUDA_VISIBLE_DEVICES="$GPU"

python -m experiments.p1_concept_propagation_bootstrap.prepare_features \
  --split train \
  --features outputs/coco2014_b14/train.pth \
  --image-root "$COCO_ROOT" \
  --teacher-weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --output "$BASE/train" \
  --device "$DEVICE" \
  --limit 2

python -m experiments.p1_concept_propagation_bootstrap.prepare_features \
  --split val \
  --features outputs/coco2014_b14/val.pth \
  --image-root "$COCO_ROOT" \
  --teacher-weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --output "$BASE/val" \
  --device "$DEVICE" \
  --limit 2

python -m experiments.p1_concept_propagation_bootstrap.train_student \
  --train-features outputs/coco2014_b14/train.pth \
  --val-features outputs/coco2014_b14/val.pth \
  --train-cache "$BASE/train" \
  --val-cache "$BASE/val" \
  --config configs/vitb_mlp_infonce.yaml \
  --output outputs/p1_concept_propagation_bootstrap/smoke_student \
  --device "$DEVICE" \
  --seed 123 \
  --epochs 1 \
  --batch-size 4 \
  --max-train-samples 8 \
  --max-val-samples 8 \
  --allow-partial-cache
