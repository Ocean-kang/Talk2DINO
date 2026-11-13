#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CLIP_CKPT:?Set CLIP_CKPT to an existing local OpenAI CLIP ViT-B/16 checkpoint .pt}"
python -m experiments.role_aware_concept_composition.build_ra_c2c \
  --base-train outputs/coco2014_b14/train.pth \
  --concept-source f5=outputs/f5_ponytail/run1/audit.jsonl \
  --concept-source cocu=outputs/local_cocu_talk2dino_new_train/manifest.json \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --clip-checkpoint "$CLIP_CKPT" --device cuda --batch-size 512 \
  --output-dir outputs/role_aware_concept_composition/f5_cocu_hybrid \
  --missing-policy keep_raw
