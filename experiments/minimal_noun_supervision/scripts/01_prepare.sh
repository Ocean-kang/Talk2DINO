#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -u "$MNS_DIR/prepare.py" \
  --train-features "${TRAIN_PTH:?Set TRAIN_PTH to existing original train.pth}" \
  --val-features "${VAL_PTH:?Set VAL_PTH to existing original val.pth}" \
  --gallery "${MNS_GALLERY:?Set MNS_GALLERY to existing MSCOCO14 gallery}" \
  --clip-checkpoint "${MNS_CLIP_CHECKPOINT:?Set MNS_CLIP_CHECKPOINT to local CLIP weight FILE}" \
  --dino-checkpoint "${MNS_DINO_CHECKPOINT:?Set MNS_DINO_CHECKPOINT to existing DINO weight FILE}" \
  --feature-metadata "${MNS_FEATURE_METADATA:?Set MNS_FEATURE_METADATA to verified source metadata}" \
  --output "$MNS_DATA" "$@"
