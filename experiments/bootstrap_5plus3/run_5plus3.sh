#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/bootstrap_5plus3/run_5plus3.sh build 0
#   bash experiments/bootstrap_5plus3/run_5plus3.sh train-clip 0
#   bash experiments/bootstrap_5plus3/run_5plus3.sh train-image 1
#   bash experiments/bootstrap_5plus3/run_5plus3.sh train-local 2
#
# Run from the Talk2DINO repository root.

ACTION="${1:-}"
GPU="${2:-0}"

BASE_TRAIN="outputs/coco2014_b14/train.pth"
VAL="outputs/coco2014_b14/val.pth"
CFG="configs/vitb_mlp_infonce.yaml"
OUTDIR="outputs/bootstrap_5plus3"

case "${ACTION}" in
  build)
    CUDA_VISIBLE_DEVICES="${GPU}" \
    python experiments/bootstrap_5plus3/build_5plus3.py \
      --method all \
      --base-train "${BASE_TRAIN}" \
      --output-dir "${OUTDIR}" \
      --num-curated 3 \
      --clip-model "ViT-B/16" \
      --batch-size 512
    ;;

  train-clip)
    CUDA_VISIBLE_DEVICES="${GPU}" \
    python train.py \
      --model_config "${CFG}" \
      --train_dataset "${OUTDIR}/train_clip_vitb16_5plus3.pth" \
      --val_dataset "${VAL}" \
      --feature_name disentangled_self_attn \
      --text_features ann_feats \
      --name_pedix clip_vitb16_5plus3
    ;;

  train-image)
    CUDA_VISIBLE_DEVICES="${GPU}" \
    python train.py \
      --model_config "${CFG}" \
      --train_dataset "${OUTDIR}/train_talk2dino_image_cls_imagelevel_5plus3.pth" \
      --val_dataset "${VAL}" \
      --feature_name disentangled_self_attn \
      --text_features ann_feats \
      --name_pedix talk2dino_image_cls_imagelevel_5plus3
    ;;

  train-local)
    CUDA_VISIBLE_DEVICES="${GPU}" \
    python train.py \
      --model_config "${CFG}" \
      --train_dataset "${OUTDIR}/train_local_cocu_compare_5plus3.pth" \
      --val_dataset "${VAL}" \
      --feature_name disentangled_self_attn \
      --text_features ann_feats \
      --name_pedix local_cocu_compare_5plus3
    ;;

  *)
    echo "Usage:"
    echo "  $0 build [gpu]"
    echo "  $0 train-clip [gpu]"
    echo "  $0 train-image [gpu]"
    echo "  $0 train-local [gpu]"
    exit 1
    ;;
esac
