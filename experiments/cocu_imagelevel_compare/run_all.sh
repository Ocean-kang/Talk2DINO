#!/usr/bin/env bash
set -euo pipefail

# Run from the Talk2DINO repository root:
#   cd /home/master/code/oymk/Talking2dino/Talk2DINO
#   bash experiments/cocu_imagelevel_compare/run_all.sh
#
# IMPORTANT:
# DINO_PTH must be the SAME CLS-feature PTH used to train
# vitb_mlp_infonce_cls_imagelevel*.pth.

GPU="${GPU:-2}"

CAPTIONS="data/coco2014/annotations/captions_train2014.json"
IMAGES="data/coco2014/train2014"
GALLERY="experiments/cocu_imagelevel_compare/MSCOCO14_train_gallery_exactmatch.json"

# Change this only if your image-level training used another PTH.
DINO_PTH="${DINO_PTH:-outputs/coco2014_cls/train_cls.pth}"

CONFIG="configs/vitb_mlp_infonce.yaml"
CACHE="outputs/cocu_imagelevel_compare/cache"
OUT="outputs/cocu_imagelevel_compare"

mkdir -p "${CACHE}" "${OUT}"

echo "[1/3] CLIP ViT-B/16"
CUDA_VISIBLE_DEVICES="${GPU}" \
python experiments/cocu_imagelevel_compare/build_curated_concepts.py \
  --backend clip \
  --captions-json "${CAPTIONS}" \
  --images-dir "${IMAGES}" \
  --gallery "${GALLERY}" \
  --output "${OUT}/clip_vitb16.json" \
  --cache-dir "${CACHE}" \
  --clip-model "ViT-B/16" \
  --top-k 8 \
  --num-curated 3 \
  --device cuda

echo "[2/3] Talk2DINO-image: vitb_mlp_infonce_cls_imagelevel.pth"
CUDA_VISIBLE_DEVICES="${GPU}" \
python experiments/cocu_imagelevel_compare/build_curated_concepts.py \
  --backend talk2dino_image \
  --captions-json "${CAPTIONS}" \
  --gallery "${GALLERY}" \
  --output "${OUT}/talk2dino_image_cls_imagelevel.json" \
  --cache-dir "${CACHE}" \
  --clip-model "ViT-B/16" \
  --projector-config "${CONFIG}" \
  --projector-weight "weights/vitb_mlp_infonce_cls_imagelevel.pth" \
  --dino-features-pth "${DINO_PTH}" \
  --top-k 8 \
  --num-curated 3 \
  --device cuda

echo "[3/3] Talk2DINO-image: vitb_mlp_infonce_cls_imagelevel_epoch1000.pth"
CUDA_VISIBLE_DEVICES="${GPU}" \
python experiments/cocu_imagelevel_compare/build_curated_concepts.py \
  --backend talk2dino_image \
  --captions-json "${CAPTIONS}" \
  --gallery "${GALLERY}" \
  --output "${OUT}/talk2dino_image_cls_imagelevel_epoch1000.json" \
  --cache-dir "${CACHE}" \
  --clip-model "ViT-B/16" \
  --projector-config "${CONFIG}" \
  --projector-weight "weights/vitb_mlp_infonce_cls_imagelevel_epoch1000.pth" \
  --dino-features-pth "${DINO_PTH}" \
  --top-k 8 \
  --num-curated 3 \
  --device cuda

echo "Done."
echo "Outputs:"
echo "  ${OUT}/clip_vitb16.json"
echo "  ${OUT}/talk2dino_image_cls_imagelevel.json"
echo "  ${OUT}/talk2dino_image_cls_imagelevel_epoch1000.json"
