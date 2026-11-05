#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/coco2014_subset_50k}"
CAPTIONS="${CAPTIONS:-${DATA_DIR}/5k_rich_weaken_captions.json}"
WORK_DIR="${WORK_DIR:-outputs/raw_rich_weaken_5k/work}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-8}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-256}"

if [[ ! -f "$CAPTIONS" ]]; then
  echo "[ERROR] Caption JSON not found: $CAPTIONS" >&2
  echo "Set CAPTIONS=/path/to/5k_rich_weaken_captions.json if the filename/path differs." >&2
  exit 1
fi

mkdir -p "$WORK_DIR"

echo "============================================================"
echo "Talk2DINO 5K Raw / Rich / Rich-Weaken preprocessing"
echo "Repository : $ROOT"
echo "Data root  : $DATA_DIR"
echo "Captions   : $CAPTIONS"
echo "Work dir   : $WORK_DIR"
echo "============================================================"

python experiments/raw_rich_weaken_5k/prepare.py init \
  --captions "$CAPTIONS" \
  --images "$DATA_DIR" \
  --work "$WORK_DIR"

if [[ ! -f "$WORK_DIR/dino.pth" || "${FORCE_DINO:-0}" == "1" ]]; then
  echo "[DINO] Extracting shared DINOv2 ViT-B/14 reg features once..."
  python dino_extraction_v2.py \
    --ann_path "$WORK_DIR/base.pth" \
    --data_dir "$WORK_DIR/images" \
    --out_path "$WORK_DIR/dino.pth" \
    --model dinov2_vitb14_reg \
    --resize_dim 448 \
    --crop_dim 448 \
    --batch_size "$DINO_BATCH_SIZE" \
    --extract_avg_self_attn \
    --extract_disentangled_self_attn
else
  echo "[DINO] Reusing existing $WORK_DIR/dino.pth"
fi

python experiments/raw_rich_weaken_5k/prepare.py assemble \
  --captions "$CAPTIONS" \
  --work "$WORK_DIR" \
  --seed 123 \
  --val-images 500

echo "[CLIP] Extracting ViT-B/16 text features for all six files..."
for arm in raw rich rich_weaken; do
  for split in train val; do
    FILE="$WORK_DIR/${arm}_${split}.pth"
    python text_features_extraction.py \
      --ann_path "$FILE" \
      --out_path "$FILE" \
      --model ViT-B/16 \
      --batch_size "$CLIP_BATCH_SIZE"
  done
done

python experiments/raw_rich_weaken_5k/prepare.py audit \
  --work "$WORK_DIR" \
  --expected-images 5000 \
  --val-images 500

echo "[DONE] Preprocessing and data audit completed."
