#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/coco2014_subset_50k}"
SOURCE_CAPTIONS="${SOURCE_CAPTIONS:-${DATA_DIR}/5k_rich_diversity.json}"
ECSD_CAPTIONS="${ECSD_CAPTIONS:-${DATA_DIR}/5k_ECSD.json}"
OUT_DIR="${OUT_DIR:-outputs/5k_ecsd}"
WORK_DIR="${WORK_DIR:-${OUT_DIR}/work}"
CLIP_CACHE="${CLIP_CACHE:-$HOME/.cache/clip}"
ECSD_BATCH_SIZE="${ECSD_BATCH_SIZE:-256}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-8}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-256}"
DINO_SOURCE="${DINO_SOURCE:-outputs/5k_rich_diversity/work/dino.pth}"

if [[ ! -f "$SOURCE_CAPTIONS" ]]; then
  echo "[ERROR] Source captions not found: $SOURCE_CAPTIONS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/logs" "$WORK_DIR"

echo "============================================================"
echo "Talk2DINO Subset5K ECSD-naive preprocessing"
echo "Repository       : $ROOT"
echo "Data dir         : $DATA_DIR"
echo "Source captions  : $SOURCE_CAPTIONS"
echo "ECSD captions    : $ECSD_CAPTIONS"
echo "Work dir         : $WORK_DIR"
echo "============================================================"

if [[ ! -f "$ECSD_CAPTIONS" || "${REGENERATE_ECSD:-0}" == "1" ]]; then
  echo "[ECSD] Generating evidence-constrained captions..."
  python experiments/5k_ecsd/generate_ecsd.py \
    --input "$SOURCE_CAPTIONS" \
    --output "$ECSD_CAPTIONS" \
    --trace "$OUT_DIR/ecsd_selection_trace.jsonl" \
    --summary "$OUT_DIR/generation_summary.json" \
    --clip-cache "$CLIP_CACHE" \
    --batch-size "$ECSD_BATCH_SIZE" \
    --overwrite
else
  echo "[ECSD] Reusing existing $ECSD_CAPTIONS"
fi

echo "[CISD] Computing per-image CISD for all five caption sets..."
python experiments/5k_ecsd/evaluate_cisd.py \
  --captions "$ECSD_CAPTIONS" \
  --output-dir "$OUT_DIR/cisd" \
  --clip-cache "$CLIP_CACHE" \
  --batch-size "$ECSD_BATCH_SIZE" \
  --expected-images 5000 \
  2>&1 | tee "$OUT_DIR/logs/cisd.log"

echo "[DATA] Validating captions and staging image symlinks..."
python experiments/5k_ecsd/prepare.py init \
  --captions "$ECSD_CAPTIONS" \
  --images "$DATA_DIR" \
  --work "$WORK_DIR" \
  --expected-images 5000

if [[ "${FORCE_DINO:-0}" == "1" ]]; then
  rm -f "$WORK_DIR/dino.pth"
fi

if [[ ! -f "$WORK_DIR/dino.pth" ]]; then
  if [[ -f "$DINO_SOURCE" && "${FORCE_DINO:-0}" != "1" ]]; then
    echo "[DINO] Reusing existing DINO features: $DINO_SOURCE"
    ln -s "$(realpath "$DINO_SOURCE")" "$WORK_DIR/dino.pth"
  else
    echo "[DINO] Extracting shared DINOv2 ViT-B/14-reg features..."
    python dino_extraction_v2.py \
      --ann_path "$WORK_DIR/base.pth" \
      --data_dir "$WORK_DIR/images" \
      --out_path "$WORK_DIR/dino.pth" \
      --model dinov2_vitb14_reg \
      --resize_dim 448 \
      --crop_dim 448 \
      --batch_size "$DINO_BATCH_SIZE" \
      --extract_avg_self_attn \
      --extract_disentangled_self_attn \
      2>&1 | tee "$OUT_DIR/logs/dino_extract.log"
  fi
fi

echo "[DATA] Assembling five caption arms with one shared 4500/500 split..."
python experiments/5k_ecsd/prepare.py assemble \
  --captions "$ECSD_CAPTIONS" \
  --work "$WORK_DIR" \
  --seed 123 \
  --val-images 500 \
  --expected-images 5000

echo "[CLIP] Extracting CLIP ViT-B/16 ann_feats for all ten files..."
for arm in raw rich rich_weaken rich_diversity ecsd; do
  for split in train val; do
    FILE="$WORK_DIR/${arm}_${split}.pth"
    python text_features_extraction.py \
      --ann_path "$FILE" \
      --out_path "$FILE" \
      --model ViT-B/16 \
      --batch_size "$CLIP_BATCH_SIZE" \
      2>&1 | tee "$OUT_DIR/logs/clip_${arm}_${split}.log"
  done
done

python experiments/5k_ecsd/prepare.py audit \
  --work "$WORK_DIR" \
  --expected-images 5000 \
  --val-images 500

echo "[DONE] ECSD preprocessing, CISD evaluation, and feature audit completed."
