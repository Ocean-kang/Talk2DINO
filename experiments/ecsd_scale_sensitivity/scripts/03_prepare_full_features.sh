#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COCO_ROOT="${COCO_ROOT:-coco}"
TRAIN_ANN="${TRAIN_ANN:-$COCO_ROOT/annotations/captions_train2014.json}"
VAL_ANN="${VAL_ANN:-$COCO_ROOT/annotations/captions_val2014.json}"
RAW_DIR="${RAW_DIR:-data/coco2014_scale}"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
WORK_DIR="${WORK_DIR:-$OUT_ROOT/full_features}"
DINO_TRAIN_SOURCE="${DINO_TRAIN_SOURCE:-outputs/coco2014_b14/train.pth}"
DINO_VAL_SOURCE="${DINO_VAL_SOURCE:-outputs/coco2014_b14/val.pth}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-256}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-8}"
mkdir -p "$WORK_DIR" "$OUT_ROOT/logs"
for f in "$RAW_DIR/raw_train.json" "$RAW_DIR/raw_val.json" "$RAW_DIR/ecsd_train_b4_v3.json"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing $f. Complete Stages 1 and 2 first." >&2; exit 1; }
done

python experiments/full_mscoco14_ecsd/prepare.py build-base \
  --raw-captions "$RAW_DIR/raw_train.json" --annotations "$TRAIN_ANN" \
  --split train --image-root "$COCO_ROOT" --out "$WORK_DIR/base_train.pth"
python experiments/full_mscoco14_ecsd/prepare.py build-base \
  --raw-captions "$RAW_DIR/raw_val.json" --annotations "$VAL_ANN" \
  --split val --image-root "$COCO_ROOT" --out "$WORK_DIR/base_val.pth"

if [[ "${FORCE_DINO:-0}" == "1" ]]; then
  rm -f "$WORK_DIR/dino_train.pth" "$WORK_DIR/dino_val.pth"
fi
if [[ ! -f "$WORK_DIR/dino_train.pth" ]]; then
  if [[ "${REUSE_DINO:-1}" == "1" && -f "$DINO_TRAIN_SOURCE" ]]; then
    python experiments/full_mscoco14_ecsd/prepare.py reuse-dino \
      --base "$WORK_DIR/base_train.pth" --source "$DINO_TRAIN_SOURCE" \
      --out "$WORK_DIR/dino_train.pth" --required-feature disentangled_self_attn
  else
    python dino_extraction_v2.py --ann_path "$WORK_DIR/base_train.pth" \
      --data_dir "$COCO_ROOT" --out_path "$WORK_DIR/dino_train.pth" \
      --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 \
      --batch_size "$DINO_BATCH_SIZE" --extract_avg_self_attn --extract_disentangled_self_attn \
      2>&1 | tee "$OUT_ROOT/logs/dino_train.log"
  fi
fi
if [[ ! -f "$WORK_DIR/dino_val.pth" ]]; then
  if [[ "${REUSE_DINO:-1}" == "1" && -f "$DINO_VAL_SOURCE" ]]; then
    python experiments/full_mscoco14_ecsd/prepare.py reuse-dino \
      --base "$WORK_DIR/base_val.pth" --source "$DINO_VAL_SOURCE" \
      --out "$WORK_DIR/dino_val.pth" --required-feature avg_self_attn_out
  else
    python dino_extraction_v2.py --ann_path "$WORK_DIR/base_val.pth" \
      --data_dir "$COCO_ROOT" --out_path "$WORK_DIR/dino_val.pth" \
      --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 \
      --batch_size "$DINO_BATCH_SIZE" --extract_avg_self_attn --extract_disentangled_self_attn \
      2>&1 | tee "$OUT_ROOT/logs/dino_val.log"
  fi
fi

python experiments/ecsd_scale_sensitivity/data/build_feature_sets.py assemble \
  --train-raw "$RAW_DIR/raw_train.json" \
  --train-ecsd "$RAW_DIR/ecsd_train_b4_v3.json" \
  --val-raw "$RAW_DIR/raw_val.json" \
  --dino-train "$WORK_DIR/dino_train.pth" \
  --dino-val "$WORK_DIR/dino_val.pth" \
  --out-dir "$WORK_DIR"

for f in "$WORK_DIR/full_raw_train.pth" "$WORK_DIR/full_ecsd_train.pth" "$WORK_DIR/val_shared.pth"; do
  if [[ "${FORCE_CLIP:-0}" == "1" ]] || ! python experiments/ecsd_scale_sensitivity/data/build_feature_sets.py check-clip --file "$f" >/dev/null 2>&1; then
    echo "[CLIP] Extracting ann_feats for $f"
    python text_features_extraction.py --ann_path "$f" --out_path "$f" \
      --model ViT-B/16 --batch_size "$CLIP_BATCH_SIZE" \
      2>&1 | tee "$OUT_ROOT/logs/clip_$(basename "$f" .pth).log"
  else
    echo "[CLIP] Reusing ann_feats in $f"
  fi
  python experiments/ecsd_scale_sensitivity/data/build_feature_sets.py check-clip --file "$f"
done
echo "[DONE] Full Raw/ECSD/Val feature files are ready under $WORK_DIR"
