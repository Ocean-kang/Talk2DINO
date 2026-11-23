#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COCO_ROOT="${COCO_ROOT:-coco}"
TRAIN_ANN="${TRAIN_ANN:-${COCO_ROOT}/annotations/captions_train2014.json}"
VAL_ANN="${VAL_ANN:-${COCO_ROOT}/annotations/captions_val2014.json}"
RAW_DIR="${RAW_DIR:-data/coco2014_full}"
OUT_DIR="${OUT_DIR:-outputs/full_mscoco14_ecsd}"
WORK_DIR="${WORK_DIR:-${OUT_DIR}/work}"
CLIP_CACHE="${CLIP_CACHE:-$HOME/.cache/clip}"
ECSD_CLIP_BATCH_SIZE="${ECSD_CLIP_BATCH_SIZE:-256}"
ECSD_IMAGE_BATCH_SIZE="${ECSD_IMAGE_BATCH_SIZE:-256}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-256}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-8}"
DINO_TRAIN_SOURCE="${DINO_TRAIN_SOURCE:-outputs/coco2014_b14/train.pth}"
DINO_VAL_SOURCE="${DINO_VAL_SOURCE:-outputs/coco2014_b14/val.pth}"
REUSE_DINO="${REUSE_DINO:-1}"

mkdir -p "$RAW_DIR" "$OUT_DIR/logs" "$WORK_DIR"
for f in "$TRAIN_ANN" "$VAL_ANN" "$COCO_ROOT/train2014" "$COCO_ROOT/val2014"; do
  [[ -e "$f" ]] || { echo "[ERROR] Missing COCO input: $f" >&2; exit 1; }
done

if [[ "${REBUILD_RAW:-0}" == "1" || ! -f "$RAW_DIR/raw_train.json" || ! -f "$RAW_DIR/raw_val.json" ]]; then
  python experiments/full_mscoco14_ecsd/build_raw_captions.py \
    --train-annotations "$TRAIN_ANN" --val-annotations "$VAL_ANN" --output-dir "$RAW_DIR" \
    2>&1 | tee "$OUT_DIR/logs/build_raw.log"
else
  echo "[RAW] Reusing $RAW_DIR/raw_train.json and raw_val.json"
fi

for split in train val; do
  input="$RAW_DIR/raw_${split}.json"
  output="$RAW_DIR/ecsd_${split}.json"
  trace="$OUT_DIR/ecsd_trace_${split}.jsonl"
  summary="$OUT_DIR/ecsd_generation_${split}.json"
  if [[ "${REGENERATE_ECSD:-0}" == "1" || ! -f "$output" ]]; then
    extra=()
    if [[ "${REGENERATE_ECSD:-0}" == "1" ]]; then extra+=(--overwrite); else extra+=(--resume); fi
    python experiments/full_mscoco14_ecsd/generate_ecsd.py \
      --input "$input" --output "$output" --trace "$trace" --summary "$summary" \
      --clip-cache "$CLIP_CACHE" --batch-size "$ECSD_CLIP_BATCH_SIZE" \
      --image-batch-size "$ECSD_IMAGE_BATCH_SIZE" --device "${ECSD_DEVICE:-auto}" "${extra[@]}" \
      2>&1 | tee "$OUT_DIR/logs/ecsd_generate_${split}.log"
  else
    echo "[ECSD] Reusing $output"
  fi
done

for split in train val; do
  python experiments/full_mscoco14_ecsd/evaluate_cisd.py \
    --raw "$RAW_DIR/raw_${split}.json" --ecsd "$RAW_DIR/ecsd_${split}.json" \
    --output-dir "$OUT_DIR/cisd/$split" --clip-cache "$CLIP_CACHE" \
    --batch-size "$ECSD_CLIP_BATCH_SIZE" --image-batch-size 512 \
    2>&1 | tee "$OUT_DIR/logs/cisd_${split}.log"
done

python experiments/full_mscoco14_ecsd/prepare.py build-base \
  --raw-captions "$RAW_DIR/raw_train.json" --annotations "$TRAIN_ANN" --split train --image-root "$COCO_ROOT" \
  --out "$WORK_DIR/base_train.pth"
python experiments/full_mscoco14_ecsd/prepare.py build-base \
  --raw-captions "$RAW_DIR/raw_val.json" --annotations "$VAL_ANN" --split val --image-root "$COCO_ROOT" \
  --out "$WORK_DIR/base_val.pth"

if [[ "${FORCE_DINO:-0}" == "1" ]]; then
  rm -f "$WORK_DIR/dino_train.pth" "$WORK_DIR/dino_val.pth"
  REUSE_DINO=0
fi

if [[ ! -f "$WORK_DIR/dino_train.pth" ]]; then
  if [[ "$REUSE_DINO" == "1" && -f "$DINO_TRAIN_SOURCE" ]]; then
    echo "[DINO] Validating/reusing $DINO_TRAIN_SOURCE"
    python experiments/full_mscoco14_ecsd/prepare.py reuse-dino \
      --base "$WORK_DIR/base_train.pth" --source "$DINO_TRAIN_SOURCE" \
      --out "$WORK_DIR/dino_train.pth" --required-feature disentangled_self_attn
  else
    echo "[DINO] Extracting Train2014 features"
    python dino_extraction_v2.py --ann_path "$WORK_DIR/base_train.pth" \
      --data_dir "$COCO_ROOT" --out_path "$WORK_DIR/dino_train.pth" \
      --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 \
      --batch_size "$DINO_BATCH_SIZE" --extract_avg_self_attn --extract_disentangled_self_attn \
      2>&1 | tee "$OUT_DIR/logs/dino_train.log"
  fi
else
  echo "[DINO] Reusing existing $WORK_DIR/dino_train.pth"
fi

if [[ ! -f "$WORK_DIR/dino_val.pth" ]]; then
  if [[ "$REUSE_DINO" == "1" && -f "$DINO_VAL_SOURCE" ]]; then
    echo "[DINO] Validating/reusing $DINO_VAL_SOURCE"
    python experiments/full_mscoco14_ecsd/prepare.py reuse-dino \
      --base "$WORK_DIR/base_val.pth" --source "$DINO_VAL_SOURCE" \
      --out "$WORK_DIR/dino_val.pth" --required-feature avg_self_attn_out
  else
    echo "[DINO] Extracting Val2014 features"
    python dino_extraction_v2.py --ann_path "$WORK_DIR/base_val.pth" \
      --data_dir "$COCO_ROOT" --out_path "$WORK_DIR/dino_val.pth" \
      --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 \
      --batch_size "$DINO_BATCH_SIZE" --extract_avg_self_attn --extract_disentangled_self_attn \
      2>&1 | tee "$OUT_DIR/logs/dino_val.log"
  fi
else
  echo "[DINO] Reusing existing $WORK_DIR/dino_val.pth"
fi

for split in train val; do
  if [[ "$split" == "train" ]]; then dino="$WORK_DIR/dino_train.pth"; else dino="$WORK_DIR/dino_val.pth"; fi
  for arm in raw ecsd; do
    python experiments/full_mscoco14_ecsd/prepare.py assemble \
      --raw "$RAW_DIR/raw_${split}.json" --ecsd "$RAW_DIR/ecsd_${split}.json" \
      --dino "$dino" --split "$split" --arm "$arm" \
      --out "$WORK_DIR/${arm}_${split}.pth"
  done
done

for split in train val; do
  for arm in raw ecsd; do
    file="$WORK_DIR/${arm}_${split}.pth"
    if [[ "${FORCE_CLIP:-0}" == "1" ]] || ! python experiments/full_mscoco14_ecsd/prepare.py check-clip --file "$file" >/dev/null 2>&1; then
      echo "[CLIP] Extracting ann_feats: ${arm}_${split}"
      python text_features_extraction.py --ann_path "$file" --out_path "$file" \
        --model ViT-B/16 --batch_size "$CLIP_BATCH_SIZE" \
        2>&1 | tee "$OUT_DIR/logs/clip_${arm}_${split}.log"
    else
      echo "[CLIP] Reusing ann_feats: ${arm}_${split}"
    fi
  done
done

python experiments/full_mscoco14_ecsd/prepare.py audit --work "$WORK_DIR" \
  --expected-train-images "${EXPECTED_TRAIN_IMAGES:-82783}" \
  --expected-val-images "${EXPECTED_VAL_IMAGES:-40504}" \
  2>&1 | tee "$OUT_DIR/logs/audit.log"

echo "[DONE] Full MSCOCO14 ECSD preprocessing completed."
