#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
COCO_ROOT="${COCO_ROOT:-coco}"
TRAIN_ANN="${TRAIN_ANN:-$COCO_ROOT/annotations/captions_train2014.json}"
VAL_ANN="${VAL_ANN:-$COCO_ROOT/annotations/captions_val2014.json}"
RAW_DIR="${RAW_DIR:-data/coco2014_scale}"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
mkdir -p "$RAW_DIR" "$OUT_ROOT/logs"
for f in "$TRAIN_ANN" "$VAL_ANN"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing annotation file: $f" >&2; exit 1; }
done
if [[ -f "$RAW_DIR/raw_train.json" || -f "$RAW_DIR/raw_val.json" ]]; then
  if [[ "${OVERWRITE_RAW:-0}" != "1" ]]; then
    echo "[ERROR] Raw outputs already exist. Set OVERWRITE_RAW=1 to rebuild." >&2
    exit 2
  fi
fi
python experiments/full_mscoco14_ecsd/build_raw_captions.py \
  --train-annotations "$TRAIN_ANN" --val-annotations "$VAL_ANN" \
  --output-dir "$RAW_DIR" 2>&1 | tee "$OUT_ROOT/logs/build_raw.log"
