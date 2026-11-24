#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
RAW_DIR="${RAW_DIR:-data/coco2014_scale}"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
CLIP_CACHE="${CLIP_CACHE:-$HOME/.cache/clip}"
[[ -f "$RAW_DIR/raw_train.json" && -f "$RAW_DIR/ecsd_train_b4_v3.json" ]] || {
  echo "[ERROR] Run Raw and ECSD generation first." >&2; exit 1;
}
python experiments/full_mscoco14_ecsd/evaluate_cisd.py \
  --raw "$RAW_DIR/raw_train.json" \
  --ecsd "$RAW_DIR/ecsd_train_b4_v3.json" \
  --output-dir "$OUT_ROOT/caption_metrics/full_train" \
  --clip-cache "$CLIP_CACHE" \
  --batch-size "${ECSD_CLIP_BATCH_SIZE:-256}" \
  --image-batch-size "${CISD_IMAGE_BATCH_SIZE:-512}" \
  2>&1 | tee "$OUT_ROOT/logs/cisd_full_train.log"
