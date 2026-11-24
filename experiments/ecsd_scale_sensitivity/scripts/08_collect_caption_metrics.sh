#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
RAW_DIR="${RAW_DIR:-data/coco2014_scale}"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
WORK_DIR="${WORK_DIR:-$OUT_ROOT/work}"
python experiments/ecsd_scale_sensitivity/analysis/collect_caption_scale_metrics.py \
  --raw "$RAW_DIR/raw_train.json" \
  --ecsd "$RAW_DIR/ecsd_train_b4_v3.json" \
  --trace "$OUT_ROOT/ecsd_b4_trace_train.jsonl" \
  --manifest "$WORK_DIR/subset_manifest.json" \
  --per-image-cisd "$OUT_ROOT/caption_metrics/full_train/per_image_cisd.csv" \
  --output "$OUT_ROOT/summary/caption_scale_metrics.csv" \
  --long-tail-max-image-frequency "${LONG_TAIL_MAX_FREQ:-5}"
