#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
RAW_DIR="${RAW_DIR:-data/coco2014_scale}"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
CLIP_CACHE="${CLIP_CACHE:-$HOME/.cache/clip}"
mkdir -p "$OUT_ROOT/logs"
[[ -f "$RAW_DIR/raw_train.json" ]] || { echo "[ERROR] Run Stage 1 first." >&2; exit 1; }
ARGS=()
if [[ "${RESUME_ECSD:-0}" == "1" ]]; then ARGS+=(--resume); fi
if [[ "${OVERWRITE_ECSD:-0}" == "1" ]]; then ARGS+=(--overwrite); fi
if [[ "${RESUME_ECSD:-0}" == "1" && "${OVERWRITE_ECSD:-0}" == "1" ]]; then
  echo "[ERROR] Do not combine RESUME_ECSD and OVERWRITE_ECSD." >&2; exit 2
fi
python experiments/ecsd_scale_sensitivity/data/generate_ecsd_b4_full.py \
  --input "$RAW_DIR/raw_train.json" \
  --output "$RAW_DIR/ecsd_train_b4_v3.json" \
  --trace "$OUT_ROOT/ecsd_b4_trace_train.jsonl" \
  --summary "$OUT_ROOT/ecsd_b4_generation_summary.json" \
  --config experiments/5k_ecsd_v3/configs/default.json \
  --clip-cache "$CLIP_CACHE" \
  --image-chunk-size "${ECSD_IMAGE_BATCH_SIZE:-128}" \
  --clip-batch-size "${ECSD_CLIP_BATCH_SIZE:-256}" \
  --device "${ECSD_DEVICE:-auto}" \
  --expected-images "${EXPECTED_TRAIN_IMAGES:-82783}" \
  "${ARGS[@]}" 2>&1 | tee "$OUT_ROOT/logs/ecsd_b4_train.log"
