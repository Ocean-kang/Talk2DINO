#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ECSD_CONFIG="${ECSD_CONFIG:-experiments/5k_ecsd_next/configs/default.json}"
CFG_DATA_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["data_dir"])' "$ECSD_CONFIG")"
CFG_SOURCE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["source_captions"])' "$ECSD_CONFIG")"
CFG_OUT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$ECSD_CONFIG")"
CFG_CLIP="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["clip_cache"])' "$ECSD_CONFIG")"
CFG_IMAGES="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["expected_images"])' "$ECSD_CONFIG")"
DATA_DIR="${DATA_DIR:-$CFG_DATA_DIR}"
SOURCE_CAPTIONS="${SOURCE_CAPTIONS:-$CFG_SOURCE}"
LEGACY_SELECTED="${LEGACY_SELECTED:-${DATA_DIR}/5k_ECSD.json}"
OUT_DIR="${OUT_DIR:-$CFG_OUT}"
CLIP_CACHE="${CLIP_CACHE:-$CFG_CLIP}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-$CFG_IMAGES}"

[[ -f "$SOURCE_CAPTIONS" ]] || { echo "[ERROR] Missing $SOURCE_CAPTIONS" >&2; exit 1; }
mkdir -p "$OUT_DIR/legacy" "$OUT_DIR/logs"

# Reuse an existing legacy ECSD output if present. Otherwise generate it into the new output tree.
if [[ -f "$LEGACY_SELECTED" ]]; then
  echo "[A] Reusing legacy selected captions: $LEGACY_SELECTED"
else
  LEGACY_SELECTED="$OUT_DIR/legacy/5k_ECSD_legacy.json"
  echo "[A] Creating legacy ECSD baseline in $LEGACY_SELECTED"
  python experiments/5k_ecsd/generate_ecsd.py \
    --input "$SOURCE_CAPTIONS" \
    --output "$LEGACY_SELECTED" \
    --trace "$OUT_DIR/legacy/legacy_selection_trace.jsonl" \
    --summary "$OUT_DIR/legacy/legacy_generation_summary.json" \
    --clip-cache "$CLIP_CACHE"
fi

python experiments/5k_ecsd_next/candidate_quality/analyze_candidates.py \
  --input "$SOURCE_CAPTIONS" \
  --selected-json "$LEGACY_SELECTED" \
  --selected-field ecsd_captions \
  --output-dir "$OUT_DIR/candidate_quality" \
  --expected-images "$EXPECTED_IMAGES" \
  2>&1 | tee "$OUT_DIR/logs/candidate_quality.log"

ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi
python experiments/5k_ecsd_next/candidate_quality/improve_candidate_construction.py \
  --input "$SOURCE_CAPTIONS" \
  --output "$OUT_DIR/candidates/ecsd_v2_candidates.jsonl" \
  --summary "$OUT_DIR/candidates/candidate_pool_summary.json" \
  --expected-images "$EXPECTED_IMAGES" \
  "${ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/logs/improved_candidate_construction.log"

GEN_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then GEN_ARGS+=(--overwrite); fi
python experiments/5k_ecsd_next/selection_ablation/generate_ablation_captions.py \
  --input "$SOURCE_CAPTIONS" \
  --candidate-pool "$OUT_DIR/candidates/ecsd_v2_candidates.jsonl" \
  --output-dir "$DATA_DIR" \
  --trace-dir "$OUT_DIR/selection_traces" \
  --methods B4 \
  --write-v2 \
  --v2-only \
  --clip-cache "$CLIP_CACHE" \
  --batch-size "${CLIP_BATCH_SIZE:-256}" \
  "${GEN_ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/logs/ecsd_v2_generation.log"

AUDIT_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then AUDIT_ARGS+=(--overwrite); fi
python experiments/5k_ecsd_next/candidate_quality/export_manual_audit.py \
  --source "$SOURCE_CAPTIONS" \
  --candidate-pool "$OUT_DIR/candidates/ecsd_v2_candidates.jsonl" \
  --selected-json "$DATA_DIR/5k_ECSD_v2.json" \
  --selected-field ecsd_v2_captions \
  --output-dir "$OUT_DIR/manual_audit" \
  --sample-size 100 \
  --seed 42 \
  "${AUDIT_ARGS[@]}"

echo "[DONE] Stage A outputs: $OUT_DIR"
