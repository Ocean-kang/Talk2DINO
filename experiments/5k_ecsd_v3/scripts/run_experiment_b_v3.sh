#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONFIG="${ECSD_V3_CONFIG:-experiments/5k_ecsd_v3/configs/default.json}"
SOURCE="${SOURCE_CAPTIONS:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["source_captions"])' "$CONFIG")}"
DATA_DIR="${DATA_DIR:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["data_dir"])' "$CONFIG")}"
POOL="${CANDIDATE_POOL:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["candidate_pool"])' "$CONFIG")}"
OUT="${OUTPUT_ROOT:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$CONFIG")}"
CLIP_CACHE="${CLIP_CACHE:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["clip_cache"])' "$CONFIG")}"
EMBED_CACHE="${EMBED_CACHE:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["embedding_cache"])' "$CONFIG")}"
EXPECTED="${EXPECTED_IMAGES:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["expected_images"])' "$CONFIG")}"
BATCH="${CLIP_BATCH_SIZE:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["metrics"]["clip_batch_size"])' "$CONFIG")}"

[[ -f "$POOL" ]] || { echo "[ERROR] Missing v3 candidate pool. Run Stage A first: $POOL" >&2; exit 1; }
mkdir -p "$OUT/logs" "$OUT/selection_traces" "$OUT/ablation_metrics"
GEN_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then GEN_ARGS+=(--overwrite); fi

python experiments/5k_ecsd_v3/selection_ablation/generate_ablation_captions_v3.py \
  --input "$SOURCE" --candidate-pool "$POOL" --config "$CONFIG" \
  --output-dir "$DATA_DIR" --trace-dir "$OUT/selection_traces" \
  --cache-dir "$EMBED_CACHE" --clip-cache "$CLIP_CACHE" \
  --batch-size "$BATCH" --expected-images "$EXPECTED" \
  "${GEN_ARGS[@]}" 2>&1 | tee "$OUT/logs/generate_ablation_v3.log"

METRIC_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then METRIC_ARGS+=(--overwrite); fi
python experiments/5k_ecsd_v3/selection_ablation/evaluate_metrics_v3.py \
  --source "$SOURCE" --candidate-pool "$POOL" --caption-dir "$DATA_DIR" \
  --trace-dir "$OUT/selection_traces" --output-dir "$OUT/ablation_metrics" \
  --config "$CONFIG" --clip-cache "$CLIP_CACHE" --embedding-cache-dir "$EMBED_CACHE" \
  --batch-size "$BATCH" --expected-images "$EXPECTED" \
  "${METRIC_ARGS[@]}" 2>&1 | tee "$OUT/logs/evaluate_metrics_v3.log"

AUDIT_CSV="$OUT/manual_audit/manual_annotation_v3.csv"
AUDIT_HTML="$OUT/manual_audit/manual_audit_viewer_v3.html"
AUDIT_GUIDE="$OUT/manual_audit/ANNOTATION_GUIDELINES_v3.md"
if [[ -f "$AUDIT_CSV" || -f "$AUDIT_HTML" || -f "$AUDIT_GUIDE" ]]; then
  echo "[PRESERVE] Existing manual audit artifacts kept unchanged: $OUT/manual_audit"
  echo "[INFO] Use migrate_manual_annotations_v3.py for label migration, or export manually to a new folder."
else
  python experiments/5k_ecsd_v3/candidate_quality/export_manual_audit_v3.py \
    --source "$SOURCE" --candidate-pool "$POOL" \
    --trace "$OUT/selection_traces/B4_selection_trace.jsonl" \
    --output-dir "$OUT/manual_audit" \
    --sample-size "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["manual_audit_images"])' "$CONFIG")" \
    --seed "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["manual_audit_seed"])' "$CONFIG")"
fi

echo "[DONE] Stage B v3. Outputs: $DATA_DIR and $OUT"
