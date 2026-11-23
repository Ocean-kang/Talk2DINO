#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ECSD_CONFIG="${ECSD_CONFIG:-experiments/5k_ecsd_next/configs/default.json}"
CFG_DATA_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["data_dir"])' "$ECSD_CONFIG")"
CFG_SOURCE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["source_captions"])' "$ECSD_CONFIG")"
CFG_OUT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$ECSD_CONFIG")"
CFG_CLIP="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["clip_cache"])' "$ECSD_CONFIG")"
DATA_DIR="${DATA_DIR:-$CFG_DATA_DIR}"
SOURCE_CAPTIONS="${SOURCE_CAPTIONS:-$CFG_SOURCE}"
OUT_DIR="${OUT_DIR:-$CFG_OUT}"
CLIP_CACHE="${CLIP_CACHE:-$CFG_CLIP}"
mkdir -p "$OUT_DIR/logs"

[[ -f "$OUT_DIR/candidates/ecsd_v2_candidates.jsonl" ]] || {
  echo "[ERROR] Candidate pool missing. Run scripts/run_experiment_a.sh first." >&2; exit 1;
}
[[ -f "$DATA_DIR/5k_ECSD_v2.json" ]] || {
  echo "[ERROR] 5k_ECSD_v2.json missing. Run Stage A first." >&2; exit 1;
}

ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi

python experiments/5k_ecsd_next/selection_ablation/generate_ablation_captions.py \
  --input "$SOURCE_CAPTIONS" \
  --candidate-pool "$OUT_DIR/candidates/ecsd_v2_candidates.jsonl" \
  --output-dir "$DATA_DIR" \
  --trace-dir "$OUT_DIR/selection_traces" \
  --methods B1 B2 B3 B4 \
  --write-v2 \
  --clip-cache "$CLIP_CACHE" \
  --batch-size "${CLIP_BATCH_SIZE:-256}" \
  "${ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/logs/selection_ablation_generation.log"

python experiments/5k_ecsd_next/selection_ablation/evaluate_metrics.py \
  --base-captions "$DATA_DIR/5k_ECSD_v2.json" \
  --ablation-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR/ablation_metrics" \
  --clip-cache "$CLIP_CACHE" \
  --batch-size "${CLIP_BATCH_SIZE:-256}" \
  --expected-images 5000 \
  --include-ablation \
  2>&1 | tee "$OUT_DIR/logs/ablation_metrics.log"

python experiments/5k_ecsd_next/selection_ablation/analyze_ablation.py \
  --summary "$OUT_DIR/ablation_metrics/summary.csv" \
  --output-dir "$OUT_DIR/ablation_analysis"

echo "[DONE] Stage B outputs: $OUT_DIR"
