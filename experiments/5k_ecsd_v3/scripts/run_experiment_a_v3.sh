#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONFIG="${ECSD_V3_CONFIG:-experiments/5k_ecsd_v3/configs/default.json}"
SOURCE="${SOURCE_CAPTIONS:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["source_captions"])' "$CONFIG")}"
POOL="${CANDIDATE_POOL:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["candidate_pool"])' "$CONFIG")}"
OUT="${OUTPUT_ROOT:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$CONFIG")}"
EXPECTED="${EXPECTED_IMAGES:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["expected_images"])' "$CONFIG")}"
mkdir -p "$OUT/candidates" "$OUT/logs"
ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then ARGS+=(--overwrite); fi

python experiments/5k_ecsd_v3/candidate_quality/build_candidate_pool_v3.py \
  --input "$SOURCE" --output "$POOL" \
  --summary "$OUT/candidates/candidate_pool_summary_v3.json" \
  --config "$CONFIG" --expected-images "$EXPECTED" \
  "${ARGS[@]}" 2>&1 | tee "$OUT/logs/build_candidate_pool_v3.log"

echo "[DONE] Stage A v3: $POOL"
