#!/usr/bin/env bash
# Sequential S0-S6 benchmark on the same image subset; avoids redundant calibration/embedding.
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${1:-0}"
LIMIT="${2:-800}"
OUT="${3:-outputs/Head_caption_completeness_diagnosis/run${LIMIT}}"
shift $(( $#>=3 ? 3 : $# ))
CUDA_VISIBLE_DEVICES="$GPU" python -m experiments.Head_caption_completeness_diagnosis.run \
  --device cuda --limit "$LIMIT" --output "$OUT" --methods S0,S1,S2,S3,S4,S5,S6 "$@"
