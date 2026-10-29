#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VARIANT="${1:-E3}"
SEED="${2:-123}"
STAGE="${3:-all}"
GPU="${4:-0}"
shift "$(( $# >= 4 ? 4 : $# ))"
CUDA_VISIBLE_DEVICES="$GPU" python -m experiments.Head_caption_S5_Editing.pipeline \
  --variant "$VARIANT" --seed "$SEED" --stage "$STAGE" --gpu "$GPU" "$@"
