#!/usr/bin/env bash
# Run in Talk2DINO root: bash experiments/Head_caption_completeness/run.sh E3 123 all 0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VARIANT="${1:-E3}"
SEED="${2:-123}"
STAGE="${3:-all}"
GPU="${4:-0}"
shift $(( $# >= 4 ? 4 : $# ))
python -m experiments.Head_caption_completeness.pipeline --variant "$VARIANT" --seed "$SEED" \
    --stage "$STAGE" --gpu "$GPU" "$@"
