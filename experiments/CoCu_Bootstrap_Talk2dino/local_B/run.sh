#!/usr/bin/env bash
set -euo pipefail
# Execute from the Talk2DINO repository root.
# Usage: bash experiments/CoCu_Bootstrap_Talk2dino/local_B/run.sh [all|concepts|build|train|verify] [GPU] [--rebuild-concepts]
STAGE="${1:-all}"
GPU="${2:-0}"
shift $(($# > 2 ? 2 : $#))
python experiments/CoCu_Bootstrap_Talk2dino/pipeline.py \
  --source local --scheme B --stage "$STAGE" --gpu "$GPU" "$@"
