#!/usr/bin/env bash
set -euo pipefail
# bash experiments/local_cocu_novel/run.sh [all|extract|filter|match|build|train|eval] [GPU] [extra pipeline arguments]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
STAGE="${1:-all}"
GPU="${2:-0}"
if (($# >= 2)); then shift 2; elif (($# == 1)); then shift 1; fi
exec python "$HERE/pipeline.py" --stage "$STAGE" --gpu "$GPU" "$@"
