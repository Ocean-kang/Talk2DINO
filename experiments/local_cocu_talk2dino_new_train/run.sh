#!/usr/bin/env bash
set -euo pipefail
# bash experiments/local_cocu_talk2dino_new_train/run.sh A 5plus1 all 1 [extra flags]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
SCHEME="${1:?Usage: run.sh A 5plus1|5 build|train|eval|all GPU [options]}"
MODE="${2:?Expected 5plus1 or 5}"
STAGE="${3:-all}"
GPU="${4:-0}"
if (( $# >= 4 )); then shift 4; elif (( $# == 3 )); then shift 3; elif (( $# == 2 )); then shift 2; else shift; fi
cd "$ROOT"
exec python "$HERE/pipeline.py" --scheme "$SCHEME" --mode "$MODE" --stage "$STAGE" --gpu "$GPU" "$@"
