#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then
 echo 'Usage: bash experiments/Head_Match/run.sh <A|B|C|D> [all|mine|build|train|eval] [gpu] [rounds] [extra options]'
 exit 2
fi
VARIANT="$1"; STAGE="${2:-all}"; GPU="${3:-0}"; ROUNDS="${4:-2}"
shift $(( $# > 4 ? 4 : $# ))
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python -m experiments.Head_Match.pipeline --variant "$VARIANT" --stage "$STAGE" --gpu "$GPU" --rounds "$ROUNDS" "$@"
