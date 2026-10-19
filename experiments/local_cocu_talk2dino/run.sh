#!/usr/bin/env bash
set -euo pipefail
# bash experiments/local_cocu_talk2dino/run.sh A all 1 [options]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SCHEME="${1:?Expected A..H}"
STAGE="${2:-all}"
GPU="${3:-0}"
if [ $# -ge 3 ]; then shift 3; elif [ $# -eq 2 ]; then shift 2; else shift; fi
case "$SCHEME" in
 A) folder=A_direct_retrieval;;
 B) folder=B_head_image_retrieval;;
 C) folder=C_uncovered_heads;;
 D) folder=D_region_discovery;;
 E) folder=E_cross_image_consistency;;
 F) folder=F_counterfactual;;
 G) folder=G_multiview_agreement;;
 H) folder=H_iterative_bootstrap;;
 *) echo "Scheme must be A-H" >&2; exit 2;;
esac
exec python "experiments/local_cocu_talk2dino/$folder/run.py" --stage "$STAGE" --gpu "$GPU" "$@"
