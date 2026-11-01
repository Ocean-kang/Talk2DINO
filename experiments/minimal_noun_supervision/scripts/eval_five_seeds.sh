#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
METHOD="${1:?Usage: eval_five_seeds.sh raw|cne|hvne|cnp [extra eval arguments]}"
shift
for SEED in 123 124 125 126 127; do
  bash "$MNS_DIR/scripts/eval_one.sh" "$METHOD" "$SEED" "$@"
done
