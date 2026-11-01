#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
METHOD="${1:?Usage: eval_one.sh raw|cne|hvne|cnp [seed] [extra eval arguments]}"
SEED="${2:-123}"
shift
if [[ $# -gt 0 ]]; then shift; fi
python -u "$MNS_DIR/evaluate.py" --method "$METHOD" --seed "$SEED" --tag "$MNS_TAG" "$@"
