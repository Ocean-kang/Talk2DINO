#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
METHOD="${1:?Usage: train_one.sh raw|cne|hvne|cnp [seed] [extra train arguments]}"
SEED="${2:-123}"
shift
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONHASHSEED="$SEED"
RUN_DIR="outputs/minimal_noun_supervision/$MNS_TAG/$METHOD/seed_$SEED"
# Logs live separately so the Python runner can refuse an existing run directory.
LOG_DIR="outputs/minimal_noun_supervision/$MNS_TAG/logs"
mkdir -p "$LOG_DIR"
[[ ! -e "$RUN_DIR" ]] || { echo "Run already exists: $RUN_DIR" >&2; exit 1; }
python -u "$MNS_DIR/train.py" --method "$METHOD" --data "$MNS_DATA" --config "$MNS_CONFIG" \
  --tag "$MNS_TAG" --seed "$SEED" "$@" 2>&1 | tee "$LOG_DIR/${METHOD}_seed${SEED}.log"
