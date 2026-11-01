#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -u "$MNS_DIR/select_hvne.py" --data "$MNS_DATA" \
  --teacher "${MNS_TEACHER:?Set MNS_TEACHER to an existing local Teacher weight}" \
  --model-config "$MNS_CONFIG" "$@"
