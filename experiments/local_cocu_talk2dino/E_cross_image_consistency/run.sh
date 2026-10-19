#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec python experiments/local_cocu_talk2dino/E_cross_image_consistency/run.py "$@"
