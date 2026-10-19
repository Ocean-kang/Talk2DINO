#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec python experiments/local_cocu_talk2dino/A_direct_retrieval/run.py "$@"
