#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PATCH="experiments/5k_ecsd_next/patches/train_seed.patch"

if grep -q "parser.add_argument('--seed', type=int, default=123" train.py; then
  echo "[INFO] train.py already exposes --seed; no patch needed."
  exit 0
fi
git apply --check "$PATCH"
git apply "$PATCH"
python -m py_compile train.py src/train_util.py
echo "[DONE] Added backward-compatible --seed support."
