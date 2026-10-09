#!/usr/bin/env bash
set -euo pipefail
# Runs six arms sequentially on one GPU. Extracts each CoCu source once,
# caches it and constructs independent fusion datasets for all three schemes.
# Usage: bash experiments/CoCu_Bootstrap_Talk2dino/run_all.sh [GPU]
GPU="${1:-0}"
for SOURCE in clip local; do
  for SCHEME in 5plus3 B D; do
    echo "===== $SOURCE x $SCHEME ====="
    bash "experiments/CoCu_Bootstrap_Talk2dino/${SOURCE}_${SCHEME}/run.sh" all "$GPU"
  done
done
