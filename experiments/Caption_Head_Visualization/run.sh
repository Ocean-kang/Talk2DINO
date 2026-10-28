#!/usr/bin/env bash
set -euo pipefail
# Usage from Talk2DINO repo root:
# CUDA_VISIBLE_DEVICES=0 bash experiments/Caption_Head_Visualization/run.sh /path/to/COCO/root [extra args...]
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash experiments/Caption_Head_Visualization/run.sh COCO_IMAGE_ROOT [additional run.py args]' >&2
  exit 2
fi
IMAGE_ROOT="$1"
shift
python -m experiments.Caption_Head_Visualization.run --image-root "$IMAGE_ROOT" "$@"
