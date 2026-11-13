#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash experiments/caption_semantic_diversity/run_with_grounding.sh <captions_json> <coco_image_dir> [output_dir]"
  echo "Example: bash .../run_with_grounding.sh data/coco2014_subset_50k/5k_rich_diversity.json /data/coco/train2014"
  exit 1
fi

CAPTIONS_JSON="$1"
IMAGE_DIR="$2"
OUTPUT_DIR="${3:-experiments/caption_semantic_diversity/outputs_with_grounding}"

python experiments/caption_semantic_diversity/eval_caption_diversity.py \
  --captions-json "${CAPTIONS_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --clip-model ViT-B/16 \
  --image-dir "${IMAGE_DIR}" \
  --image-pattern 'COCO_train2014_{image_id:012d}.jpg' \
  --batch-size 128 \
  --bootstrap-iterations 10000 \
  --seed 42
