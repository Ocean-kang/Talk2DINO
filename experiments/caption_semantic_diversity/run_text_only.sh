#!/usr/bin/env bash
set -euo pipefail

CAPTIONS_JSON="${1:-data/coco2014_subset_50k/5k_rich_diversity.json}"
OUTPUT_DIR="${2:-experiments/caption_semantic_diversity/outputs}"

python experiments/caption_semantic_diversity/eval_caption_diversity.py \
  --captions-json "${CAPTIONS_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --clip-model ViT-B/16 \
  --batch-size 256 \
  --bootstrap-iterations 10000 \
  --seed 42
