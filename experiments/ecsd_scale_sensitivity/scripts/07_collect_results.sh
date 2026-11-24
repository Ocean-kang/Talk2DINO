#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
python experiments/ecsd_scale_sensitivity/analysis/collect_scale_results.py \
  --output-root "$OUT_ROOT" --weights-dir "${WEIGHTS_DIR:-weights}" \
  --seg-results-dir "${SEG_RESULTS_DIR:-segmentation_results}" \
  --scales "${SCALES:-5k 10k 20k 40k full}" \
  --seeds "${SEEDS:-123 456 789}" \
  --benchmarks "${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"
python experiments/ecsd_scale_sensitivity/analysis/plot_scale_sensitivity.py \
  --summary-dir "$OUT_ROOT/summary" \
  --benchmarks "${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"
