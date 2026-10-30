#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WEIGHT_STEM="${1:?Usage: bash run_segmentation_eval.sh WEIGHT_STEM [EVAL_CFG] [EVAL_BASE_CFG] [OUTPUT_DIR]}"
EVAL_CFG="${2:-src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml}"
EVAL_BASE_CFG="${3:-src/open_vocabulary_segmentation/configs/voc/eval_voc_pamr.yml}"
OUTPUT_DIR="${4:-outputs/projector_semi_orthogonal/seg_eval/${WEIGHT_STEM}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ ! -f "weights/${WEIGHT_STEM}.pth" ]]; then
  echo "Checkpoint not found: weights/${WEIGHT_STEM}.pth" >&2
  exit 1
fi

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  src/open_vocabulary_segmentation/main.py \
  --eval \
  --eval_cfg "${EVAL_CFG}" \
  --eval_base_cfg "${EVAL_BASE_CFG}" \
  --output "${OUTPUT_DIR}" \
  --opts "model.proj_name=${WEIGHT_STEM}"

echo "Evaluation output: ${OUTPUT_DIR}"
