#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WEIGHT="${1:?Usage: bash run_all_segmentation_eval.sh WEIGHT_STEM}"

BASE="src/open_vocabulary_segmentation/configs"
OUT="outputs/projector_semi_orthogonal/seg_eval/${WEIGHT}"

run_eval() {
    local name="$1"
    local model_cfg="$2"
    local eval_cfg="$3"

    echo ""
    echo "======================================"
    echo "Evaluating: ${name}"
    echo "Weight: ${WEIGHT}"
    echo "======================================"

    bash "${SCRIPT_DIR}/run_segmentation_eval.sh" \
        "${WEIGHT}" \
        "${BASE}/${model_cfg}" \
        "${BASE}/${eval_cfg}" \
        "${OUT}/${name}"
}

# 1. VOC20
run_eval voc20 \
    voc/dinotext_voc_vitb_mlp_infonce.yml \
    voc/eval_voc_pamr.yml

# 2. VOC20b
run_eval voc20b \
    voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml \
    voc_bg/eval_voc_bg_pamr.yml

# 3. Pascal Context 59
run_eval pc59 \
    context/dinotext_context_vitb_mlp_infonce.yml \
    context/eval_context_pamr.yml

# 4. ADE20K 150
run_eval ade150 \
    ade/dinotext_ade_vitb_mlp_infonce.yml \
    ade/eval_ade_pamr.yml

# 5. COCO-Stuff 164K
run_eval coco_stuff164k \
    stuff/dinotext_stuff_vitb_mlp_infonce.yml \
    stuff/eval_stuff_pamr.yml

echo ""
echo "======================================"
echo "All segmentation evaluations finished."
echo "Results: ${OUT}"
echo "======================================"
