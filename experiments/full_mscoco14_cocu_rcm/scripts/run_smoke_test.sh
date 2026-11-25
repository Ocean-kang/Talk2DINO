#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

python -m py_compile \
  experiments/full_mscoco14_cocu_rcm/common/*.py \
  experiments/full_mscoco14_cocu_rcm/data/*.py \
  experiments/full_mscoco14_cocu_rcm/training/*.py \
  experiments/full_mscoco14_cocu_rcm/models/*.py \
  experiments/full_mscoco14_cocu_rcm/diagnostics/*.py \
  experiments/full_mscoco14_cocu_rcm/evaluation/*.py
bash -n experiments/full_mscoco14_cocu_rcm/scripts/*.sh
python -m unittest discover -s experiments/full_mscoco14_cocu_rcm/tests -v

if [[ "${RUN_GPU_SMOKE:-0}" == "1" ]]; then
  WORK_DIR="${WORK_DIR:-outputs/full_mscoco14_cocu_rcm/work}"
  python experiments/full_mscoco14_cocu_rcm/training/train_full_cocu_rcm.py \
    --arm cocu_rcm --seed 123 \
    --base-train "$WORK_DIR/raw_train.pth" --base-val "$WORK_DIR/raw_val.pth" \
    --augmented-train "$WORK_DIR/train_augmented.pth" \
    --sampling-dir "$WORK_DIR/sampling_index_cross_k${K_SHOT:-2}" \
    --region-feature-dir "$WORK_DIR/region_gallery/features" \
    --region-index "$WORK_DIR/region_gallery/region_feature_index.json" \
    --config "${CONFIG:-configs/vitb_mlp_infonce.yaml}" \
    --output-dir outputs/full_mscoco14_cocu_rcm \
    --weights-dir weights --max-steps 2 --num-workers 0 --no-save
fi
echo "[PASS] Full MSCOCO14 CoCu-RCM static/unit smoke tests completed."
