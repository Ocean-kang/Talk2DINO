#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python -m unittest discover -s experiments/5k_cocu_rcm/tests -v

if [[ "${RUN_TRAIN_SMOKE:-0}" == "1" ]]; then
  BASE_TRAIN="${BASE_TRAIN:-outputs/5k_rich_diversity/work/raw_train.pth}"
  BASE_VAL="${BASE_VAL:-outputs/5k_rich_diversity/work/raw_val.pth}"
  WORK_DIR="${WORK_DIR:-outputs/5k_cocu_rcm/work}"
  python experiments/5k_cocu_rcm/training/train_cocu_rcm.py \
    --arm cocu_rcm --seed 123 \
    --base-train "$BASE_TRAIN" --base-val "$BASE_VAL" \
    --augmented-train "$WORK_DIR/train_augmented.pth" \
    --sampling-index "$WORK_DIR/sampling_index_cross_k2.json" \
    --region-features "$WORK_DIR/region_gallery/region_features.pt" \
    --region-index "$WORK_DIR/region_gallery/region_feature_index.json" \
    --config configs/vitb_mlp_infonce.yaml \
    --output-dir outputs/5k_cocu_rcm --weights-dir weights \
    --epochs 1 --batch-size 8 --num-workers 0 --max-steps 2 \
    --lambda-local 0.1 --temperature 0.07 --no-save --tag smoke
fi
echo "[smoke] PASS"
