#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_TRAIN="${BASE_TRAIN:-outputs/5k_rich_diversity/work/raw_train.pth}"
BASE_VAL="${BASE_VAL:-outputs/5k_rich_diversity/work/raw_val.pth}"
WORK_DIR="${WORK_DIR:-outputs/5k_cocu_rcm/work}"
CONFIG="${CONFIG:-configs/vitb_mlp_infonce.yaml}"
TRAIN_ARMS="${TRAIN_ARMS:-raw cocu cocu_same cocu_rcm cocu_rcm_shuffled}"
SEEDS="${SEEDS:-123}"
K_SHOT="${K_SHOT:-2}"
LAMBDA_LOCAL="${LAMBDA_LOCAL:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.07}"
OVERWRITE="${OVERWRITE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

for f in "$BASE_TRAIN" "$BASE_VAL" "$WORK_DIR/train_augmented.pth" \
  "$WORK_DIR/region_gallery/region_features.pt" \
  "$WORK_DIR/region_gallery/region_feature_index.json"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing $f. Run run_preprocess.sh first." >&2; exit 1; }
done

read -r -a ARMS <<< "$TRAIN_ARMS"
read -r -a SEED_LIST <<< "$SEEDS"
mkdir -p outputs/5k_cocu_rcm/logs weights

for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    case "$arm" in
      raw|cocu) SAMPLE="" ;;
      cocu_same) SAMPLE="$WORK_DIR/sampling_index_same_k${K_SHOT}.json" ;;
      cocu_rcm) SAMPLE="$WORK_DIR/sampling_index_cross_k${K_SHOT}.json" ;;
      cocu_rcm_shuffled) SAMPLE="$WORK_DIR/sampling_index_shuffled_k${K_SHOT}.json" ;;
      *) echo "[ERROR] Unknown arm: $arm" >&2; exit 2 ;;
    esac
    WEIGHT="weights/vitb_mlp_infonce_5k_cocu_rcm_${arm}_seed${seed}.pth"
    if [[ -f "$WEIGHT" && "$OVERWRITE" != "1" ]]; then
      echo "[ERROR] Refusing to overwrite $WEIGHT. Set OVERWRITE=1 deliberately." >&2
      exit 2
    fi
    LOG="outputs/5k_cocu_rcm/logs/train_${arm}_seed${seed}.log"
    EXTRA=()
    if [[ -n "$SAMPLE" ]]; then
      [[ -f "$SAMPLE" ]] || { echo "[ERROR] Missing $SAMPLE" >&2; exit 1; }
      EXTRA+=(--sampling-index "$SAMPLE"
        --region-features "$WORK_DIR/region_gallery/region_features.pt"
        --region-index "$WORK_DIR/region_gallery/region_feature_index.json")
    fi
    [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
    python experiments/5k_cocu_rcm/training/train_cocu_rcm.py \
      --arm "$arm" --seed "$seed" \
      --base-train "$BASE_TRAIN" --base-val "$BASE_VAL" \
      --augmented-train "$WORK_DIR/train_augmented.pth" \
      --config "$CONFIG" --output-dir outputs/5k_cocu_rcm --weights-dir weights \
      --lambda-local "$LAMBDA_LOCAL" --temperature "$TEMPERATURE" \
      --num-workers "$NUM_WORKERS" "${EXTRA[@]}" 2>&1 | tee "$LOG"
  done
done
