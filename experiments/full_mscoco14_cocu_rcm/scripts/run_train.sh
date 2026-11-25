#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

WORK_DIR="${WORK_DIR:-outputs/full_mscoco14_cocu_rcm/work}"
OUT_DIR="${OUT_DIR:-outputs/full_mscoco14_cocu_rcm}"
CONFIG="${CONFIG:-configs/vitb_mlp_infonce.yaml}"
TRAIN_ARMS="${TRAIN_ARMS:-raw cocu cocu_same cocu_rcm cocu_rcm_shuffled}"
SEEDS="${SEEDS:-123}"
K_SHOT="${K_SHOT:-2}"
LAMBDA_LOCAL="${LAMBDA_LOCAL:-}"
TEMPERATURE="${TEMPERATURE:-0.07}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
RESTART="${RESTART:-0}"

for f in "$WORK_DIR/raw_train.pth" "$WORK_DIR/raw_val.pth" "$WORK_DIR/train_augmented.pth" "$CONFIG"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing $f; run scripts/run_preprocess.sh first." >&2; exit 1; }
done

read -r -a ARMS <<< "$TRAIN_ARMS"
read -r -a SEED_LIST <<< "$SEEDS"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/checkpoints" weights

for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    case "$arm" in
      raw|cocu) SAMPLE="" ;;
      cocu_same|cocu_rcm|cocu_rcm_shuffled) SAMPLE="$WORK_DIR/sampling_index_${arm#cocu_}_k${K_SHOT}" ;;
      *) echo "[ERROR] Unknown arm: $arm" >&2; exit 2 ;;
    esac
    # Map the arm name to the on-disk directory names.
    case "$arm" in
      cocu_same) SAMPLE="$WORK_DIR/sampling_index_same_k${K_SHOT}" ;;
      cocu_rcm) SAMPLE="$WORK_DIR/sampling_index_cross_k${K_SHOT}" ;;
      cocu_rcm_shuffled) SAMPLE="$WORK_DIR/sampling_index_shuffled_k${K_SHOT}" ;;
    esac
    WEIGHT="weights/vitb_mlp_infonce_full_mscoco14_cocu_rcm_${arm}_seed${seed}.pth"
    if [[ -f "$WEIGHT" && "$SKIP_EXISTING" == "1" ]]; then
      echo "[SKIP] $WEIGHT"; continue
    fi
    if [[ -f "$WEIGHT" && "$OVERWRITE" != "1" ]]; then
      echo "[ERROR] Refusing to overwrite $WEIGHT. Set SKIP_EXISTING=1 or OVERWRITE=1." >&2
      exit 2
    fi
    EXTRA=()
    if [[ -n "$SAMPLE" ]]; then
      [[ -f "$SAMPLE/manifest.json" ]] || { echo "[ERROR] Missing $SAMPLE/manifest.json" >&2; exit 1; }
      EXTRA+=(--sampling-dir "$SAMPLE"
        --region-feature-dir "$WORK_DIR/region_gallery/features"
        --region-index "$WORK_DIR/region_gallery/region_feature_index.json")
    fi
    [[ -n "$LAMBDA_LOCAL" ]] && EXTRA+=(--lambda-local "$LAMBDA_LOCAL")
    [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
    [[ "$RESTART" == "1" ]] && EXTRA+=(--restart)
    LOG="$OUT_DIR/logs/train_${arm}_seed${seed}.log"
    python experiments/full_mscoco14_cocu_rcm/training/train_full_cocu_rcm.py \
      --arm "$arm" --seed "$seed" \
      --base-train "$WORK_DIR/raw_train.pth" --base-val "$WORK_DIR/raw_val.pth" \
      --augmented-train "$WORK_DIR/train_augmented.pth" \
      --config "$CONFIG" --arm-config "experiments/full_mscoco14_cocu_rcm/configs/${arm}.yaml" \
      --output-dir "$OUT_DIR" --weights-dir weights \
      --temperature "$TEMPERATURE" --num-workers "$NUM_WORKERS" \
      "${EXTRA[@]}" 2>&1 | tee "$LOG"
    [[ -f "$WEIGHT" ]] || {
      if [[ -f "$OUT_DIR/checkpoints/${arm}_seed${seed}.resume.pt" ]]; then
        echo "[PARTIAL] $arm seed=$seed saved an epoch resume checkpoint; rerun to continue."
      else
        echo "[ERROR] Training did not create $WEIGHT" >&2; exit 1
      fi
    }
  done
done
