#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/5k_ecsd}"
WORK_DIR="${WORK_DIR:-${OUT_DIR}/work}"
CONFIG="${CONFIG:-configs/vitb_mlp_infonce.yaml}"
WEIGHT_TAG="${WEIGHT_TAG:-ecsd5k}"
TRAIN_ARMS="${TRAIN_ARMS:-raw rich rich_weaken rich_diversity ecsd}"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/config_snapshot" weights
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing config: $CONFIG" >&2
  exit 1
fi

cp "$CONFIG" "$OUT_DIR/config_snapshot/$(basename "$CONFIG")"
sha256sum "$CONFIG" > "$OUT_DIR/config_snapshot/$(basename "$CONFIG").sha256"

read -r -a ARMS <<< "$TRAIN_ARMS"
for arm in "${ARMS[@]}"; do
  case "$arm" in raw|rich|rich_weaken|rich_diversity|ecsd) ;; *)
    echo "[ERROR] Unknown arm: $arm" >&2; exit 1 ;;
  esac
  for split in train val; do
    [[ -f "$WORK_DIR/${arm}_${split}.pth" ]] || {
      echo "[ERROR] Missing $WORK_DIR/${arm}_${split}.pth. Run run_preprocess.sh first." >&2
      exit 1
    }
  done
  WEIGHT_NAME="vitb_mlp_infonce_${WEIGHT_TAG}_${arm}"
  if [[ -e "weights/${WEIGHT_NAME}.pth" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[ERROR] Refusing to overwrite weights/${WEIGHT_NAME}.pth" >&2
    echo "Set OVERWRITE=1 only if you intentionally want to replace it." >&2
    exit 2
  fi
done

echo "============================================================"
echo "Original Talk2DINO training recipe"
echo "Config     : $CONFIG"
echo "Train arms : ${ARMS[*]}"
echo "Weight tag : $WEIGHT_TAG"
echo "100 epochs / batch 128 / lr 1e-4 / InfoNCE / Adam / seed 123"
echo "============================================================"

for arm in "${ARMS[@]}"; do
  WEIGHT_NAME="vitb_mlp_infonce_${WEIGHT_TAG}_${arm}"
  LOG_PATH="$OUT_DIR/logs/train_${arm}.log"
  echo
  echo "-------------------- TRAIN: ${arm} --------------------"
  python train.py \
    --model_config "$CONFIG" \
    --feature_name disentangled_self_attn \
    --text_features ann_feats \
    --train_dataset "$WORK_DIR/${arm}_train.pth" \
    --val_dataset "$WORK_DIR/${arm}_val.pth" \
    --optimizer Adam \
    --scheduler linear \
    --warmup 0 \
    --name_pedix "${WEIGHT_TAG}_${arm}" \
    2>&1 | tee "$LOG_PATH"

  [[ -f "weights/${WEIGHT_NAME}.pth" ]] || {
    echo "[ERROR] Training did not create weights/${WEIGHT_NAME}.pth" >&2
    exit 1
  }
done

python experiments/5k_ecsd/collect_results.py \
  --root "$OUT_DIR" \
  --weights-dir weights \
  --seg-results-dir segmentation_results \
  --weight-tag "$WEIGHT_TAG"

echo "[DONE] Requested ECSD training arms completed."
