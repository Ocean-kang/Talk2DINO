#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/raw_rich_weaken_5k}"
WORK_DIR="${WORK_DIR:-${OUT_DIR}/work}"
CONFIG="configs/vitb_mlp_infonce.yaml"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/config_snapshot" weights

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing original Talk2DINO config: $CONFIG" >&2
  exit 1
fi

for arm in raw rich rich_weaken; do
  for split in train val; do
    if [[ ! -f "$WORK_DIR/${arm}_${split}.pth" ]]; then
      echo "[ERROR] Missing $WORK_DIR/${arm}_${split}.pth. Run run_preprocess.sh first." >&2
      exit 1
    fi
  done
done

cp "$CONFIG" "$OUT_DIR/config_snapshot/vitb_mlp_infonce.yaml"
sha256sum "$CONFIG" > "$OUT_DIR/config_snapshot/vitb_mlp_infonce.sha256"

for arm in raw rich rich_weaken; do
  WEIGHT_NAME="vitb_mlp_infonce_subset5k_${arm}"
  WEIGHT_PATH="weights/${WEIGHT_NAME}.pth"

  if [[ -e "$WEIGHT_PATH" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "[ERROR] Refusing to overwrite existing weight: $WEIGHT_PATH" >&2
    echo "Set OVERWRITE=1 to intentionally retrain and replace it." >&2
    exit 2
  fi
done

echo "============================================================"
echo "Training with the original Talk2DINO train.py / do_train()"
echo "Config: $CONFIG"
echo "Expected config values: 100 epochs, batch 128, lr 1e-4, InfoNCE"
echo "Optimizer: Adam; seed: 123; no warmup / no scheduler decay"
echo "============================================================"

for arm in raw rich rich_weaken; do
  WEIGHT_NAME="vitb_mlp_infonce_subset5k_${arm}"
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
    --name_pedix "subset5k_${arm}" \
    2>&1 | tee "$LOG_PATH"

  if [[ ! -f "weights/${WEIGHT_NAME}.pth" ]]; then
    echo "[ERROR] Training ended without producing weights/${WEIGHT_NAME}.pth" >&2
    exit 1
  fi
done

python experiments/raw_rich_weaken_5k/collect_results.py \
  --root "$OUT_DIR" \
  --weights-dir weights \
  --seg-results-dir segmentation_results

echo "[DONE] All three caption arms have finished training."
