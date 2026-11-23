#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_DIR="${OUT_DIR:-outputs/full_mscoco14_ecsd}"
WORK_DIR="${WORK_DIR:-${OUT_DIR}/work}"
CONFIG="${CONFIG:-configs/vitb_mlp_infonce.yaml}"
SEEDS="${SEEDS:-123}"
TRAIN_ARMS="${TRAIN_ARMS:-raw ecsd}"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/config_snapshot" weights
[[ -f "$CONFIG" ]] || { echo "[ERROR] Missing config: $CONFIG" >&2; exit 1; }
cp "$CONFIG" "$OUT_DIR/config_snapshot/$(basename "$CONFIG")"
sha256sum "$CONFIG" > "$OUT_DIR/config_snapshot/$(basename "$CONFIG").sha256"
for seed in $SEEDS; do
  for arm in $TRAIN_ARMS; do
    [[ "$arm" == "raw" || "$arm" == "ecsd" ]] || { echo "[ERROR] Unknown arm $arm" >&2; exit 1; }
    for split in train val; do
      [[ -f "$WORK_DIR/${arm}_${split}.pth" ]] || { echo "[ERROR] Missing $WORK_DIR/${arm}_${split}.pth; run run_preprocess.sh" >&2; exit 1; }
    done
    suffix="full14_s${seed}_${arm}"
    weight="weights/vitb_mlp_infonce_${suffix}.pth"
    if [[ -f "$weight" ]]; then
      if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then echo "[SKIP] $weight exists"; continue; fi
      if [[ "${OVERWRITE:-0}" != "1" ]]; then echo "[ERROR] Refusing to overwrite $weight; set SKIP_EXISTING=1 or OVERWRITE=1" >&2; exit 2; fi
    fi
    echo "[TRAIN] seed=$seed arm=$arm"
    python experiments/full_mscoco14_ecsd/train_seeded.py \
      --model_config "$CONFIG" \
      --feature_name disentangled_self_attn \
      --text_features ann_feats \
      --train_dataset "$WORK_DIR/${arm}_train.pth" \
      --val_dataset "$WORK_DIR/${arm}_val.pth" \
      --optimizer Adam --scheduler linear --warmup 0 \
      --name_pedix "$suffix" --seed "$seed" \
      2>&1 | tee "$OUT_DIR/logs/train_s${seed}_${arm}.log"
    [[ -f "$weight" ]] || { echo "[ERROR] Training did not create $weight" >&2; exit 1; }
  done
done
echo "[DONE] Requested Raw/ECSD training runs completed."
