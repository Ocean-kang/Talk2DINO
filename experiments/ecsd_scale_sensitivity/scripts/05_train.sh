#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
WORK_DIR="${WORK_DIR:-$OUT_ROOT/work}"
CONFIG="${CONFIG:-configs/vitb_mlp_infonce.yaml}"
SEEDS="${SEEDS:-123 456 789}"
SCALES="${SCALES:-5k 10k 20k 40k full}"
mkdir -p weights "$OUT_ROOT/logs" "$OUT_ROOT/config_snapshot"
[[ -f "$WORK_DIR/audit_report.json" ]] || { echo "[ERROR] Run Stage 4 and pass its audit first." >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "[ERROR] Missing config $CONFIG" >&2; exit 1; }
cp "$CONFIG" "$OUT_ROOT/config_snapshot/"
sha256sum "$CONFIG" > "$OUT_ROOT/config_snapshot/$(basename "$CONFIG").sha256"

for seed in $SEEDS; do
  for scale in $SCALES; do
    for arm in raw ecsd; do
      train_file="$WORK_DIR/$scale/train_${arm}.pth"
      val_file="$WORK_DIR/val_shared.pth"
      [[ -f "$train_file" && -f "$val_file" ]] || { echo "[ERROR] Missing data file for $scale/$arm" >&2; exit 1; }
      suffix="scale${scale}_s${seed}_${arm}"
      weight="weights/vitb_mlp_infonce_${suffix}.pth"
      if [[ -f "$weight" ]]; then
        if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then echo "[SKIP] $weight"; continue; fi
        if [[ "${OVERWRITE:-0}" != "1" ]]; then echo "[ERROR] $weight exists; use SKIP_EXISTING=1 or OVERWRITE=1" >&2; exit 2; fi
      fi
      echo "[TRAIN] scale=$scale seed=$seed arm=$arm"
      python experiments/full_mscoco14_ecsd/train_seeded.py \
        --model_config "$CONFIG" --feature_name disentangled_self_attn \
        --text_features ann_feats --train_dataset "$train_file" --val_dataset "$val_file" \
        --optimizer Adam --scheduler linear --warmup 0 \
        --name_pedix "$suffix" --seed "$seed" \
        2>&1 | tee "$OUT_ROOT/logs/train_${scale}_s${seed}_${arm}.log"
      [[ -f "$weight" ]] || { echo "[ERROR] Training did not create $weight" >&2; exit 1; }
    done
  done
done
echo "[DONE] Training matrix finished."
