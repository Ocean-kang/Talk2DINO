#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ECSD_CONFIG="${ECSD_CONFIG:-experiments/5k_ecsd_next/configs/default.json}"
CFG_OUT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$ECSD_CONFIG")/multiseed"
CFG_TRAIN_CONFIG="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["training_config"])' "$ECSD_CONFIG")"
CFG_ARMS="$(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["training"]["arms"]))' "$ECSD_CONFIG")"
CFG_SEEDS="$(python -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["training"]["seeds"])))' "$ECSD_CONFIG")"
OUT_DIR="${OUT_DIR:-$CFG_OUT}"
WORK_DIR="${WORK_DIR:-${OUT_DIR}/work}"
CONFIG="${CONFIG:-$CFG_TRAIN_CONFIG}"
TRAIN_ARMS="${TRAIN_ARMS:-$CFG_ARMS}"
SEEDS="${SEEDS:-$CFG_SEEDS}"

grep -q "parser.add_argument('--seed'" train.py || {
  echo "[ERROR] train.py does not expose --seed. Apply experiments/5k_ecsd_next/patches/train_seed.patch first." >&2
  exit 1
}
[[ -f "$CONFIG" ]] || { echo "[ERROR] Missing config $CONFIG" >&2; exit 1; }
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/config_snapshot" "$OUT_DIR/run_configs" weights
cp "$CONFIG" "$OUT_DIR/config_snapshot/$(basename "$CONFIG")"
sha256sum "$CONFIG" > "$OUT_DIR/config_snapshot/$(basename "$CONFIG").sha256"

read -r -a ARMS <<< "$TRAIN_ARMS"
read -r -a SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    case "$arm" in raw|rich|rich_diversity|ecsdv2) ;; *)
      echo "[ERROR] Unknown arm $arm" >&2; exit 1 ;;
    esac
    for split in train val; do
      [[ -f "$WORK_DIR/${arm}_${split}.pth" ]] || {
        echo "[ERROR] Missing $WORK_DIR/${arm}_${split}.pth. Run Stage C preparation first." >&2
        exit 1
      }
    done
    WEIGHT_NAME="vitb_mlp_infonce_${arm}_seed${seed}"
    if [[ -e "weights/${WEIGHT_NAME}.pth" && "${OVERWRITE:-0}" != "1" ]]; then
      echo "[ERROR] Refusing to overwrite weights/${WEIGHT_NAME}.pth" >&2
      echo "Set OVERWRITE=1 only when intentionally retraining this exact run." >&2
      exit 2
    fi
  done
done

echo "[INFO] Seeds: ${SEED_LIST[*]}"
echo "[INFO] Arms : ${ARMS[*]}"
echo "[INFO] Recipe: original vitb_mlp_infonce.yaml (100 epochs, batch 128, lr 1e-4, InfoNCE, Adam)"

for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    WEIGHT_NAME="vitb_mlp_infonce_${arm}_seed${seed}"
    LOG_PATH="$OUT_DIR/logs/train_${arm}_seed${seed}.log"
    RUN_CONFIG="$OUT_DIR/run_configs/${arm}_seed${seed}.json"
    python - "$RUN_CONFIG" "$seed" "$arm" "$CONFIG" "$WORK_DIR" "$WEIGHT_NAME" <<'PY'
import hashlib, json, os, sys
out, seed, arm, config, work, weight = sys.argv[1:]
with open(config, "rb") as handle:
    digest = hashlib.sha256(handle.read()).hexdigest()
record = {
    "seed": int(seed), "caption_arm": arm,
    "training_config": config, "training_config_sha256": digest,
    "train_dataset": os.path.join(work, arm + "_train.pth"),
    "validation_dataset": os.path.join(work, arm + "_val.pth"),
    "checkpoint": os.path.join("weights", weight + ".pth"),
    "optimizer": "Adam", "scheduler": "linear", "warmup": 0,
    "feature_name": "disentangled_self_attn", "text_features": "ann_feats",
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2)
PY
    echo "================ TRAIN ${arm} / seed=${seed} ================"
    python train.py \
      --model_config "$CONFIG" \
      --feature_name disentangled_self_attn \
      --text_features ann_feats \
      --train_dataset "$WORK_DIR/${arm}_train.pth" \
      --val_dataset "$WORK_DIR/${arm}_val.pth" \
      --optimizer Adam \
      --scheduler linear \
      --warmup 0 \
      --name_pedix "${arm}_seed${seed}" \
      --seed "$seed" \
      2>&1 | tee "$LOG_PATH"
    [[ -f "weights/${WEIGHT_NAME}.pth" ]] || {
      echo "[ERROR] Training did not create weights/${WEIGHT_NAME}.pth" >&2
      exit 1
    }
  done
done
echo "[DONE] Multi-seed training finished."
