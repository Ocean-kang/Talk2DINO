#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
OUT_DIR="${OUT_DIR:-outputs/5k_rich_diversity_strengthen}"
WORK_DIR="${WORK_DIR:-$OUT_DIR/work}"
WEIGHT_TAG="${WEIGHT_TAG:-subset5k_strengthen}"
CONFIG=configs/vitb_mlp_infonce.yaml
read -r -a ARMS <<< "${TRAIN_ARMS:-rich_diversity_strengthen}"

[[ "$WEIGHT_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid WEIGHT_TAG" >&2; exit 2; }
python - "$WORK_DIR" "${ARMS[@]}" <<'PY'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
report_path = work / "audit_report.json"
if not report_path.is_file():
    sys.exit("Run run_preprocess.sh successfully before training")
report = json.loads(report_path.read_text(encoding="utf-8"))
arms = sys.argv[2:]
if not arms or len(arms) != len(set(arms)):
    sys.exit("Specify distinct TRAIN_ARMS")
if report["status"] != "ok" or not set(arms).issubset(report["arms"]):
    sys.exit("The requested arms must pass preprocessing audit first")
for arm in arms:
    for split in ("train", "val"):
        path = work / f"{arm}_{split}.pth"
        if not path.is_file():
            sys.exit(f"Missing feature file: {path}")
PY

for arm in "${ARMS[@]}"; do
  weight="weights/vitb_mlp_infonce_${WEIGHT_TAG}_${arm}.pth"
  if [[ -e "$weight" || -L "$weight" ]]; then
    echo "Checkpoint already exists: $weight. Choose a fresh WEIGHT_TAG or select pending arms." >&2
    exit 2
  fi
done

LOG_DIR="$OUT_DIR/logs/$WEIGHT_TAG"
mkdir -p "$LOG_DIR" weights
for arm in "${ARMS[@]}"; do
  name="${WEIGHT_TAG}_${arm}"
  snapshot="$OUT_DIR/config_snapshot/$WEIGHT_TAG/$arm"
  mkdir -p "$snapshot"
  cp -- "$CONFIG" "$WORK_DIR/audit_report.json" "$WORK_DIR/val_ids.json" "$snapshot/"
  python -u train.py \
    --model_config "$CONFIG" --feature_name disentangled_self_attn --text_features ann_feats \
    --train_dataset "$WORK_DIR/${arm}_train.pth" --val_dataset "$WORK_DIR/${arm}_val.pth" \
    --optimizer Adam --scheduler linear --warmup 0 --name_pedix "$name" \
    2>&1 | tee "$LOG_DIR/train_$arm.log"
  test -s "weights/vitb_mlp_infonce_$name.pth"
done
