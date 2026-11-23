#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ECSD_CONFIG="${ECSD_CONFIG:-experiments/5k_ecsd_next/configs/default.json}"
CFG_DATA_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["data_dir"])' "$ECSD_CONFIG")"
CFG_CAPTIONS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["ecsd_v2_captions"])' "$ECSD_CONFIG")"
CFG_OUT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$ECSD_CONFIG")/multiseed"
CFG_DINO="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["dino_source"])' "$ECSD_CONFIG")"
CFG_CLIP="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["clip_cache"])' "$ECSD_CONFIG")"
CFG_IMAGES="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["expected_images"])' "$ECSD_CONFIG")"
CFG_VAL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["val_images"])' "$ECSD_CONFIG")"
CFG_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["split_seed"])' "$ECSD_CONFIG")"
DATA_DIR="${DATA_DIR:-$CFG_DATA_DIR}"
CAPTIONS="${CAPTIONS:-$CFG_CAPTIONS}"
IMAGES="${IMAGES:-$DATA_DIR}"
OUT_DIR="${OUT_DIR:-$CFG_OUT}"
WORK_DIR="${WORK_DIR:-${OUT_DIR}/work}"
DINO_SOURCE="${DINO_SOURCE:-$CFG_DINO}"
CFG_CLIP_EXPANDED="${CFG_CLIP/#\~/$HOME}"
CLIP_WEIGHT="${CLIP_WEIGHT:-${CFG_CLIP_EXPANDED}/ViT-B-16.pt}"
DEFAULT_CLIP_WEIGHT="$HOME/.cache/clip/ViT-B-16.pt"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-256}"

[[ -f "$CAPTIONS" ]] || { echo "[ERROR] Missing ECSD v2 captions: $CAPTIONS" >&2; exit 1; }
[[ -f "$CLIP_WEIGHT" ]] || {
  echo "[ERROR] Existing CLIP ViT-B/16 cache not found: $CLIP_WEIGHT" >&2
  echo "This experiment does not download weights. Configure the path to an existing ViT-B-16.pt." >&2
  exit 1
}
# Upstream text_features_extraction.py hardcodes ~/.cache/clip. Link the configured local weight there if needed.
if [[ "$CLIP_WEIGHT" != "$DEFAULT_CLIP_WEIGHT" ]]; then
  mkdir -p "$HOME/.cache/clip"
  if [[ -e "$DEFAULT_CLIP_WEIGHT" ]]; then
    [[ "$(realpath "$DEFAULT_CLIP_WEIGHT")" == "$(realpath "$CLIP_WEIGHT")" ]] || {
      echo "[ERROR] $DEFAULT_CLIP_WEIGHT already points to a different CLIP weight; refusing to replace it." >&2
      exit 1
    }
  else
    ln -s "$(realpath "$CLIP_WEIGHT")" "$DEFAULT_CLIP_WEIGHT"
  fi
fi
if [[ ! -f "$DINO_SOURCE" ]]; then
  if [[ -f outputs/5k_ecsd/work/dino.pth ]]; then
    DINO_SOURCE="outputs/5k_ecsd/work/dino.pth"
  elif [[ -f outputs/5k_rich_diversity/work/dino.pth ]]; then
    DINO_SOURCE="outputs/5k_rich_diversity/work/dino.pth"
  else
    echo "[ERROR] No existing 5K DINO feature file found." >&2
    echo "Set DINO_SOURCE to a compatible dino.pth. No model download/extraction is performed." >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/config_snapshot"
python experiments/5k_ecsd_next/multiseed/prepare_multiseed.py init \
  --captions "$CAPTIONS" --images "$IMAGES" --work "$WORK_DIR" \
  --dino-source "$DINO_SOURCE" --expected-images "$CFG_IMAGES" \
  --split-seed "$CFG_SPLIT" --val-images "$CFG_VAL"

ASSEMBLE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then ASSEMBLE_ARGS+=(--overwrite); fi
python experiments/5k_ecsd_next/multiseed/prepare_multiseed.py assemble \
  --captions "$CAPTIONS" --work "$WORK_DIR" \
  --expected-images "$CFG_IMAGES" --split-seed "$CFG_SPLIT" --val-images "$CFG_VAL" \
  "${ASSEMBLE_ARGS[@]}"

for arm in raw rich rich_diversity ecsdv2; do
  for split in train val; do
    FILE="$WORK_DIR/${arm}_${split}.pth"
    python text_features_extraction.py \
      --ann_path "$FILE" --out_path "$FILE" \
      --model ViT-B/16 --batch_size "$CLIP_BATCH_SIZE" \
      2>&1 | tee "$OUT_DIR/logs/clip_${arm}_${split}.log"
  done
done

python experiments/5k_ecsd_next/multiseed/prepare_multiseed.py audit \
  --work "$WORK_DIR" --expected-images "$CFG_IMAGES" --val-images "$CFG_VAL"

if [[ "${RUN_TRAIN:-1}" == "1" ]]; then
  bash experiments/5k_ecsd_next/multiseed/run_train.sh
fi
if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  bash experiments/5k_ecsd_next/multiseed/run_seg_eval.sh
  python experiments/5k_ecsd_next/multiseed/aggregate_results.py \
    --root "$OUT_DIR" --weights-dir weights \
    --seg-results-dir segmentation_results \
    --metrics-summary outputs/5k_ecsd_next/ablation_metrics/summary.csv
fi

echo "[DONE] Stage C workflow completed."
