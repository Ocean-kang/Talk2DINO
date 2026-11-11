#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
DATA_DIR="${DATA_DIR:-data/coco2014_subset_50k}"
CAPTIONS="${CAPTIONS:-data/coco2014_subset_50k/5k_rich_diversity_strengthen.json}"
OUT_DIR="${OUT_DIR:-outputs/5k_rich_diversity_strengthen}"
WORK_DIR="${WORK_DIR:-$OUT_DIR/work}"
DINO="${DINO_SOURCE:-$WORK_DIR/dino.pth}"
read -r -a ARMS <<< "${PREP_ARMS:-rich_diversity_strengthen}"

check=(--captions "$CAPTIONS")
if [[ -n "${PREVIOUS_CAPTIONS:-}" ]]; then
  check+=(--previous-captions "$PREVIOUS_CAPTIONS")
fi
if [[ -n "${PREVIOUS_VAL_IDS:-}" ]]; then
  check+=(--previous-val-ids "$PREVIOUS_VAL_IDS")
fi
python "$HERE/prepare.py" check "${check[@]}"
mkdir -p "$WORK_DIR"
rm -f -- "$WORK_DIR/audit_report.json"

if [[ -z "${DINO_SOURCE:-}" ]]; then
  python "$HERE/prepare.py" init --captions "$CAPTIONS" --images "$DATA_DIR" --work "$WORK_DIR"
  if [[ ! -f "$DINO" ]]; then
    python dino_extraction_v2.py \
      --ann_path "$WORK_DIR/base.pth" --data_dir "$WORK_DIR/images" \
      --out_path "$DINO.tmp" --model dinov2_vitb14_reg \
      --resize_dim 448 --crop_dim 448 --batch_size "${DINO_BATCH_SIZE:-8}" \
      --extract_avg_self_attn --extract_disentangled_self_attn
    mv -- "$DINO.tmp" "$DINO"
  fi
fi

args=(--captions "$CAPTIONS" --work "$WORK_DIR" --dino "$DINO" --arms "${ARMS[@]}")
python "$HERE/prepare.py" assemble "${args[@]}"
for arm in "${ARMS[@]}"; do
  for split in train val; do
    prefix="$WORK_DIR/${arm}_${split}"
    python text_features_extraction.py \
      --ann_path "$prefix.input.pth" --out_path "$prefix.pth.tmp" \
      --model ViT-B/16 --batch_size "${CLIP_BATCH_SIZE:-256}"
    mv -- "$prefix.pth.tmp" "$prefix.pth"
    rm -- "$prefix.input.pth"
  done
done
python "$HERE/prepare.py" audit "${args[@]}"
