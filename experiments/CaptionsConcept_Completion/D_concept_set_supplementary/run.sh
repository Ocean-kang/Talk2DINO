#!/usr/bin/env bash
set -euo pipefail
# Run from Talk2DINO repository root.
ACTION="${1:-}"
SOURCE="${2:-}"
GPU="${3:-0}"
BASE="outputs/coco2014_b14/train.pth"
VAL="outputs/coco2014_b14/val.pth"
CFG="configs/vitb_mlp_infonce.yaml"
EXP="experiments/CaptionsConcept_Completion/D_concept_set_supplementary"
OUT="outputs/CaptionsConcept_Completion/D"
mkdir -p "$OUT"
case "$SOURCE" in
  clip) SRCARG="clip"; MANIFEST="$EXP/clip/concepts_manifest.json"; PTH="$OUT/train_clip.pth"; SUFFIX="ccD_clip" ;;
  local|local_talk2dino) SRCARG="local_talk2dino"; MANIFEST="$EXP/local_talk2dino/concepts_manifest.json"; PTH="$OUT/train_local_talk2dino.pth"; SUFFIX="ccD_local_talk2dino" ;;
  *) echo "source must be: clip | local"; exit 1 ;;
esac
case "$ACTION" in
  build)
    CUDA_VISIBLE_DEVICES="$GPU" python "$EXP/build.py" \
      --source "$SRCARG" \
      --manifest "$MANIFEST" --base-train "$BASE" --output "$PTH" \
      --audit "$OUT/${SUFFIX}_audit.jsonl" --summary "$OUT/${SUFFIX}_summary.json" \
      --clip-model "ViT-B/16" --batch-size 512 --image-batch-size 256 --device cuda \
      --min-visual-ratio 0.85 --max-concepts 3
    ;;
  train)
    CUDA_VISIBLE_DEVICES="$GPU" python train.py \
      --model_config "$CFG" --train_dataset "$PTH" --val_dataset "$VAL" \
      --feature_name disentangled_self_attn --text_features ann_feats --name_pedix "$SUFFIX"
    ;;
  *) echo "Usage: $0 {build|train} {clip|local} [gpu]"; exit 1 ;;
esac
