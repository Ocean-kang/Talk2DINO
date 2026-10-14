#!/usr/bin/env bash
set -euo pipefail
# bash experiments/local_cocu_novel/eval.sh [GPU]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
GPU="${1:-0}"
PROJ="vitb_mlp_infonce_local_cocu_novel_5plus3"
[[ -f "weights/$PROJ.pth" ]] || { echo "Missing weights/$PROJ.pth" >&2; exit 1; }
CFG="src/open_vocabulary_segmentation/configs"
TASKS=(voc20 context59 ade20k coco_stuff)
CONFIGS=(
  "$CFG/voc/dinotext_voc_vitb_mlp_infonce.yml"
  "$CFG/context/dinotext_context_vitb_mlp_infonce.yml"
  "$CFG/ade/dinotext_ade_vitb_mlp_infonce.yml"
  "$CFG/stuff/dinotext_stuff_vitb_mlp_infonce.yml"
)
BASES=(
  "$CFG/voc/eval_voc_pamr.yml"
  "$CFG/context/eval_context_pamr.yml"
  "$CFG/ade/eval_ade_pamr.yml"
  "$CFG/stuff/eval_stuff_pamr.yml"
)
mkdir -p segmentation_results
for i in "${!TASKS[@]}"; do
  PORT=$((29501+i))
  echo "[EVAL] $PROJ -> ${TASKS[$i]} GPU=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
  python src/open_vocabulary_segmentation/main.py \
    --eval --eval_cfg "${CONFIGS[$i]}" --eval_base_cfg "${BASES[$i]}" \
    --output "output/eval/local_cocu_novel/$PROJ/${TASKS[$i]}" \
    --opts "model.proj_name=$PROJ"
done
