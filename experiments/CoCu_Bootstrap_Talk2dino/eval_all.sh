#!/usr/bin/env bash
set -euo pipefail
# Run from repository root.
# Usage: bash experiments/CoCu_Bootstrap_Talk2dino/eval_all.sh [GPU] [one_arm]
# Example: bash experiments/CoCu_Bootstrap_Talk2dino/eval_all.sh 1 clip_B
GPU="${1:-1}"
SELECT="${2:-all}"
ARMS=(clip_5plus3 clip_B clip_D local_5plus3 local_B local_D)
CFG_ROOT="src/open_vocabulary_segmentation/configs"
for ARM in "${ARMS[@]}"; do
  [[ "$SELECT" == "all" || "$SELECT" == "$ARM" ]] || continue
  PROJ_NAME="vitb_mlp_infonce_cocu_bt_${ARM}_a1"
  WEIGHT="weights/${PROJ_NAME}.pth"
  [[ -f "$WEIGHT" ]] || { echo "Missing: $WEIGHT" >&2; exit 1; }
  mkdir -p segmentation_results
  TASKS=(voc20 context59 ade20k coco_stuff)
  CONFIGS=(
    "$CFG_ROOT/voc/dinotext_voc_vitb_mlp_infonce.yml"
    "$CFG_ROOT/context/dinotext_context_vitb_mlp_infonce.yml"
    "$CFG_ROOT/ade/dinotext_ade_vitb_mlp_infonce.yml"
    "$CFG_ROOT/stuff/dinotext_stuff_vitb_mlp_infonce.yml"
  )
  BASES=(
    "$CFG_ROOT/voc/eval_voc_pamr.yml"
    "$CFG_ROOT/context/eval_context_pamr.yml"
    "$CFG_ROOT/ade/eval_ade_pamr.yml"
    "$CFG_ROOT/stuff/eval_stuff_pamr.yml"
  )
  for i in "${!TASKS[@]}"; do
    PORT=$((29501+i))
    echo "[EVAL] arm=$ARM dataset=${TASKS[$i]} proj=$PROJ_NAME"
    CUDA_VISIBLE_DEVICES="$GPU" MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    python src/open_vocabulary_segmentation/main.py \
      --eval --eval_cfg "${CONFIGS[$i]}" --eval_base_cfg "${BASES[$i]}" \
      --output "output/eval/CoCu_Bootstrap_Talk2dino/${PROJ_NAME}/${TASKS[$i]}" \
      --opts "model.proj_name=${PROJ_NAME}"
  done
done
