#!/usr/bin/env bash
set -euo pipefail

# Run from Talk2DINO repository root after full Student training.
GPU="${1:-0}"
SEED="${2:-123}"
PROJ_NAME="vitb_mlp_infonce_p1_bootstrap_seed${SEED}"
STUDENT_WEIGHT="outputs/p1_concept_propagation_bootstrap/student_seed${SEED}/student_projector.pth"
TARGET_WEIGHT="weights/${PROJ_NAME}.pth"
CFG_ROOT="src/open_vocabulary_segmentation/configs"

[[ -f "$STUDENT_WEIGHT" ]] || {
  echo "Missing Student checkpoint: $STUDENT_WEIGHT" >&2
  exit 1
}

# DINOText loads projector weights from weights/{model.proj_name}.pth.
# Never overwrite a different existing checkpoint.
if [[ -e "$TARGET_WEIGHT" ]]; then
  if ! cmp -s "$STUDENT_WEIGHT" "$TARGET_WEIGHT"; then
    echo "Refusing to overwrite different existing weight: $TARGET_WEIGHT" >&2
    exit 1
  fi
else
  cp "$STUDENT_WEIGHT" "$TARGET_WEIGHT"
fi

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
  PORT=$((39601+i))
  echo "[EVAL] P1 Bootstrap ${TASKS[$i]} -> ${PROJ_NAME}"
  CUDA_VISIBLE_DEVICES="$GPU" MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
  python src/open_vocabulary_segmentation/main.py \
    --eval \
    --eval_cfg "${CONFIGS[$i]}" \
    --eval_base_cfg "${BASES[$i]}" \
    --output "output/eval/P1_Concept_Propagation_Bootstrap/${PROJ_NAME}/${TASKS[$i]}" \
    --opts "model.proj_name=${PROJ_NAME}"
done
