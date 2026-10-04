#!/usr/bin/env bash

set -e

ROOT_DIR="/home/master/code/oymk/Talking2dino/Talk2DINO"
CUDA_ID=1

cd "$ROOT_DIR"

WEIGHTS=(
    "weights/bootstrap_teacher/a0/vitb_mlp_infonce_teacher_bootstrap_a0.pth"
    "weights/bootstrap_teacher/a0p1/vitb_mlp_infonce_teacher_bootstrap_a0p1.pth"
    "weights/bootstrap_teacher/a0p5/vitb_mlp_infonce_teacher_bootstrap_a0p3.pth"
    "weights/bootstrap_teacher/a0p5/vitb_mlp_infonce_teacher_bootstrap_a0p5.pth"
    "weights/bootstrap_teacher/a0p5/vitb_mlp_infonce_teacher_bootstrap_a0p7.pth"
    "weights/bootstrap_teacher/a1/vitb_mlp_infonce_teacher_bootstrap_a1.pth"
)

for WEIGHT in "${WEIGHTS[@]}"
do

    # --------------------------------------------------------
    # weights/bootstrap_teacher/a0/xxx.pth
    # ->
    # bootstrap_teacher/a0/xxx
    # --------------------------------------------------------

    PROJ_NAME="${WEIGHT#weights/}"
    PROJ_NAME="${PROJ_NAME%.pth}"

    # 仅用于普通 output 文件夹名称
    PROJ=$(basename "$WEIGHT" .pth)

    # log_results() 不会自动创建 proj_name 中间目录
    mkdir -p "segmentation_results/$(dirname "$PROJ_NAME")"

    echo
    echo "============================================================"
    echo "Evaluating projector"
    echo "Weight    : $WEIGHT"
    echo "proj_name : $PROJ_NAME"
    echo "GPU       : $CUDA_ID"
    echo "============================================================"
    echo


    # ========================================================
    # 1. VOC20
    # ========================================================

    echo "========== VOC20 =========="

    CUDA_VISIBLE_DEVICES=$CUDA_ID \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29501 \
    RANK=0 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    LOCAL_WORLD_SIZE=1 \
    python src/open_vocabulary_segmentation/main.py \
        --eval \
        --eval_cfg \
        src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml \
        --eval_base_cfg \
        src/open_vocabulary_segmentation/configs/voc/eval_voc_pamr.yml \
        --output \
        "output/eval/bootstrap_teacher/${PROJ}/voc20" \
        --opts \
        model.proj_name="$PROJ_NAME"


    # ========================================================
    # 2. Context59
    # ========================================================

    echo
    echo "========== Context59 =========="

    CUDA_VISIBLE_DEVICES=$CUDA_ID \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29502 \
    RANK=0 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    LOCAL_WORLD_SIZE=1 \
    python src/open_vocabulary_segmentation/main.py \
        --eval \
        --eval_cfg \
        src/open_vocabulary_segmentation/configs/context/dinotext_context_vitb_mlp_infonce.yml \
        --eval_base_cfg \
        src/open_vocabulary_segmentation/configs/context/eval_context_pamr.yml \
        --output \
        "output/eval/bootstrap_teacher/${PROJ}/context59" \
        --opts \
        model.proj_name="$PROJ_NAME"


    # ========================================================
    # 3. ADE20K
    # ========================================================

    echo
    echo "========== ADE20K =========="

    CUDA_VISIBLE_DEVICES=$CUDA_ID \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29503 \
    RANK=0 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    LOCAL_WORLD_SIZE=1 \
    python src/open_vocabulary_segmentation/main.py \
        --eval \
        --eval_cfg \
        src/open_vocabulary_segmentation/configs/ade/dinotext_ade_vitb_mlp_infonce.yml \
        --eval_base_cfg \
        src/open_vocabulary_segmentation/configs/ade/eval_ade_pamr.yml \
        --output \
        "output/eval/bootstrap_teacher/${PROJ}/ade20k" \
        --opts \
        model.proj_name="$PROJ_NAME"


    # ========================================================
    # 4. COCO-Stuff
    # ========================================================

    echo
    echo "========== COCO-Stuff =========="

    CUDA_VISIBLE_DEVICES=$CUDA_ID \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29504 \
    RANK=0 \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    LOCAL_WORLD_SIZE=1 \
    python src/open_vocabulary_segmentation/main.py \
        --eval \
        --eval_cfg \
        src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml \
        --eval_base_cfg \
        src/open_vocabulary_segmentation/configs/stuff/eval_stuff_pamr.yml \
        --output \
        "output/eval/bootstrap_teacher/${PROJ}/coco_stuff" \
        --opts \
        model.proj_name="$PROJ_NAME"


    echo
    echo "============================================================"
    echo "Finished: $PROJ"
    echo "============================================================"

done

echo
echo "============================================================"
echo "All evaluations finished."
echo "============================================================"