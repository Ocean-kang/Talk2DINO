#!/usr/bin/env bash
# Edit only these existing local paths. Run from repository root in current environment.
TRAIN_PTH="outputs/coco2014_b14/train.pth"
VAL_PTH="outputs/coco2014_b14/val.pth"
CAPTIONS_JSON="data/coco2014/annotations/captions_train2014.json"
INSTANCES_JSON="data/coco2014/annotations/instances_train2014.json"
IMAGE_ROOT="data/coco2014"
GALLERY="experiments/cocu_imagelevel_compare/MSCOCO14_train_gallery_exactmatch.json"
TEACHER="weights/vitb_mlp_infonce.pth"
CLIP_CHECKPOINT="${HOME}/.cache/clip/ViT-B-16.pt"
DINO_REPO="${HOME}/.cache/torch/hub/facebookresearch_dinov2_main"
DINO_CHECKPOINT="${HOME}/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"
CONFIG="configs/vitb_mlp_infonce.yaml"
OUT="outputs/ecca"
# After calibration, set this to outputs/ecca/calibration_report/selection_frozen.json.
SELECTION_CONFIG=""
# Use an existing file with one object/stuff concept per line to restrict semantic type.
CONCEPT_ALLOWLIST=""
SEEDS=(123 124 125 126 127)
