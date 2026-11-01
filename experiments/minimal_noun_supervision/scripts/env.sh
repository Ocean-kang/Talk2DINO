#!/usr/bin/env bash
MNS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$MNS_ROOT"
MNS_DIR=experiments/minimal_noun_supervision
export PYTHONPATH="$MNS_ROOT/$MNS_DIR/offline_guard:$MNS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TRAIN_PTH="${TRAIN_PTH:-}"
VAL_PTH="${VAL_PTH:-}"
MNS_GALLERY="${MNS_GALLERY:-}"
MNS_CLIP_CHECKPOINT="${MNS_CLIP_CHECKPOINT:-}"
MNS_DINO_CHECKPOINT="${MNS_DINO_CHECKPOINT:-}"
MNS_FEATURE_METADATA="${MNS_FEATURE_METADATA:-}"
MNS_DATA="${MNS_DATA:-outputs/minimal_noun_supervision/data_gallery}"
MNS_CONFIG="${MNS_CONFIG:-configs/vitb_mlp_infonce.yaml}"
MNS_TEACHER="${MNS_TEACHER:-}"
MNS_TAG="${MNS_TAG:-full_gallery}"
