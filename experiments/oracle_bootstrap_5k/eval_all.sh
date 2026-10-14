#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/oracle_bootstrap_5k/eval_all.sh [GPU] [RUN_NAME|all] [DATASET|all]
# Examples:
#   bash experiments/oracle_bootstrap_5k/eval_all.sh 1
#   bash experiments/oracle_bootstrap_5k/eval_all.sh 1 rich_a1
#   bash experiments/oracle_bootstrap_5k/eval_all.sh 1 rich_a1 ade20k
# PORT_BASE=29600 bash ...   # if ports collide
# DRY_RUN=1 bash ...     # print commands only

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
cd "$ROOT"
GPU="${1:-1}"
SELECT="${2:-all}"
DATASET="${3:-all}"
PORT_BASE="${PORT_BASE:-29500}"
CFG_ROOT="src/open_vocabulary_segmentation/configs"

if [[ "$SELECT" == all ]]; then
  ARMS=(raw_a0 rich_a0 raw_a1 rich_a1)
else
  if [[ ! "$SELECT" =~ ^(raw|rich)_a[0-9]+(p[0-9]+)?$ ]]; then
    echo "ERROR: invalid run name: $SELECT" >&2
    exit 2
  fi
  ARMS=("$SELECT")
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

if [[ "$DATASET" != all ]]; then
  found=0
  for task in "${TASKS[@]}"; do
    if [[ "$DATASET" == "$task" ]]; then found=1; fi
  done
  if [[ "$found" != 1 ]]; then
    echo "ERROR: dataset must be all, voc20, context59, ade20k or coco_stuff" >&2
    exit 2
  fi
fi

for ARM in "${ARMS[@]}"; do
  PROJ_NAME="oracle_bootstrap_5k/${ARM}"
  WEIGHT="weights/${PROJ_NAME}.pth"
  if [[ "${DRY_RUN:-0}" != 1 && ! -f "$WEIGHT" ]]; then
    echo "ERROR: missing weight: $ROOT/$WEIGHT" >&2
    exit 1
  fi

  # eval logs use model.proj_name, which contains '/'. Ensure parent exists.
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "segmentation_results/oracle_bootstrap_5k"
  fi

  for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    [[ "$DATASET" == all || "$DATASET" == "$TASK" ]] || continue
    PORT=$((PORT_BASE+i+1))
    CMD=(python src/open_vocabulary_segmentation/main.py
      --eval
      --eval_cfg "${CONFIGS[$i]}"
      --eval_base_cfg "${BASES[$i]}"
      --output "output/eval/oracle_bootstrap_5k/${ARM}/${TASK}"
      --opts "model.proj_name=${PROJ_NAME}")
    echo "[EVAL] arm=$ARM dataset=$TASK gpu=$GPU proj=$PROJ_NAME port=$PORT"
    printf '[CMD] CUDA_VISIBLE_DEVICES=%q MASTER_ADDR=127.0.0.1 MASTER_PORT=%q RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 ' "$GPU" "$PORT"
    printf '%q ' "${CMD[@]}"
    printf '\n'

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      continue
    fi
    for f in "${CONFIGS[$i]}" "${BASES[$i]}"; do
      if [[ ! -f "$f" ]]; then
        echo "ERROR: missing eval config: $ROOT/$f" >&2
        exit 1
      fi
    done
    CUDA_VISIBLE_DEVICES="$GPU" MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
      "${CMD[@]}"
  done
done
