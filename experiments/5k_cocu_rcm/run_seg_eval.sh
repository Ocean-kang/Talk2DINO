#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-123}"
EVAL_ARMS="${EVAL_ARMS:-raw cocu cocu_same cocu_rcm cocu_rcm_shuffled}"
BENCHMARKS="${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"
PAMR="${PAMR:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29617}"
if [[ "$PAMR" != "0" && "$PAMR" != "1" ]]; then echo "[ERROR] PAMR must be 0 or 1" >&2; exit 2; fi
read -r -a ARMS <<< "$EVAL_ARMS"
read -r -a SEED_LIST <<< "$SEEDS"
mkdir -p outputs/5k_cocu_rcm/segmentation outputs/5k_cocu_rcm/logs segmentation_results
SUFFIX=""; [[ "$PAMR" == "1" ]] && SUFFIX="_pamr"

for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    WEIGHT_NAME="vitb_mlp_infonce_5k_cocu_rcm_${arm}_seed${seed}"
    [[ -f "weights/${WEIGHT_NAME}.pth" ]] || { echo "[ERROR] Missing weights/${WEIGHT_NAME}.pth" >&2; exit 1; }
    for benchmark in $BENCHMARKS; do
      case "$benchmark" in
        voc20)
          MODEL_CFG="src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml"
          EVAL_CFG="src/open_vocabulary_segmentation/configs/voc/eval_voc${SUFFIX}.yml" ;;
        context59|pc59)
          MODEL_CFG="src/open_vocabulary_segmentation/configs/context/dinotext_context_vitb_mlp_infonce.yml"
          EVAL_CFG="src/open_vocabulary_segmentation/configs/context/eval_context${SUFFIX}.yml"
          benchmark="context59" ;;
        coco_stuff)
          MODEL_CFG="src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml"
          EVAL_CFG="src/open_vocabulary_segmentation/configs/stuff/eval_stuff${SUFFIX}.yml" ;;
        ade20k)
          MODEL_CFG="src/open_vocabulary_segmentation/configs/ade/dinotext_ade_vitb_mlp_infonce.yml"
          EVAL_CFG="src/open_vocabulary_segmentation/configs/ade/eval_ade${SUFFIX}.yml" ;;
        *) echo "[ERROR] Unsupported benchmark: $benchmark" >&2; exit 2 ;;
      esac
      OUT="outputs/5k_cocu_rcm/segmentation/${WEIGHT_NAME}/${benchmark}"
      LOG="outputs/5k_cocu_rcm/logs/seg_${arm}_seed${seed}_${benchmark}.log"
      mkdir -p "$OUT"
      python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
        src/open_vocabulary_segmentation/main.py \
        --eval --eval_cfg "$MODEL_CFG" --eval_base_cfg "$EVAL_CFG" \
        --output "$OUT" --opts "model.proj_name=${WEIGHT_NAME}" 2>&1 | tee "$LOG"
    done
  done
done

python experiments/5k_cocu_rcm/collect_results.py --root outputs/5k_cocu_rcm \
  --weights-dir weights --seg-results-dir segmentation_results \
  --arms "$EVAL_ARMS" --seeds "$SEEDS" --benchmarks "$BENCHMARKS"
