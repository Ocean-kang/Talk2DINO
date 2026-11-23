#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT_DIR="${OUT_DIR:-outputs/full_mscoco14_ecsd}"
SEG_OUT="${SEG_OUT:-${OUT_DIR}/segmentation}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PAMR="${PAMR:-0}"
SEEDS="${SEEDS:-123}"
EVAL_ARMS="${EVAL_ARMS:-raw ecsd}"
BENCHMARKS="${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"
mkdir -p "$SEG_OUT" "$OUT_DIR/logs" segmentation_results
[[ "$PAMR" == "0" || "$PAMR" == "1" ]] || { echo "[ERROR] PAMR must be 0 or 1" >&2; exit 1; }

run_one_eval() {
  local seed="$1" arm="$2" benchmark="$3"
  local suffix="full14_s${seed}_${arm}"
  local weight="vitb_mlp_infonce_${suffix}"
  local model_cfg="" eval_cfg="" eval_suffix=""
  [[ "$PAMR" == "1" ]] && eval_suffix="_pamr"
  case "$benchmark" in
    voc20)
      model_cfg="src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/voc/eval_voc${eval_suffix}.yml" ;;
    context59|pc59)
      model_cfg="src/open_vocabulary_segmentation/configs/context/dinotext_context_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/context/eval_context${eval_suffix}.yml"
      benchmark="context59" ;;
    coco_stuff)
      model_cfg="src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/stuff/eval_stuff${eval_suffix}.yml" ;;
    ade20k)
      model_cfg="src/open_vocabulary_segmentation/configs/ade/dinotext_ade_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/ade/eval_ade${eval_suffix}.yml" ;;
    *) echo "[ERROR] Unsupported benchmark: $benchmark" >&2; return 2 ;;
  esac
  [[ -f "$model_cfg" && -f "$eval_cfg" ]] || { echo "[ERROR] Missing eval config $model_cfg or $eval_cfg" >&2; return 1; }
  [[ -f "weights/${weight}.pth" ]] || { echo "[ERROR] Missing weights/${weight}.pth" >&2; return 1; }
  local run_out="$SEG_OUT/seed${seed}/${arm}/${benchmark}"
  local log="$OUT_DIR/logs/seg_s${seed}_${arm}_${benchmark}.log"
  mkdir -p "$run_out"
  echo "[EVAL] seed=$seed arm=$arm benchmark=$benchmark PAMR=$PAMR"
  python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    src/open_vocabulary_segmentation/main.py --eval \
    --eval_cfg "$model_cfg" --eval_base_cfg "$eval_cfg" \
    --output "$run_out" --opts "model.proj_name=${weight}" \
    2>&1 | tee "$log"
}

for seed in $SEEDS; do
  for arm in $EVAL_ARMS; do
    for benchmark in $BENCHMARKS; do run_one_eval "$seed" "$arm" "$benchmark"; done
  done
done
python experiments/full_mscoco14_ecsd/collect_results.py \
  --root "$OUT_DIR" --weights-dir weights --seg-results-dir segmentation_results \
  --seeds "$SEEDS" --arms "$EVAL_ARMS" --benchmarks "$BENCHMARKS"
echo "[DONE] Full MSCOCO14 Raw/ECSD segmentation evaluation completed."
