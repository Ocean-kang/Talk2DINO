#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/5k_ecsd}"
SEG_OUT="${SEG_OUT:-${OUT_DIR}/segmentation}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PAMR="${PAMR:-0}"
WEIGHT_TAG="${WEIGHT_TAG:-ecsd5k}"
EVAL_ARMS="${EVAL_ARMS:-raw rich rich_weaken rich_diversity ecsd}"
BENCHMARKS="${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"

mkdir -p "$SEG_OUT" "$OUT_DIR/logs" segmentation_results
if [[ "$PAMR" != "0" && "$PAMR" != "1" ]]; then
  echo "[ERROR] PAMR must be 0 or 1." >&2
  exit 1
fi

read -r -a ARMS <<< "$EVAL_ARMS"
for arm in "${ARMS[@]}"; do
  case "$arm" in raw|rich|rich_weaken|rich_diversity|ecsd) ;; *)
    echo "[ERROR] Unknown arm: $arm" >&2; exit 1 ;;
  esac
  WEIGHT_NAME="vitb_mlp_infonce_${WEIGHT_TAG}_${arm}"
  [[ -f "weights/${WEIGHT_NAME}.pth" ]] || {
    echo "[ERROR] Missing weights/${WEIGHT_NAME}.pth. Train this arm first." >&2
    exit 1
  }
done

run_one_eval() {
  local arm="$1"
  local benchmark="$2"
  local weight_name="vitb_mlp_infonce_${WEIGHT_TAG}_${arm}"
  local model_cfg=""
  local eval_cfg=""
  local suffix=""
  [[ "$PAMR" == "1" ]] && suffix="_pamr"

  case "$benchmark" in
    voc20)
      model_cfg="src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/voc/eval_voc${suffix}.yml"
      ;;
    context59|pc59)
      model_cfg="src/open_vocabulary_segmentation/configs/context/dinotext_context_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/context/eval_context${suffix}.yml"
      benchmark="context59"
      ;;
    coco_stuff)
      model_cfg="src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/stuff/eval_stuff${suffix}.yml"
      ;;
    ade20k)
      model_cfg="src/open_vocabulary_segmentation/configs/ade/dinotext_ade_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/ade/eval_ade${suffix}.yml"
      ;;
    *)
      echo "[ERROR] Unsupported benchmark: $benchmark" >&2
      return 2
      ;;
  esac

  [[ -f "$model_cfg" && -f "$eval_cfg" ]] || {
    echo "[ERROR] Missing eval config: $model_cfg or $eval_cfg" >&2
    return 1
  }

  local run_out="$SEG_OUT/$arm/$benchmark"
  local log_path="$OUT_DIR/logs/seg_${arm}_${benchmark}.log"
  mkdir -p "$run_out"
  echo "-------------------- SEG: $arm / $benchmark (PAMR=$PAMR) --------------------"

  python -m torch.distributed.run \
    --nproc_per_node="$NPROC_PER_NODE" \
    src/open_vocabulary_segmentation/main.py \
    --eval \
    --eval_cfg "$model_cfg" \
    --eval_base_cfg "$eval_cfg" \
    --output "$run_out" \
    --opts "model.proj_name=${weight_name}" \
    2>&1 | tee "$log_path"
}

for arm in "${ARMS[@]}"; do
  for benchmark in $BENCHMARKS; do
    run_one_eval "$arm" "$benchmark"
  done
done

python experiments/5k_ecsd/collect_results.py \
  --root "$OUT_DIR" \
  --weights-dir weights \
  --seg-results-dir segmentation_results \
  --weight-tag "$WEIGHT_TAG" \
  --arms "$EVAL_ARMS" \
  --benchmarks "$BENCHMARKS"

echo "[DONE] ECSD segmentation evaluation completed."
