#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT_ROOT="${OUT_ROOT:-outputs/ecsd_scale_sensitivity}"
SEG_OUT="${SEG_OUT:-$OUT_ROOT/segmentation}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEEDS="${SEEDS:-123 456 789}"
SCALES="${SCALES:-5k 10k 20k 40k full}"
ARMS="${ARMS:-raw ecsd}"
BENCHMARKS="${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"
mkdir -p "$SEG_OUT" "$OUT_ROOT/logs" segmentation_results

run_one() {
  local scale="$1" seed="$2" arm="$3" benchmark="$4"
  local suffix="scale${scale}_s${seed}_${arm}"
  local weight="vitb_mlp_infonce_${suffix}"
  local model_cfg="" eval_cfg=""
  case "$benchmark" in
    voc20)
      model_cfg="src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/voc/eval_voc.yml" ;;
    context59|pc59)
      model_cfg="src/open_vocabulary_segmentation/configs/context/dinotext_context_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/context/eval_context.yml"
      benchmark="context59" ;;
    coco_stuff)
      model_cfg="src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml" ;;
    ade20k)
      model_cfg="src/open_vocabulary_segmentation/configs/ade/dinotext_ade_vitb_mlp_infonce.yml"
      eval_cfg="src/open_vocabulary_segmentation/configs/ade/eval_ade.yml" ;;
    *) echo "[ERROR] Unsupported benchmark $benchmark" >&2; return 2 ;;
  esac
  [[ -f "$model_cfg" && -f "$eval_cfg" ]] || { echo "[ERROR] Missing eval config for $benchmark" >&2; return 1; }
  [[ -f "weights/${weight}.pth" ]] || { echo "[ERROR] Missing weights/${weight}.pth" >&2; return 1; }
  local run_out="$SEG_OUT/$scale/seed${seed}/${arm}/${benchmark}"
  local log="$OUT_ROOT/logs/seg_${scale}_s${seed}_${arm}_${benchmark}.log"
  mkdir -p "$run_out"
  echo "[EVAL] scale=$scale seed=$seed arm=$arm benchmark=$benchmark"
  python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
    src/open_vocabulary_segmentation/main.py --eval \
    --eval_cfg "$model_cfg" --eval_base_cfg "$eval_cfg" \
    --output "$run_out" --opts "model.proj_name=${weight}" \
    2>&1 | tee "$log"
}
for seed in $SEEDS; do
  for scale in $SCALES; do
    for arm in $ARMS; do
      for benchmark in $BENCHMARKS; do
        run_one "$scale" "$seed" "$arm" "$benchmark"
      done
    done
  done
done
echo "[DONE] OVSS evaluation finished."
