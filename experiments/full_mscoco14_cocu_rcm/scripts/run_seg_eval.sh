#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/full_mscoco14_cocu_rcm}"
SEG_OUT="${SEG_OUT:-${OUT_DIR}/segmentation}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PAMR="${PAMR:-0}"
SEG_RESULTS_DIR="${SEG_RESULTS_DIR:-}"
if [[ -z "$SEG_RESULTS_DIR" ]]; then
  if [[ "$PAMR" == "1" ]]; then SEG_RESULTS_DIR="${OUT_DIR}/segmentation_results_pamr"; else SEG_RESULTS_DIR="${OUT_DIR}/segmentation_results"; fi
fi
SEEDS="${SEEDS:-123}"
EVAL_ARMS="${EVAL_ARMS:-raw cocu cocu_same cocu_rcm cocu_rcm_shuffled}"
BENCHMARKS="${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"
MASTER_PORT="${MASTER_PORT:-29627}"
mkdir -p "$SEG_OUT" "$OUT_DIR/logs" "$SEG_RESULTS_DIR" segmentation_results
[[ "$PAMR" == "0" || "$PAMR" == "1" ]] || { echo "[ERROR] PAMR must be 0 or 1" >&2; exit 1; }
EVAL_STARTED="$(date +%s)"
# Do not let the collector silently reuse metrics from a previous evaluation.
for seed in $SEEDS; do
  for arm in $EVAL_ARMS; do
    stale_name="vitb_mlp_infonce_full_mscoco14_cocu_rcm_${arm}_seed${seed}.json"
    rm -f "$SEG_RESULTS_DIR/$stale_name"
  done
done

for seed in $SEEDS; do
  for arm in $EVAL_ARMS; do
    suffix="full_mscoco14_cocu_rcm_${arm}_seed${seed}"
    weight="vitb_mlp_infonce_${suffix}"
    [[ -f "weights/${weight}.pth" ]] || { echo "[ERROR] Missing weights/${weight}.pth" >&2; exit 1; }
    for benchmark in $BENCHMARKS; do
      eval_suffix=""; [[ "$PAMR" == "1" ]] && eval_suffix="_pamr"
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
        *) echo "[ERROR] Unsupported benchmark: $benchmark" >&2; exit 2 ;;
      esac
      [[ -f "$model_cfg" && -f "$eval_cfg" ]] || { echo "[ERROR] Missing eval config for $benchmark" >&2; exit 1; }
      run_out="$SEG_OUT/seed${seed}/${arm}/${benchmark}"
      log="$OUT_DIR/logs/seg_${arm}_seed${seed}_${benchmark}.log"
      mkdir -p "$run_out"
      python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" src/open_vocabulary_segmentation/main.py \
        --eval --eval_cfg "$model_cfg" --eval_base_cfg "$eval_cfg" \
        --output "$run_out" --opts "model.proj_name=${weight}" \
        2>&1 | tee "$log"
      python experiments/full_mscoco14_cocu_rcm/evaluation/write_manifest.py \
        --output "$run_out/evaluation_manifest.json" --arm "$arm" --seed "$seed" \
        --benchmark "$benchmark" --checkpoint "weights/${weight}.pth" \
        --model-config "$model_cfg" --eval-config "$eval_cfg" --pamr "$PAMR" \
        --nproc-per-node "$NPROC_PER_NODE" --log "$log"
    done
  done
done

# The upstream evaluator writes to the repository-level segmentation_results/.
# Copy its finished JSON into the requested isolated collector directory.
if [[ "$SEG_RESULTS_DIR" != "segmentation_results" ]]; then
  for seed in $SEEDS; do
    for arm in $EVAL_ARMS; do
      name="vitb_mlp_infonce_full_mscoco14_cocu_rcm_${arm}_seed${seed}"
      result="segmentation_results/${name}.json"
      if [[ -f "$result" ]]; then
        modified="$(stat -c %Y "$result")"
        if (( modified >= EVAL_STARTED )); then
          cp "$result" "$SEG_RESULTS_DIR/${name}.json"
        else
          echo "[WARN] Result JSON is stale (not modified in this evaluation): $result" >&2
        fi
      else
        echo "[WARN] Upstream result JSON not found: $result" >&2
      fi
    done
  done
fi

python experiments/full_mscoco14_cocu_rcm/diagnostics/collect_results.py \
  --root "$OUT_DIR" --weights-dir weights --seg-results-dir "$SEG_RESULTS_DIR" \
  --arms "$EVAL_ARMS" --seeds "$SEEDS" --benchmarks "$BENCHMARKS"
