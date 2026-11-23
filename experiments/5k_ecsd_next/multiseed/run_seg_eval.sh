#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ECSD_CONFIG="${ECSD_CONFIG:-experiments/5k_ecsd_next/configs/default.json}"
CFG_OUT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["output_root"])' "$ECSD_CONFIG")/multiseed"
CFG_ARMS="$(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["training"]["arms"]))' "$ECSD_CONFIG")"
CFG_SEEDS="$(python -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["training"]["seeds"])))' "$ECSD_CONFIG")"
CFG_BENCHMARKS="$(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["training"]["benchmarks"]))' "$ECSD_CONFIG")"
OUT_DIR="${OUT_DIR:-$CFG_OUT}"
SEG_OUT="${SEG_OUT:-${OUT_DIR}/segmentation}"
SEG_RESULTS_DIR="${SEG_RESULTS_DIR:-segmentation_results}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EVAL_ARMS="${EVAL_ARMS:-$CFG_ARMS}"
SEEDS="${SEEDS:-$CFG_SEEDS}"
BENCHMARKS="${BENCHMARKS:-$CFG_BENCHMARKS}"

mkdir -p "$OUT_DIR/logs" "$SEG_OUT" "$SEG_RESULTS_DIR"
read -r -a ARMS <<< "$EVAL_ARMS"
read -r -a SEED_LIST <<< "$SEEDS"

resolve_cfg() {
  case "$1" in
    voc20)
      MODEL_CFG="src/open_vocabulary_segmentation/configs/voc/dinotext_voc_vitb_mlp_infonce.yml"
      EVAL_CFG="src/open_vocabulary_segmentation/configs/voc/eval_voc.yml"
      ;;
    context59)
      MODEL_CFG="src/open_vocabulary_segmentation/configs/context/dinotext_context_vitb_mlp_infonce.yml"
      EVAL_CFG="src/open_vocabulary_segmentation/configs/context/eval_context.yml"
      ;;
    coco_stuff)
      MODEL_CFG="src/open_vocabulary_segmentation/configs/stuff/dinotext_stuff_vitb_mlp_infonce.yml"
      EVAL_CFG="src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml"
      ;;
    ade20k)
      MODEL_CFG="src/open_vocabulary_segmentation/configs/ade/dinotext_ade_vitb_mlp_infonce.yml"
      EVAL_CFG="src/open_vocabulary_segmentation/configs/ade/eval_ade.yml"
      ;;
    *)
      echo "[ERROR] Unsupported benchmark $1" >&2; return 2 ;;
  esac
  [[ -f "$MODEL_CFG" && -f "$EVAL_CFG" ]] || {
    echo "[ERROR] Missing config $MODEL_CFG or $EVAL_CFG" >&2; return 1;
  }
  grep -Eq '^[[:space:]]*pamr:[[:space:]]*false' "$EVAL_CFG" || {
    echo "[ERROR] $EVAL_CFG does not explicitly set pamr: false." >&2; return 1;
  }
}

for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    WEIGHT_NAME="vitb_mlp_infonce_${arm}_seed${seed}"
    [[ -f "weights/${WEIGHT_NAME}.pth" ]] || {
      echo "[ERROR] Missing weights/${WEIGHT_NAME}.pth" >&2; exit 1;
    }
    RESULT_FILE="$SEG_RESULTS_DIR/${WEIGHT_NAME}.json"
    if [[ "${OVERWRITE:-0}" == "1" ]]; then
      rm -f "$RESULT_FILE"
    fi

    for benchmark in $BENCHMARKS; do
      resolve_cfg "$benchmark"
      if [[ -f "$RESULT_FILE" ]]; then
        if python - "$RESULT_FILE" "$benchmark" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
except Exception:
    raise SystemExit(3)
raise SystemExit(0 if key in data else 1)
PY
        then
          status=0
        else
          status=$?
        fi
        if [[ "$status" == "0" ]]; then
          echo "[SKIP] Existing metric ${WEIGHT_NAME} / ${benchmark}; set OVERWRITE=1 to rerun all."
          continue
        elif [[ "$status" != "1" ]]; then
          echo "[ERROR] Could not parse $RESULT_FILE" >&2; exit 1
        fi
      fi

      RUN_OUT="$SEG_OUT/$WEIGHT_NAME/$benchmark"
      LOG_PATH="$OUT_DIR/logs/seg_${arm}_seed${seed}_${benchmark}.log"
      mkdir -p "$RUN_OUT"
      echo "================ EVAL ${arm} / seed=${seed} / ${benchmark} (PAMR=False) ================"
      python -m torch.distributed.run \
        --nproc_per_node="$NPROC_PER_NODE" \
        src/open_vocabulary_segmentation/main.py \
        --eval \
        --eval_cfg "$MODEL_CFG" \
        --eval_base_cfg "$EVAL_CFG" \
        --output "$RUN_OUT" \
        --opts "model.proj_name=${WEIGHT_NAME}" \
        2>&1 | tee "$LOG_PATH"
    done
  done
done

echo "[DONE] Multi-seed OVSS evaluation finished. Metrics: $SEG_RESULTS_DIR"
