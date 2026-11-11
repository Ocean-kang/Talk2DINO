#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
OUT_DIR="${OUT_DIR:-outputs/5k_rich_diversity_strengthen}"
WEIGHT_TAG="${WEIGHT_TAG:-subset5k_strengthen}"
PAMR="${PAMR:-0}"
read -r -a ARMS <<< "${EVAL_ARMS:-rich_diversity_strengthen}"
read -r -a TASKS <<< "${BENCHMARKS:-voc20 context59 coco_stuff ade20k}"

[[ "$WEIGHT_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid WEIGHT_TAG" >&2; exit 2; }
case "$PAMR" in
  0) mode=no_pamr; suffix="" ;;
  1) mode=pamr; suffix=_pamr ;;
  *) echo "PAMR must be 0 or 1" >&2; exit 2 ;;
esac
for task in "${TASKS[@]}"; do
  case "$task" in
    voc20|context59|coco_stuff|ade20k) ;;
    *) echo "Unknown benchmark: $task" >&2; exit 2 ;;
  esac
done
for arm in "${ARMS[@]}"; do
  case "$arm" in
    raw|rich|rich_weaken|rich_diversity|rich_diversity_strengthen) ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
  esac
  weight="weights/vitb_mlp_infonce_${WEIGHT_TAG}_${arm}.pth"
  [[ -s "$weight" ]] || { echo "Missing checkpoint: $weight" >&2; exit 1; }
done

LOG_DIR="$OUT_DIR/logs/$WEIGHT_TAG/$mode"
mkdir -p "$LOG_DIR"
for arm in "${ARMS[@]}"; do
  name="vitb_mlp_infonce_${WEIGHT_TAG}_${arm}"
  proj="${name}_${mode}"
  alias="weights/$proj.pth"
  # The upstream evaluator keys the metrics JSON by model.proj_name.
  if [[ -L "$alias" && "$(readlink -- "$alias")" == "$name.pth" ]]; then
    :
  else
    ln -sT -- "$name.pth" "$alias"
  fi

  for task in "${TASKS[@]}"; do
    case "$task" in
      voc20) dataset=voc ;;
      context59) dataset=context ;;
      coco_stuff) dataset=stuff ;;
      ade20k) dataset=ade ;;
    esac
    cfg="src/open_vocabulary_segmentation/configs/$dataset"
    output="$OUT_DIR/segmentation/$WEIGHT_TAG/$mode/$arm/$task"
    mkdir -p "$output"
    python -m torch.distributed.run --standalone --nproc_per_node=1 \
      src/open_vocabulary_segmentation/main.py --eval \
      --eval_cfg "$cfg/dinotext_${dataset}_vitb_mlp_infonce.yml" \
      --eval_base_cfg "$cfg/eval_${dataset}${suffix}.yml" \
      --output "$output" --opts "model.proj_name=$proj" \
      2>&1 | tee "$LOG_DIR/seg_${arm}_$task.log"
  done
done

python "$HERE/collect_results.py" --out-dir "$OUT_DIR" --weight-tag "$WEIGHT_TAG" \
  --arms "${ARMS[@]}"
