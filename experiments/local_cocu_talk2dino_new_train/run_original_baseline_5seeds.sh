#!/usr/bin/env bash
# Original Talk2DINO raw-caption baseline, five independent seeds.
# Place at:
#   experiments/local_cocu_talk2dino_new_train/run_original_baseline_5seeds.sh
#
# Run from the Talk2DINO repository root:
#   bash experiments/local_cocu_talk2dino_new_train/run_original_baseline_5seeds.sh 1 all
#   bash experiments/local_cocu_talk2dino_new_train/run_original_baseline_5seeds.sh 1 train
#   bash experiments/local_cocu_talk2dino_new_train/run_original_baseline_5seeds.sh 1 eval
#   bash experiments/local_cocu_talk2dino_new_train/run_original_baseline_5seeds.sh 1 summary
#
# This uses the original raw-caption train.pth and original train.py/config.
# The only intervention is the RNG seed, injected in-process because the
# repository's train.py/do_train currently hard-code seed=123.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${1:-1}"
STAGE="${2:-all}"
case "$STAGE" in all|train|eval|summary) ;;
  *) echo "Usage: $0 [gpu:1] [all|train|eval|summary]" >&2; exit 2 ;;
esac

SEEDS=(123 124 125 126 127)

CONFIG="configs/vitb_mlp_infonce.yaml"
TRAIN_PTH="outputs/coco2014_b14/train.pth"
VAL_PTH="outputs/coco2014_b14/val.pth"

RUN_OUT="outputs/local_cocu_talk2dino_baseline/five_seeds"
WEIGHT_DIR="weights/local_cocu_talk2dino_baseline/five_seeds"
CFG_ROOT="src/open_vocabulary_segmentation/configs"

TASKS=(voc20 voc coco_stuff ade20k context59)
EVAL_CFG=(
  'voc/dinotext_voc_vitb_mlp_infonce.yml'
  'voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml'
  'stuff/dinotext_stuff_vitb_mlp_infonce.yml'
  'ade/dinotext_ade_vitb_mlp_infonce.yml'
  'context/dinotext_context_vitb_mlp_infonce.yml'
)
EVAL_BASE_CFG=(
  'voc/eval_voc_pamr.yml'
  'voc_bg/eval_voc_bg_pamr.yml'
  'stuff/eval_stuff_pamr.yml'
  'ade/eval_ade_pamr.yml'
  'context/eval_context_pamr.yml'
)

fail() { echo "[ERROR] $*" >&2; exit 1; }
mkdir -p "$RUN_OUT" "$WEIGHT_DIR"

if [[ "$STAGE" == all || "$STAGE" == train ]]; then
  [[ -f "$CONFIG" ]] || fail "Missing config: $CONFIG"
  [[ -f "$TRAIN_PTH" ]] || fail "Missing original raw-caption training data: $TRAIN_PTH"
  [[ -f "$VAL_PTH" ]] || fail "Missing validation data: $VAL_PTH"
  sha256sum "$TRAIN_PTH" | tee "$RUN_OUT/raw_train_dataset.sha256"
  sha256sum "$VAL_PTH" | tee "$RUN_OUT/val_dataset.sha256"
fi

# Save experiment settings and data fingerprints.
python - "$RUN_OUT/settings.json" "$TRAIN_PTH" "$VAL_PTH" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
out, train, val, config = map(Path, sys.argv[1:])
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
record = {
    "experiment": "original_talk2dino_raw_caption_baseline",
    "seeds": [123,124,125,126,127],
    "train_dataset": str(train),
    "train_dataset_sha256": sha(train) if train.exists() else None,
    "validation_dataset": str(val),
    "validation_dataset_sha256": sha(val) if val.exists() else None,
    "model_config": str(config),
    "training_entry": "repository train.py; original raw-caption features; in-process RNG seed override only",
    "segmentation_tasks": ["voc20","voc","coco_stuff","ade20k","context59"],
}
out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
PY

for si in "${!SEEDS[@]}"; do
  seed="${SEEDS[$si]}"
  PROJ="vitb_mlp_infonce_original_baseline_seed${seed}"
  NAME_PEDIX="original_baseline_seed${seed}"
  RUNDIR="$RUN_OUT/seed_${seed}"
  CHECKPOINT="$WEIGHT_DIR/seed_${seed}/${PROJ}.pth"
  ROOT_CKPT="weights/${PROJ}.pth"
  mkdir -p "$RUNDIR" "$(dirname "$CHECKPOINT")"

  echo
  echo "============================================================"
  echo "Original raw-caption Talk2DINO | seed=$seed | stage=$STAGE"
  echo "Train data: $TRAIN_PTH"
  echo "Weight:     $CHECKPOINT"
  echo "============================================================"

  if [[ "$STAGE" == all || "$STAGE" == train ]]; then
    if [[ -f "$CHECKPOINT" ]]; then
      echo "[SKIP TRAIN] checkpoint already exists: $CHECKPOINT"
    else
      [[ ! -e "$ROOT_CKPT" && ! -L "$ROOT_CKPT" ]] || fail \
        "Conflicting root checkpoint exists: $ROOT_CKPT. Inspect/move it manually; refusing overwrite."
      echo "[TRAIN] seed=$seed (original training pipeline, raw captions)"
      CUDA_VISIBLE_DEVICES="$GPU" \
      PYTHONHASHSEED="$seed" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      python -u - "$seed" \
        --model_config "$CONFIG" \
        --train_dataset "$TRAIN_PTH" \
        --val_dataset "$VAL_PTH" \
        --feature_name disentangled_self_attn \
        --text_features ann_feats \
        --name_pedix "$NAME_PEDIX" <<'PY' 2>&1 | tee "$RUNDIR/train.log"
import runpy
import sys
import src.train_util as train_util

seed = int(sys.argv[1])
original_set_seed = train_util.set_seed

def seed_override(_seed_from_original_code):
    # train.py and do_train both call set_seed(123) in the original repository.
    # Override only the RNG seed for this process; keep model/loss/optimizer/data
    # pipeline and all other training behavior unchanged.
    return original_set_seed(seed)

train_util.set_seed = seed_override
sys.argv = ['train.py', *sys.argv[2:]]
print(f'[REPRODUCIBILITY] original Talk2DINO; raw captions; RNG seed={seed}', flush=True)
runpy.run_path('train.py', run_name='__main__')
PY
      [[ -f "$ROOT_CKPT" && ! -L "$ROOT_CKPT" ]] || fail "Training did not create $ROOT_CKPT"
      mv -- "$ROOT_CKPT" "$CHECKPOINT"
      echo "[WEIGHT SAVED] $CHECKPOINT"
    fi
  fi

  if [[ "$STAGE" == all || "$STAGE" == eval ]]; then
    [[ -f "$CHECKPOINT" ]] || fail "Missing checkpoint $CHECKPOINT. Run stage train first."
    if [[ -L "$ROOT_CKPT" ]]; then
      [[ "$(readlink -f "$ROOT_CKPT")" == "$(readlink -f "$CHECKPOINT")" ]] || fail \
        "Conflicting symlink: $ROOT_CKPT"
    elif [[ -e "$ROOT_CKPT" ]]; then
      fail "Conflicting root checkpoint exists: $ROOT_CKPT"
    else
      ln -s "$(realpath --relative-to=weights "$CHECKPOINT")" "$ROOT_CKPT"
    fi

    for ti in "${!TASKS[@]}"; do
      task="${TASKS[$ti]}"
      TASK_DIR="$RUNDIR/eval/$task"
      METRICS="$TASK_DIR/metrics.json"
      mkdir -p "$TASK_DIR"

      if [[ -s "$METRICS" && "$METRICS" -nt "$CHECKPOINT" && "${FORCE_EVAL:-0}" != 1 ]]; then
        echo "[SKIP EVAL] seed=$seed task=$task (valid result exists)"
        continue
      fi

      echo "[EVAL] seed=$seed task=$task"
      FLAG="$TASK_DIR/.eval_started"
      touch "$FLAG"
      SOURCE="segmentation_results/${PROJ}.json"
      # Each task is saved separately in TASK_DIR, so remove the shared log
      # result before each run to guarantee the parsed value is fresh.
      rm -f "$SOURCE"

      CUDA_VISIBLE_DEVICES="$GPU" \
      MASTER_ADDR=127.0.0.1 \
      MASTER_PORT="$((29701 + si*10 + ti))" \
      RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
      python -u src/open_vocabulary_segmentation/main.py \
        --eval \
        --eval_cfg "$CFG_ROOT/${EVAL_CFG[$ti]}" \
        --eval_base_cfg "$CFG_ROOT/${EVAL_BASE_CFG[$ti]}" \
        --output "$TASK_DIR" \
        --opts "model.proj_name=$PROJ" 2>&1 | tee "$TASK_DIR/eval.log"

      [[ -s "$SOURCE" ]] || fail "Segmentation result not created: $SOURCE"
      python - "$SOURCE" "$task" "$METRICS" "$seed" "$CHECKPOINT" <<'PY'
import json, sys
from pathlib import Path
src, task, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
seed, checkpoint = int(sys.argv[4]), sys.argv[5]
obj = json.loads(src.read_text(encoding='utf-8'))
if task not in obj:
    raise SystemExit(f"Expected task={task!r} in {src}; got keys={list(obj)}")
miou = float(obj[task])
if not 0 <= miou <= 100:
    raise SystemExit(f"Invalid mIoU={miou}")
dest.write_text(json.dumps({
    "seed": seed, "task": task, "miou": miou, "checkpoint": checkpoint
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[RESULT] seed={seed} task={task} mIoU={miou:.4f}", flush=True)
PY
    done
  fi
done

if [[ "$STAGE" == all || "$STAGE" == eval || "$STAGE" == summary ]]; then
  echo
  echo "[SUMMARY] raw-caption baseline statistics"
  python - "$RUN_OUT" "${SEEDS[@]}" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2:]]
official = {
    "voc20": 88.54,
    "voc": 65.77,
    "coco_stuff": 30.23,
    "ade20k": 22.53,
    "context59": 42.42,
}
rows = []
summary = {}
for task, official_value in official.items():
    vals = []
    for seed in seeds:
        p = out / f"seed_{seed}" / "eval" / task / "metrics.json"
        if not p.is_file():
            raise SystemExit(f"Missing {p}; finish training/evaluation first.")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("seed") != seed or rec.get("task") != task:
            raise SystemExit(f"Invalid metadata in {p}: {rec}")
        vals.append(float(rec["miou"]))
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    ci_half = 2.776445 * std / math.sqrt(5)
    row = {
        "dataset": task, "official_baseline_reference": official_value,
        **{f"seed_{s}": v for s, v in zip(seeds, vals)},
        "mean": mean, "std_sample": std,
        "delta_vs_official_reference": mean - official_value,
        "ci95_mean_lower": mean - ci_half,
        "ci95_mean_upper": mean + ci_half,
    }
    rows.append(row)
    summary[task] = row

with (out / "mean_std.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(out / "mean_std.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"{'Dataset':<14}{'Official':>10}{'Mean':>10}{'Std':>10}{'Delta':>10}")
for r in rows:
    print(f"{r['dataset']:<14}{r['official_baseline_reference']:>10.2f}"
          f"{r['mean']:>10.3f}{r['std_sample']:>10.3f}"
          f"{r['delta_vs_official_reference']:>+10.3f}")
print(f"\n[SAVED] {out/'mean_std.csv'}")
print(f"[SAVED] {out/'mean_std.json'}")

# If matching F-5 metrics already exist, calculate per-seed paired differences.
f5root = Path("outputs/local_cocu_talk2dino_new_train/F_counterfactual/5/five_seeds")
paired_rows = []
paired_complete = True
for task in official:
    f5_vals, raw_vals = [], []
    for seed in seeds:
        f5p = f5root / f"seed_{seed}" / "eval" / task / "metrics.json"
        rawp = out / f"seed_{seed}" / "eval" / task / "metrics.json"
        if not f5p.is_file() or not rawp.is_file():
            paired_complete = False
            break
        f5_vals.append(float(json.loads(f5p.read_text())["miou"]))
        raw_vals.append(float(json.loads(rawp.read_text())["miou"]))
    if not paired_complete:
        break
    diffs = [f-r for f, r in zip(f5_vals, raw_vals)]
    dmean = statistics.mean(diffs)
    dstd = statistics.stdev(diffs)
    half = 2.776445 * dstd / math.sqrt(5)
    paired_rows.append({
        "dataset": task,
        **{f"seed_{s}_F5_minus_raw": d for s, d in zip(seeds, diffs)},
        "paired_delta_mean": dmean,
        "paired_delta_std": dstd,
        "ci95_lower": dmean-half,
        "ci95_upper": dmean+half,
    })
if paired_complete and paired_rows:
    paired_path = out / "paired_vs_F5.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
        w.writeheader()
        w.writerows(paired_rows)
    print(f"[SAVED] paired per-seed comparison: {paired_path}")
else:
    print("[INFO] F-5 per-seed metrics not all found; skipped paired_vs_F5.csv.")
PY
fi

echo "[DONE] stage=$STAGE"
echo "[OUTPUT] $RUN_OUT"
echo "[WEIGHTS] $WEIGHT_DIR"
