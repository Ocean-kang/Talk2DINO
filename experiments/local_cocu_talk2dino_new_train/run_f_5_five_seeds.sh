#!/usr/bin/env bash
# F Counterfactual + five captions; five independent Talk2DINO training seeds.
# Place: experiments/local_cocu_talk2dino_new_train/run_f_5_five_seeds.sh
# Usage: bash experiments/local_cocu_talk2dino_new_train/run_f_5_five_seeds.sh 1 all
#        bash experiments/local_cocu_talk2dino_new_train/run_f_5_five_seeds.sh 1 eval
#        bash experiments/local_cocu_talk2dino_new_train/run_f_5_five_seeds.sh 1 summary
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GPU="${1:-1}"
STAGE="${2:-all}"
case "$STAGE" in all|train|eval|summary) ;; *) echo "ERROR: stage must be all/train/eval/summary" >&2; exit 2;; esac

SEEDS=(123 124 125 126 127)
EXP_DIR="experiments/local_cocu_talk2dino_new_train"
BASE_OUT="outputs/local_cocu_talk2dino_new_train/F_counterfactual/5"
TRAIN_PTH="$BASE_OUT/train_augmented.pth"
MANIFEST="$BASE_OUT/train_manifest.json"
VAL_PTH="outputs/coco2014_b14/val.pth"
CONFIG="configs/vitb_mlp_infonce.yaml"
RUN_OUT="$BASE_OUT/five_seeds"
WEIGHT_DIR="weights/local_cocu_talk2dino_new_train/F_counterfactual/5/five_seeds"
RESULT_DIR="$RUN_OUT/results"

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
EVAL_ROOT='src/open_vocabulary_segmentation/configs'

fail() { echo "[ERROR] $*" >&2; exit 1; }
mkdir -p "$RUN_OUT" "$WEIGHT_DIR" "$RESULT_DIR"
[[ -f "$CONFIG" ]] || fail "Missing $CONFIG"
[[ -f "$VAL_PTH" ]] || fail "Missing $VAL_PTH"

# Reuse the same prebuilt F-5 training tensor for every seed; never remining concepts.
if [[ "$STAGE" == 'all' || "$STAGE" == 'train' ]]; then
  if [[ ! -f "$TRAIN_PTH" ]]; then
    echo "[BUILD] F-5 dataset missing: build once, using cached F concepts"
    bash "$EXP_DIR/run.sh" F 5 build "$GPU"
  fi
  [[ -f "$TRAIN_PTH" && -f "$MANIFEST" ]] || fail "Missing $TRAIN_PTH or $MANIFEST"
  # Exclude dataset changes from the stochastic repetitions.
  sha256sum "$TRAIN_PTH" | tee "$RUN_OUT/train_dataset.sha256"
fi

# Capture the comparison setting before running the five seeds.
python - "$RUN_OUT/settings.json" "$TRAIN_PTH" "$MANIFEST" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
p, ds, manifest, config = map(Path, sys.argv[1:])
record = {'experiment':'F_counterfactual_5','seeds':[123,124,125,126,127],
          'dataset':str(ds),'manifest':str(manifest),'model_config':str(config),
          'train_entry':'repository train.py with only RNG seed redirected',
          'segmentation_tasks':['voc20','voc','coco_stuff','ade20k','context59']}
if manifest.exists():
    record['build_meta'] = json.loads(manifest.read_text(encoding='utf-8')).get('meta',{})
p.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
PY

for si in "${!SEEDS[@]}"; do
  seed="${SEEDS[$si]}"
  PROJ="vitb_mlp_infonce_local_cocu_new_F_5_seed${seed}"
  NAME_PEDIX="${PROJ#vitb_mlp_infonce_}"
  RUNDIR="$RUN_OUT/seed_${seed}"
  CHECKPOINT="$WEIGHT_DIR/seed_${seed}/${PROJ}.pth"
  ROOT_CKPT="weights/${PROJ}.pth"
  mkdir -p "$RUNDIR" "$(dirname "$CHECKPOINT")"
  echo "========== SEED $seed ($STAGE) =========="

  if [[ "$STAGE" == 'all' || "$STAGE" == 'train' ]]; then
    if [[ -f "$CHECKPOINT" ]]; then
      echo "[SKIP TRAIN] $CHECKPOINT"
    else
      # Refuse to overwrite any checkpoint produced elsewhere.
      [[ ! -e "$ROOT_CKPT" && ! -L "$ROOT_CKPT" ]] || fail "Unexpected $ROOT_CKPT; move it away manually before retry"
      echo "[TRAIN] independent seed=$seed; unchanged original train.py"
      # train.py AND src.train_util.do_train both force seed 123. Override only
      # src.train_util.set_seed in memory BEFORE importing train.py; all other
      # loss/optimizer/model/DataLoader code remains exactly the repository's.
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED="$seed" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u - "$seed" \
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

def experiment_set_seed(_seed_from_original_code):
    # The original caller supplies 123; we explicitly replace it with this run's seed.
    return original_set_seed(seed)

train_util.set_seed = experiment_set_seed
sys.argv = ['train.py', *sys.argv[2:]]
print(f'[REPRODUCIBILITY] fixed training seed override = {seed}',flush=True)
runpy.run_path('train.py', run_name='__main__')
PY
      [[ -f "$ROOT_CKPT" && ! -L "$ROOT_CKPT" ]] || fail "Training did not generate $ROOT_CKPT"
      mv -- "$ROOT_CKPT" "$CHECKPOINT"
      echo "[WEIGHT] $CHECKPOINT"
    fi
  fi

  if [[ "$STAGE" == 'all' || "$STAGE" == 'eval' ]]; then
    [[ -f "$CHECKPOINT" ]] || fail "Missing checkpoint $CHECKPOINT. Run stage train first."
    if [[ -L "$ROOT_CKPT" ]]; then
      ACTUAL="$(readlink -f "$ROOT_CKPT")"
      EXPECTED="$(readlink -f "$CHECKPOINT")"
      [[ "$ACTUAL" == "$EXPECTED" ]] || fail "Conflicting checkpoint symlink $ROOT_CKPT"
    elif [[ -e "$ROOT_CKPT" ]]; then
      fail "Conflicting ordinary checkpoint at $ROOT_CKPT"
    else
      ln -s "$(realpath --relative-to=weights "$CHECKPOINT")" "$ROOT_CKPT"
    fi

    for ti in "${!TASKS[@]}"; do
      TASK="${TASKS[$ti]}"
      TASK_DIR="$RUNDIR/eval/$TASK"
      METRICS="$TASK_DIR/metrics.json"
      mkdir -p "$TASK_DIR"
      if [[ -s "$METRICS" && "$METRICS" -nt "$CHECKPOINT" ]]; then
        echo "[SKIP EVAL] seed=$seed task=$TASK"
        continue
      fi
      echo "[EVAL] seed=$seed task=$TASK"
      FLAG="$TASK_DIR/.eval_started"
      touch "$FLAG"
      CUDA_VISIBLE_DEVICES="$GPU" \
      MASTER_ADDR=127.0.0.1 MASTER_PORT="$((29601 + si*10 + ti))" \
      RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
      python -u src/open_vocabulary_segmentation/main.py \
        --eval \
        --eval_cfg "$EVAL_ROOT/${EVAL_CFG[$ti]}" \
        --eval_base_cfg "$EVAL_ROOT/${EVAL_BASE_CFG[$ti]}" \
        --output "$TASK_DIR" \
        --opts "model.proj_name=$PROJ" 2>&1 | tee "$TASK_DIR/eval.log"

      SOURCE="segmentation_results/${PROJ}.json"
      [[ -f "$SOURCE" && "$SOURCE" -nt "$FLAG" ]] || fail "Evaluation did not refresh $SOURCE for $TASK"
      python - "$SOURCE" "$TASK" "$METRICS" "$seed" "$CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
src=Path(sys.argv[1]); task=sys.argv[2]; dest=Path(sys.argv[3])
seed=int(sys.argv[4]); ckpt=sys.argv[5]
raw=json.loads(src.read_text(encoding='utf-8'))
if task not in raw:raise RuntimeError(f'No newly evaluated task={task} in {src}')
miou=float(raw[task])
if not 0 <= miou <= 100:raise RuntimeError(f'Invalid mIoU={miou}')
dest.write_text(json.dumps({'seed':seed,'task':task,'miou':miou,'checkpoint':ckpt},indent=2),encoding='utf-8')
print(f'[RESULT] seed={seed} {task}: {miou:.4f}')
PY
    done
  fi
done

if [[ "$STAGE" == 'all' || "$STAGE" == 'eval' || "$STAGE" == 'summary' ]]; then
  echo '[SUMMARY] computing 5-seed mean, sample std, 95% CI, and baseline difference'
  python - "$RUN_OUT" "${SEEDS[@]}" <<'PY'
import csv,json,math,statistics,sys
from pathlib import Path
out=Path(sys.argv[1]); seeds=[int(x) for x in sys.argv[2:]]
base={'voc20':88.54,'voc':65.77,'coco_stuff':30.23,'ade20k':22.53,'context59':42.42}
rows=[]; summary={}
for task, baseline in base.items():
    vals=[]
    for seed in seeds:
        p=out/f'seed_{seed}'/'eval'/task/'metrics.json'
        if not p.is_file():raise SystemExit(f'Missing: {p} (finish all evaluations first)')
        r=json.loads(p.read_text(encoding='utf-8'))
        assert r['seed']==seed and r['task']==task,p
        vals.append(float(r['miou']))
    n=len(vals); mean=statistics.mean(vals); std=statistics.stdev(vals) if n>1 else 0.
    # Student t critical value for df=4, 95% two-sided confidence interval.
    tcrit=2.776445 if n==5 else None
    ci=(tcrit*std/math.sqrt(n)) if tcrit is not None else None
    record={'dataset':task,'baseline':baseline,
            **{f'seed_{s}':v for s,v in zip(seeds,vals)},
            'mean':mean,'std_sample':std,'delta_mean':mean-baseline,
            'ci95_lower':mean-ci if ci is not None else None,
            'ci95_upper':mean+ci if ci is not None else None}
    rows.append(record)
    summary[task]={'baseline':baseline,'seed_values':dict(zip(map(str,seeds),vals)),
                   'mean':mean,'std_sample':std,'delta_mean':mean-baseline,
                   'ci95':[mean-ci,mean+ci] if ci is not None else None}
with (out/'mean_std.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'mean_std.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('\n'+'Dataset'.ljust(14)+'Baseline  '+'Mean ± Std'.ljust(18)+'Delta Mean')
for row in rows:
    print(f"{row['dataset']:<14}{row['baseline']:<10.2f}{row['mean']:.3f} ± {row['std_sample']:.3f}     {row['delta_mean']:+.3f}")
print('\n[SAVED]',out/'mean_std.csv',out/'mean_std.json')
PY
fi

echo "[DONE] stage=$STAGE; saved under $RUN_OUT"
