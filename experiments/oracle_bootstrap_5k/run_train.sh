#!/usr/bin/env bash
set -euo pipefail

# Usage: bash experiments/oracle_bootstrap_5k/run_train.sh [GPU] [raw|rich] [ALPHA] [WORK_DIR]
# Example: bash experiments/oracle_bootstrap_5k/run_train.sh 1 rich 1.0
# DRY_RUN=1: print without executing. OVERWRITE=1: allow replacing an existing checkpoint.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
cd "$ROOT"

GPU="${1:-1}"
ARM="${2:-rich}"
ALPHA="${3:-1.0}"
WORK="${4:-experiments/raw_rich_5k/work}"
TEACHER="weights/vitb_mlp_infonce.pth"
CONFIG="configs/vitb_mlp_infonce.yaml"
TRAINER="experiments/teacher_guided_bootstrap/train_bootstrap.py"

if [[ "$ARM" != raw && "$ARM" != rich ]]; then
  echo "ERROR: arm must be raw or rich, received: $ARM" >&2
  exit 2
fi

# Canonical tags: 0.0 -> a0, 1.0 -> a1, 0.3 -> a0p3.
ALPHA_NORM="$(python -c 'import math,sys; x=float(sys.argv[1]); (math.isfinite(x) and 0 <= x <= 1) or sys.exit("alpha must lie in [0,1]"); print(f"{x:g}")' "$ALPHA")"
ALPHA_TAG="a${ALPHA_NORM//./p}"

OUT="weights/oracle_bootstrap_5k/${ARM}_${ALPHA_TAG}.pth"
TRAIN="${WORK}/${ARM}_train.pth"
VAL="${WORK}/${ARM}_val.pth"

CMD=(python "$TRAINER"
    --teacher-weights "$TEACHER"
    --train-dataset "$TRAIN"
    --val-dataset "$VAL"
    --model-config "$CONFIG"
    --alpha "$ALPHA_NORM"
    --optimizer Adam
    --scheduler linear
    --warmup 0
    --seed 123
    --output "$OUT")

printf '[RUN] arm=%s alpha=%s GPU=%s\n' "$ARM" "$ALPHA_NORM" "$GPU"
printf '[RUN] train=%s val=%s\n' "$TRAIN" "$VAL"
printf '[RUN] teacher=%s\n' "$TEACHER"
printf '[RUN] output=%s\n' "$OUT"
printf '[CMD] CUDA_VISIBLE_DEVICES=%q ' "$GPU"
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi

for input in "$TEACHER" "$TRAIN" "$VAL" "$CONFIG" "$TRAINER" \
             "experiments/teacher_guided_bootstrap/bootstrap_model.py"; do
  if [[ ! -f "$input" ]]; then
    echo "ERROR: required input missing: $ROOT/$input" >&2
    exit 1
  fi
done
if [[ -f "$OUT" && "${OVERWRITE:-0}" != 1 ]]; then
  echo "ERROR: checkpoint exists: $OUT. Set OVERWRITE=1 to retrain/replace." >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONHASHSEED=123
"${CMD[@]}"
echo "[DONE] checkpoint=$OUT"
echo "[DONE] metrics=${OUT%.pth}.metrics.csv"
echo "[DONE] manifest=${OUT%.pth}.manifest.json"
