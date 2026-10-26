#!/usr/bin/env bash
# GPU-safe batches: one experiment per GPU; distinct variant/seed per job.
# Usage: bash experiments/Head_caption_completeness/launch_matrix.sh E0,E1,E2,E3,E4 123,124,125,126,127 0,1 all
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
IFS=',' read -r -a VARIANTS <<< "${1:-E0,E1,E2,E3,E4}"
IFS=',' read -r -a SEEDS <<< "${2:-123,124,125,126,127}"
IFS=',' read -r -a GPUS <<< "${3:-0}"
STAGE="${4:-all}"
[[ "${#GPUS[@]}" -ge 1 ]] || { echo 'Need >=1 GPU index'; exit 2; }
DIR="outputs/Head_caption_completeness/launcher_logs"
mkdir -p "$DIR"
pids=()
labels=()
slot=0
wait_batch() {
  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[FAIL] ${labels[$i]}" >&2
      failed=1
    else
      echo "[OK] ${labels[$i]}"
    fi
  done
  pids=();labels=();slot=0
  if [[ "$failed" -ne 0 ]]; then exit 1; fi
}
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$slot]}"
    name="${variant}_seed${seed}_gpu${gpu}"
    echo "[LAUNCH] $name $STAGE"
    python -m experiments.Head_caption_completeness.pipeline --variant "$variant" \
       --seed "$seed" --stage "$STAGE" --gpu "$gpu" > "$DIR/$name.log" 2>&1 &
    pids+=("$!");labels+=("$name")
    slot=$((slot+1))
    if [[ "$slot" -ge "${#GPUS[@]}" ]]; then wait_batch; fi
  done
done
if [[ "${#pids[@]}" -gt 0 ]]; then wait_batch; fi
python -m experiments.Head_caption_completeness.summarize --allow-missing \
    --variants "$(IFS=','; echo "${VARIANTS[*]}")" \
    --seeds "$(IFS=','; echo "${SEEDS[*]}")"
