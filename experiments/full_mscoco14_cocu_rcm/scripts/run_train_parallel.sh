#!/usr/bin/env bash
# Run independent training arms concurrently across GPUs. Each process keeps
# the original single-GPU Talk2DINO batch/loss semantics.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0 1}"
SEEDS="${SEEDS:-123}"
TRAIN_SCRIPT="experiments/full_mscoco14_cocu_rcm/scripts/run_train.sh"
OUT_DIR="${OUT_DIR:-outputs/full_mscoco14_cocu_rcm}"
mkdir -p "$OUT_DIR/logs" weights
read -r -a GPU_LIST <<< "$GPUS"
read -r -a SEED_LIST <<< "$SEEDS"
if [[ "${#GPU_LIST[@]}" -lt 2 ]]; then
  echo "[INFO] One GPU supplied; delegating to run_train.sh."
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" SEEDS="$SEEDS" \
    bash "$TRAIN_SCRIPT"
  exit 0
fi

# Ensure the per-seed shared initialization exists before launching concurrent
# processes. Raw training is also a required comparison arm.
for seed in "${SEED_LIST[@]}"; do
  init="$OUT_DIR/initializations/projector_seed${seed}.pth"
  raw_weight="weights/vitb_mlp_infonce_full_mscoco14_cocu_rcm_raw_seed${seed}.pth"
  if [[ ! -f "$init" && -f "$raw_weight" ]]; then
    echo "[ERROR] Raw weight exists but shared initialization is missing for seed=$seed." >&2
    echo "Restore the original init file or create a fresh output directory; refusing to overwrite Raw silently." >&2
    exit 2
  fi
  if [[ ! -f "$init" || ! -f "$raw_weight" ]]; then
    echo "[INIT] Ensuring shared initialization and Raw arm for seed=$seed on GPU ${GPU_LIST[0]}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" SEEDS="$seed" TRAIN_ARMS="raw" SKIP_EXISTING=1 \
      bash "$TRAIN_SCRIPT"
  fi
  [[ -f "$init" ]] || { echo "[ERROR] Initialization was not created for seed=$seed" >&2; exit 1; }
done

ARMS=(cocu cocu_same cocu_rcm cocu_rcm_shuffled)
status=0
for ((start=0; start<${#ARMS[@]}; start+=${#GPU_LIST[@]})); do
  PIDS=()
  for ((offset=0; offset<${#GPU_LIST[@]}; offset++)); do
    index=$((start + offset))
    [[ "$index" -lt "${#ARMS[@]}" ]] || break
    arm="${ARMS[$index]}"
    gpu="${GPU_LIST[$offset]}"
    log="$OUT_DIR/logs/parallel_${arm}.log"
    echo "[LAUNCH] arm=$arm gpu=$gpu seeds=$SEEDS"
    CUDA_VISIBLE_DEVICES="$gpu" SEEDS="$SEEDS" TRAIN_ARMS="$arm" SKIP_EXISTING=1 \
      bash "$TRAIN_SCRIPT" >"$log" 2>&1 &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
done
if [[ "$status" != "0" ]]; then
  echo "[ERROR] At least one parallel arm failed. Inspect $OUT_DIR/logs/parallel_*.log" >&2
  exit 1
fi
echo "[DONE] Parallel arm training completed."
