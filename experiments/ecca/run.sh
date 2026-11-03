#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source experiments/ecca/settings.sh
STAGE="${1:-help}"
METHOD="${2:-E0}"
SEED="${3:-123}"
MODELS=(--clip-checkpoint "$CLIP_CHECKPOINT" --dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" --teacher "$TEACHER" --model-config "$CONFIG")
SELECT=()
[[ -z "$SELECTION_CONFIG" ]] || SELECT+=(--selection-config "$SELECTION_CONFIG")
[[ -z "$CONCEPT_ALLOWLIST" ]] || SELECT+=(--concept-allowlist "$CONCEPT_ALLOWLIST")
case "$STAGE" in
check)
  python -m experiments.ecca.preflight "${MODELS[@]}" --train "$TRAIN_PTH" --val "$VAL_PTH" --gallery "$GALLERY" --image-root "$IMAGE_ROOT" ;;
manifest)
  python -m experiments.ecca.make_manifest coco --captions "$CAPTIONS_JSON" --instances "$INSTANCES_JSON" --limit 200 --output "$OUT/A/manifest.json" ;;
diagnose)
  if [[ ! -f "$OUT/A/manifest.json" ]]; then
    echo "ECCA: missing $OUT/A/manifest.json; run bash experiments/ecca/run.sh manifest successfully first." >&2
    exit 2
  fi
  python -m experiments.ecca.diagnose "${MODELS[@]}" --manifest "$OUT/A/manifest.json" --image-root "$IMAGE_ROOT" --output "$OUT/A/diagnosis" ;;
recovery)
  python -m experiments.ecca.prepare "${MODELS[@]}" --captions "$TRAIN_PTH" --gallery "$GALLERY" --image-root "$IMAGE_ROOT" --limit 200 --recovery --output "$OUT/recovery" "${SELECT[@]}"
  python -m experiments.ecca.recovery --prepared "$OUT/recovery" --output "$OUT/recovery/report.json" ;;
calibration)
  python -m experiments.ecca.prepare "${MODELS[@]}" --captions "$TRAIN_PTH" --gallery "$GALLERY" --image-root "$IMAGE_ROOT" --offset 200 --limit 100 --audit-panels --output "$OUT/calibration" "${SELECT[@]}"
  python -m experiments.ecca.recovery --prepared "$OUT/calibration" --output "$OUT/calibration/report.json" --export-audit "$OUT/calibration/audit_labels.csv" ;;
prepare)
  python -m experiments.ecca.prepare "${MODELS[@]}" --captions "$TRAIN_PTH" --gallery "$GALLERY" --image-root "$IMAGE_ROOT" --output "$OUT/prepared" "${SELECT[@]}" ;;
train|resume)
  EXTRA=()
  [[ "$METHOD" == E0 ]] || EXTRA+=(--prepared "$OUT/prepared")
  [[ "$METHOD" != C ]] || EXTRA+=(--gate-report "$OUT/A/diagnosis/report.json")
  [[ "$STAGE" != resume ]] || EXTRA+=(--resume)
  python -m experiments.ecca.train --method "$METHOD" --train "$TRAIN_PTH" --val "$VAL_PTH" --model-config "$CONFIG" --seed "$SEED" --output "$OUT/runs/$METHOD/seed_$SEED" "${EXTRA[@]}" ;;
eval|eval_off)
  PAMR=on;BG=config
  [[ "$STAGE" != eval_off ]] || { PAMR=off;BG=off; }
  python -m experiments.ecca.evaluate --checkpoint "$OUT/runs/$METHOD/seed_$SEED/last.pth" --name "${METHOD}_seed${SEED}_last" --output "$OUT/runs/$METHOD/seed_$SEED/segmentation" --clip-checkpoint "$CLIP_CHECKPOINT" --dino-repo "$DINO_REPO" --dino-checkpoint "$DINO_CHECKPOINT" --pamr "$PAMR" --bg-clean "$BG" ;;
summary)
  python -m experiments.ecca.summarize --root "$OUT/runs" --output "$OUT/summary.json" ;;
test)
  python -m unittest discover -s experiments/ecca/tests -v ;;
*)
  echo 'Run from the existing Talk2DINO environment; edit settings.sh first.'
  echo 'CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh check|manifest|diagnose|recovery|calibration|prepare'
  echo 'CUDA_VISIBLE_DEVICES=0 bash experiments/ecca/run.sh train|resume|eval|eval_off E0|E1|E2|E3|E4|E3raw|C 123'
  echo 'bash experiments/ecca/run.sh test|summary' ;;
esac
