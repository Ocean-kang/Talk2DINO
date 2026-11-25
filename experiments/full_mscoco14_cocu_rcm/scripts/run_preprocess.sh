#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

BASE_TRAIN="${BASE_TRAIN:-outputs/coco2014_b14/train.pth}"
BASE_VAL="${BASE_VAL:-outputs/coco2014_b14/val.pth}"
COCO_ANN="${COCO_ANN:-data/coco2014/annotations/captions_train2014.json}"
IMAGE_ROOT="${IMAGE_ROOT:-data/coco2014/train2014}"
COCU_MATCHES="${COCU_MATCHES:-outputs/local_cocu_talk2dino/D_region_discovery/matched/matched_concepts_detailed.json}"
HEAD_FEATURES="${HEAD_FEATURES:-}"
GALLERY="${GALLERY:-experiments/local_cocu/MSCOCO14_train_gallery.json}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-weights/vitb_mlp_infonce.pth}"
CONFIG="${CONFIG:-configs/vitb_mlp_infonce.yaml}"
WORK_DIR="${WORK_DIR:-outputs/full_mscoco14_cocu_rcm/work}"
DEVICE="${DEVICE:-cuda}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/16}"
MAX_NOVEL="${MAX_NOVEL:-3}"
K_SHOT="${K_SHOT:-2}"
NEGATIVES="${NEGATIVES:-16}"
STAGE="${STAGE:-all}"
OVERWRITE="${OVERWRITE:-0}"
AUTO_GENERATE_CONCEPTS="${AUTO_GENERATE_CONCEPTS:-1}"
AUTO_EXTRACT_HEADS="${AUTO_EXTRACT_HEADS:-0}"

mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

need_file() { [[ -f "$1" ]] || { echo "[ERROR] Missing file: $1" >&2; exit 1; }; }
check_clip_cache() {
  python -c '
import importlib, pathlib, sys, clip
impl=importlib.import_module("clip.clip")
name=sys.argv[1]
models=getattr(impl,"_MODELS",{})
if name not in models: raise SystemExit("Unknown CLIP model: "+name)
path=pathlib.Path.home()/".cache"/"clip"/models[name].split("/")[-1]
if not path.is_file(): raise SystemExit("CLIP cache missing (no download): "+str(path))
print("[CLIP CACHE]",path)
' "$CLIP_MODEL"
}

do_data() {
  need_file "$BASE_TRAIN"; need_file "$BASE_VAL"
  EXTRA=()
  [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
  python experiments/full_mscoco14_cocu_rcm/data/prepare_full_data.py \
    --base-train "$BASE_TRAIN" --base-val "$BASE_VAL" \
    --out-dir "$WORK_DIR" \
    --expected-train "${EXPECTED_TRAIN_IMAGES:-82783}" \
    --expected-val "${EXPECTED_VAL_IMAGES:-40504}" "${EXTRA[@]}" \
    2>&1 | tee "$WORK_DIR/logs/data_audit.log"
}

do_concept() {
  need_file "$WORK_DIR/raw_train.pth"
  MATCHES="$COCU_MATCHES"
  if [[ ! -f "$MATCHES" ]] || ! python experiments/full_mscoco14_cocu_rcm/data/build_full_concept_archive.py \
      --base-train "$WORK_DIR/raw_train.pth" --matches "$MATCHES" \
      --out-dir "$WORK_DIR/concept_archive" --max-concepts "$MAX_NOVEL" \
      --expected-images "${EXPECTED_TRAIN_IMAGES:-82783}" --check-only; then
    if [[ "$AUTO_GENERATE_CONCEPTS" != "1" ]]; then
      echo "[ERROR] Existing CoCu matches do not cover all Train2014 IDs. Set AUTO_GENERATE_CONCEPTS=1." >&2
      exit 1
    fi
    HEAD_ARGS=()
    if [[ -n "$HEAD_FEATURES" ]]; then
      if [[ ! -f "$HEAD_FEATURES" ]]; then
        if [[ "$AUTO_EXTRACT_HEADS" == "1" ]]; then
          need_file "$COCO_ANN"
          [[ -d "$IMAGE_ROOT" ]] || { echo "[ERROR] Missing image directory: $IMAGE_ROOT" >&2; exit 1; }
          python -c 'import pathlib,torch,sys; p=pathlib.Path(torch.hub.get_dir()); ok=list(p.glob("facebookresearch_dinov2_*")); print("[DINO HUB CACHE]",ok[0] if ok else "MISSING"); sys.exit(0 if ok else 1)' || { echo "[ERROR] DINOv2 hub cache missing; no download will be attempted." >&2; exit 1; }
          mkdir -p "$(dirname "$HEAD_FEATURES")"
          python experiments/local_cocu/extract_head_features.py \
            --annotation "$COCO_ANN" --image_dir "$IMAGE_ROOT" \
            --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 \
            --batch_size "${HEAD_BATCH_SIZE:-16}" --fp16 --output "$HEAD_FEATURES" \
            2>&1 | tee "$WORK_DIR/logs/extract_head_features.log"
        else
          echo "[ERROR] Explicit HEAD_FEATURES path does not exist: $HEAD_FEATURES" >&2
          echo "Unset HEAD_FEATURES to reuse raw_train.pth disentangled_self_attn, or set AUTO_EXTRACT_HEADS=1." >&2
          exit 1
        fi
      fi
      HEAD_ARGS+=(--head-features "$HEAD_FEATURES")
    fi
    need_file "$COCO_ANN"; need_file "$GALLERY"; need_file "$TEACHER_WEIGHTS"; need_file "$CONFIG"
    mkdir -p "$WORK_DIR/discovery"
    MATCHES="$WORK_DIR/discovery/matched_concepts_detailed_full.json"
    DISCOVERY_EXTRA=()
    [[ "$OVERWRITE" == "1" ]] && DISCOVERY_EXTRA+=(--overwrite)
    python experiments/full_mscoco14_cocu_rcm/data/generate_full_cocu_matches.py \
      --coco-annotations "$COCO_ANN" --image-root "$IMAGE_ROOT" \
      --base-train "$WORK_DIR/raw_train.pth" "${HEAD_ARGS[@]}" --gallery "$GALLERY" \
      --teacher-weights "$TEACHER_WEIGHTS" --config "$CONFIG" \
      --output "$MATCHES" --checkpoint-dir "$WORK_DIR/discovery" \
      --device "$DEVICE" --chunk-size "${CONCEPT_CHUNK_SIZE:-256}" \
      --max-concepts "$MAX_NOVEL" --expected-images "${EXPECTED_TRAIN_IMAGES:-82783}" "${DISCOVERY_EXTRA[@]}" \
      2>&1 | tee "$WORK_DIR/logs/full_cocu_discovery.log"
  fi
  need_file "$MATCHES"
  EXTRA=()
  [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
  python experiments/full_mscoco14_cocu_rcm/data/build_full_concept_archive.py \
    --base-train "$WORK_DIR/raw_train.pth" --matches "$MATCHES" \
    --out-dir "$WORK_DIR/concept_archive" --max-concepts "$MAX_NOVEL" \
    --expected-images "${EXPECTED_TRAIN_IMAGES:-82783}" \
    --shard-size "${CONCEPT_SHARD_SIZE:-1000}" "${EXTRA[@]}" \
    2>&1 | tee "$WORK_DIR/logs/concept_archive.log"
  COCU_MATCHES="$MATCHES"
}

do_build() {
  need_file "$WORK_DIR/raw_train.pth"
  need_file "$WORK_DIR/matched_concepts_detailed_train.json"
  check_clip_cache
  EXTRA=()
  [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
  python experiments/full_mscoco14_cocu_rcm/data/build_augmented_train.py \
    --base-train "$WORK_DIR/raw_train.pth" \
    --matched "$WORK_DIR/matched_concepts_detailed_train.json" \
    --output "$WORK_DIR/train_augmented.pth" \
    --manifest "$WORK_DIR/train_augmented_manifest.json" \
    --max-novel "$MAX_NOVEL" --clip-model "$CLIP_MODEL" \
    --batch-size "${CLIP_BATCH_SIZE:-256}" --device "$DEVICE" "${EXTRA[@]}" \
    2>&1 | tee "$WORK_DIR/logs/build_train_augmented.log"
}

do_region() {
  need_file "$WORK_DIR/raw_train.pth"
  need_file "$WORK_DIR/concept_archive/manifest.json"
  EXTRA=()
  [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
  python experiments/full_mscoco14_cocu_rcm/data/extract_region_gallery.py \
    --base-train "$WORK_DIR/raw_train.pth" \
    --concept-archive "$WORK_DIR/concept_archive" \
    --image-root "$IMAGE_ROOT" \
    --out-dir "$WORK_DIR/region_gallery" \
    --device "$DEVICE" --shard-size "${REGION_SHARD_SIZE:-256}" \
    --quantile 0.90 --expand 0.15 --region-scope "${REGION_SCOPE:-concepts}" "${EXTRA[@]}" \
    2>&1 | tee "$WORK_DIR/logs/region_gallery.log"
}

do_verify() {
  need_file "$WORK_DIR/concept_archive/manifest.json"
  need_file "$WORK_DIR/region_gallery/region_gallery_manifest.json"
  check_clip_cache
  EXTRA=()
  [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
  python experiments/full_mscoco14_cocu_rcm/data/verify_regions.py \
    --concept-archive "$WORK_DIR/concept_archive" \
    --region-dir "$WORK_DIR/region_gallery" \
    --image-root "$IMAGE_ROOT" \
    --out-dir "$WORK_DIR/region_verification" \
    --clip-model "$CLIP_MODEL" --device "$DEVICE" \
    --max-regions-per-concept 3 \
    --min-crop-score "${MIN_CROP_SCORE:-0.0}" \
    --min-local-gain "${MIN_LOCAL_GAIN:--1.0}" \
    --min-margin "${MIN_MARGIN:-0.0}" "${EXTRA[@]}" \
    2>&1 | tee "$WORK_DIR/logs/region_verification.log"
}

do_sample() {
  need_file "$WORK_DIR/train_augmented.pth"
  need_file "$WORK_DIR/region_gallery/region_feature_index.json"
  need_file "$WORK_DIR/region_verification/verification_manifest.json"
  EXTRA=()
  [[ "$OVERWRITE" == "1" ]] && EXTRA+=(--overwrite)
  python experiments/full_mscoco14_cocu_rcm/data/build_sampling_index.py \
    --train-augmented "$WORK_DIR/train_augmented.pth" \
    --verified-dir "$WORK_DIR/region_verification/verified_region_concepts" \
    --region-index "$WORK_DIR/region_gallery/region_feature_index.json" \
    --out-dir "$WORK_DIR" --k "$K_SHOT" --negatives "$NEGATIVES" \
    --seed "${SAMPLING_SEED:-123}" --shard-size "${SAMPLING_SHARD_SIZE:-2000}" "${EXTRA[@]}" \
    2>&1 | tee "$WORK_DIR/logs/sampling_index.log"
}

case "$STAGE" in
  data) do_data ;;
  concept) do_concept ;;
  build) do_build ;;
  region) do_region ;;
  verify) do_verify ;;
  sample) do_sample ;;
  all) do_data; do_concept; do_build; do_region; do_verify; do_sample ;;
  *) echo "[ERROR] STAGE must be data|concept|build|region|verify|sample|all" >&2; exit 2 ;;
esac
