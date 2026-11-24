#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
required=(
  "train.py"
  "configs/vitb_mlp_infonce.yaml"
  "dino_extraction_v2.py"
  "text_features_extraction.py"
  "experiments/5k_ecsd/clip_utils.py"
  "experiments/5k_ecsd_v3/configs/default.json"
  "experiments/5k_ecsd_v3/candidate_quality/build_candidate_pool_v3.py"
  "experiments/5k_ecsd_v3/selection_ablation/selection_methods_v3.py"
  "experiments/full_mscoco14_ecsd/build_raw_captions.py"
  "experiments/full_mscoco14_ecsd/prepare.py"
  "experiments/full_mscoco14_ecsd/train_seeded.py"
  "experiments/full_mscoco14_ecsd/evaluate_cisd.py"
  "experiments/5k_ecsd/ecsd_core.py"
  "src/open_vocabulary_segmentation/configs/voc/eval_voc.yml"
  "src/open_vocabulary_segmentation/configs/context/eval_context.yml"
  "src/open_vocabulary_segmentation/configs/stuff/eval_stuff.yml"
  "src/open_vocabulary_segmentation/configs/ade/eval_ade.yml"
)
for f in "${required[@]}"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing required repository file: $f" >&2; exit 1; }
done
CLIP_CACHE="${CLIP_CACHE:-$HOME/.cache/clip}"
[[ -f "$CLIP_CACHE/ViT-B-16.pt" || -f "$CLIP_CACHE/ViT-B-16.pth" ]] || {
  echo "[ERROR] Missing local CLIP ViT-B/16 weights under $CLIP_CACHE" >&2
  exit 1
}
python - <<'PY'
import torch
print("[ENV] torch", torch.__version__)
print("[ENV] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[ENV] CUDA device:", torch.cuda.get_device_name(0))
PY
mkdir -p outputs/ecsd_scale_sensitivity/config_snapshot
git rev-parse HEAD > outputs/ecsd_scale_sensitivity/config_snapshot/repository_commit.txt
sha256sum experiments/5k_ecsd_v3/configs/default.json configs/vitb_mlp_infonce.yaml \
  > outputs/ecsd_scale_sensitivity/config_snapshot/source_config_hashes.sha256
echo "[PASS] repository files and local CLIP weights found."
echo "[INFO] repository commit and source config hashes recorded under outputs/ecsd_scale_sensitivity/config_snapshot/"
