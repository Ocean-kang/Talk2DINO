# CoCu + Teacher-Guided Bootstrap Talk2DINO (α = 1.0)

This is a **side-car/Ponytail** experiment suite for the public repository
`Ocean-kang/Talk2DINO`. It neither modifies nor copies any files under `src/`,
`configs/`, or the repository root. The whole suite belongs under
`experiments/CoCu_Bootstrap_Talk2dino/`.

## Six independent experiment arms

| Experiment | CoCu extraction | Captions per image | Mechanism |
| --- | --- | ---: | --- |
| `clip_5plus3` | CLIP ViT-B/16 CoCu | 8 | 5 raw + 3 isolated concept texts |
| `clip_B` | CLIP ViT-B/16 CoCu | 5 | Filter semantic residuals, complete raw captions |
| `clip_D` | CLIP ViT-B/16 CoCu | 6 | 5 raw + 1 scene-level supplementary text |
| `local_5plus3` | Local-CoCu (DINO CLS + head matching) | 8 | 5 raw + 3 isolated concept texts |
| `local_B` | Local-CoCu (DINO CLS + head matching) | 5 | Filter semantic residuals, complete raw captions |
| `local_D` | Local-CoCu (DINO CLS + head matching) | 6 | 5 raw + 1 scene-level supplementary text |

Every experiment has its own `run.sh`, supports the full pipeline (`all`) and
individual stages (`concepts`, `build`, `verify`, `train`).

## Required existing repository assets

Run all commands from your **Talk2DINO checkout root**, for example:

```bash
cd /home/master/code/oymk/Talking2dino/Talk2DINO
conda activate /home/master/software/oymk/talk2dino
```

Check the following exist:

```text
# Input image files, raw COCO captions and pre-extracted Talk2DINO features
 data/coco2014/train2014/
 data/coco2014/annotations/captions_train2014.json
 outputs/coco2014_b14/train.pth
 outputs/coco2014_b14/val.pth

# Teacher selection projector, shared by BOTH methods
 weights/vitb_mlp_infonce.pth
 configs/vitb_mlp_infonce.yaml

# ORIGINAL CLIP-CoCu implementation and gallery
 experiments/cocu_imagelevel_compare/build_curated_concepts.py
 experiments/cocu_imagelevel_compare/MSCOCO14_train_gallery_exactmatch.json

# ORIGINAL Local-CoCu implementation and gallery
 experiments/local_cocu/{extract_dino_cls_features.py,extract_candidate_concepts.py,extract_head_features.py,match_head_concepts.py,build_local_compare.py}
 experiments/local_cocu/MSCOCO14_train_gallery.json

# ORIGINAL concept fusion and bootstrap trainers
 experiments/bootstrap_5plus3/build_5plus3.py
 experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/build.py
 experiments/CaptionsConcept_Completion/D_concept_set_supplementary/build.py
 experiments/teacher_guided_bootstrap/train_bootstrap.py
```

Runtime also needs the **same Talk2DINO CUDA/PyTorch environment**, `torchvision`,
PIL, `clip` and repository requirements. DINOv2 is loaded via `torch.hub`
by the existing Local-CoCu extraction scripts; make sure the model is cached if
GitHub access is blocked (e.g. `~/.cache/torch/hub/facebookresearch_dinov2_main`).

**ZIP does not contain model weights, the COCO data or the large gallery JSON**;
it reuses the originals found at the paths above.

## 1. Install

The zip has `CoCu_Bootstrap_Talk2dino/` as its top-level directory. Extract
**into `experiments/`**:

```bash
unzip CoCu_Bootstrap_Talk2dino.zip -d experiments/
```

Make sure the code is at:

```text
Talk2DINO/experiments/CoCu_Bootstrap_Talk2dino/run_all.sh
```

## 2. Run one experiment completely (fresh CoCu -> fusion -> bootstrap)

```bash
# CLIP CoCu + B on GPU 0:
bash experiments/CoCu_Bootstrap_Talk2dino/clip_B/run.sh all 0

# Local CoCu + D on GPU 1:
bash experiments/CoCu_Bootstrap_Talk2dino/local_D/run.sh all 1
```

Each cold run starts by **extracting Concepts from original COCO images/captions**
rather than consuming the previous experiment's manifest. The six arms share
source caches (clip / local), so after the first source extraction subsequent
arms can reuse it. **The dataset construction and student training are
independent for all six arms.** To force complete re-extraction, including DINO
features and concepts:

```bash
bash experiments/CoCu_Bootstrap_Talk2dino/local_B/run.sh all 0 --rebuild-concepts
bash experiments/CoCu_Bootstrap_Talk2dino/clip_B/run.sh all 0 --rebuild-concepts
```

The Local pipeline calls the original scripts in this order:

1. `extract_dino_cls_features.py` on original COCO images (224x224, as previous Local experiment).
2. `shared/chunked_retrieval.py` for **exact cosine Top-50 nonself nearest neighbors**. This replaces the original O(N²) similarity **allocation**, not the retrieval metric.
3. `extract_candidate_concepts.py` from neighbor captions and the existing gallery.
4. `extract_head_features.py` at 448px, using original DINOv2 ViT-B/14-register attention hooks.
5. `match_head_concepts.py` using `weights/vitb_mlp_infonce.pth`.
6. `build_local_compare.py`, then `shared/prepare_manifest.py`.

The CLIP pipeline calls the original `build_curated_concepts.py --backend clip`
using the same exact-match gallery, retrieval k=8, and 3 curated concepts, then
builds the completion manifest. No other `talk2dino_image_cls_imagelevel` model
is used in these six arms.

The new `prepare_manifest.py` uses the **same Top-3 selection** as your original
5+3 pipeline (CLIP: `new_curated_concepts` before `curated_concepts`; Local:
`concept_head_scores` descending). It **fails loudly** when a training image
has fewer than three real candidates; it never invents filler concepts.

## 3. Separate stages

If you prefer verifying intermediate results:

```bash
# 3.1 Extraction (complete CoCu pipeline)
bash experiments/CoCu_Bootstrap_Talk2dino/clip_5plus3/run.sh concepts 0

# 3.2 Build enriched train.pth
bash experiments/CoCu_Bootstrap_Talk2dino/clip_5plus3/run.sh build 0

# 3.3 Verify shapes / exactly 8 captions per image
bash experiments/CoCu_Bootstrap_Talk2dino/clip_5plus3/run.sh verify 0

# 3.4 Train with frozen teacher, alpha 1.0
bash experiments/CoCu_Bootstrap_Talk2dino/clip_5plus3/run.sh train 0
```

Dry-run prints the complete commands **without loading GPU assets**:

```bash
python experiments/CoCu_Bootstrap_Talk2dino/pipeline.py \
  --source local --scheme B --stage all --gpu 1 --dry-run
```

## 4. Train all six arms (sequentially on a GPU)

```bash
bash experiments/CoCu_Bootstrap_Talk2dino/run_all.sh 0
```

Or run individually on separate free GPUs (avoid simultaneous **first-time**
concept extraction of the same source to prevent cache collisions):

```bash
bash experiments/CoCu_Bootstrap_Talk2dino/clip_5plus3/run.sh all 0
bash experiments/CoCu_Bootstrap_Talk2dino/clip_B/run.sh all 1
bash experiments/CoCu_Bootstrap_Talk2dino/local_D/run.sh all 2
```

### Standard training settings

- Teacher = `weights/vitb_mlp_infonce.pth` (also Local head matcher projector).
- `alpha=1.0`: teacher-only `argmax` for head selection.
- Student = randomly initialized. No Teacher weight copy.
- InfoNCE, optimizer, scheduler, feature usage, and validation = original `teacher_guided_bootstrap/train_bootstrap.py`.
- Config = `configs/vitb_mlp_infonce.yaml` (100 epochs, batch size 128, Adam, lr 1e-4).
- Seed = 123, `outputs/coco2014_b14/val.pth` unchanged.
- Teacher not present at validation/inference, only student weights are saved.

## 5. Outputs

Concept sources are constructed **from scratch** (once per method unless forced):

```text
outputs/CoCu_Bootstrap_Talk2dino/concepts/clip/
  clip_vitb16.json
  concepts_manifest.json
  cache/...
outputs/CoCu_Bootstrap_Talk2dino/concepts/local/
  dino_cls/dino_cls.pt
  retrieval/neighbors.json
  candidate/candidate_concepts.json
  head_features/head_features.pt
  matched/matched_concepts.json
  matched/matched_concepts_detailed.json
  local_cocu_compare.json
  concepts_manifest.json
```

Per-arm data:

```text
outputs/CoCu_Bootstrap_Talk2dino/runs/clip_5plus3/train_clip_vitb16_5plus3.pth
outputs/CoCu_Bootstrap_Talk2dino/runs/clip_B/train.pth
outputs/CoCu_Bootstrap_Talk2dino/runs/clip_D/train.pth
outputs/CoCu_Bootstrap_Talk2dino/runs/local_5plus3/train_local_cocu_compare_5plus3.pth
outputs/CoCu_Bootstrap_Talk2dino/runs/local_B/train.pth
outputs/CoCu_Bootstrap_Talk2dino/runs/local_D/train.pth
```

Final Student `.pth` files:

```text
weights/vitb_mlp_infonce_cocu_bt_clip_5plus3_a1.pth
weights/vitb_mlp_infonce_cocu_bt_clip_B_a1.pth
weights/vitb_mlp_infonce_cocu_bt_clip_D_a1.pth
weights/vitb_mlp_infonce_cocu_bt_local_5plus3_a1.pth
weights/vitb_mlp_infonce_cocu_bt_local_B_a1.pth
weights/vitb_mlp_infonce_cocu_bt_local_D_a1.pth
```

Each checkpoint also has `.metrics.csv` and `.manifest.json` from the original
Bootstrap training script. The training model structure is unchanged and all
saved student state dictionaries are compatible with the segmentation evaluator.

## 6. Four-dataset segmentation evaluation

```bash
# Evaluate all six models on GPU 1
bash experiments/CoCu_Bootstrap_Talk2dino/eval_all.sh 1

# Evaluate only one model on GPU 1
bash experiments/CoCu_Bootstrap_Talk2dino/eval_all.sh 1 clip_B
```

Datasets: VOC20, Context59, ADE20K, COCO-Stuff. Uses the same configurations
as `experiments/teacher_guided_bootstrap/eval_all.sh`, setting
`model.proj_name=vitb_mlp_infonce_cocu_bt_<source>_<scheme>_a1`. Outputs are
separated under `output/eval/CoCu_Bootstrap_Talk2dino/`. The script assumes
no other distributed jobs use ports 29501-29504.

## 7. Interpretation / design invariants

- **Do not interpret fixed 100-epoch results as equal optimization budget**.
  5+3 has 8 annotations/image, B has 5, D has 6. Equal-step ablation can follow.
- Local `best_head` in extracted concepts is metadata only; teacher reselects the
  head for each *final fused caption* via the original Bootstrap forward.
- `B` uses exact existing thresholds: visual ratio 0.85, floor 0.18,
  margin 0.20, redundancy ceiling 0.90, max three concepts.
- `D` uses existing within-image visual ratio 0.85; maximum three concepts,
  with original fallback to highest-scoring concept during D fusion.
- All six use the same Concept source extraction configuration within their
  corresponding source; no scores/thresholds are tuned against segmentation tests.
- Zero/insufficient candidate concepts **abort**, rather than silently
  becoming synthetic data. Inspect original Local-CoCu gallery and candidate
  extraction if that happens.

## Notes about existing code

`extract_dino_cls_features.py` and `extract_head_features.py` use torch.hub;
on an offline machine ensure the local cache or patch **your local torch.hub
checkout mechanism** (do not change pretrained model identity). The original
Local-CoCu scripts may take substantial time and storage on full COCO2014; the
reduced-memory chunked retrieval only addresses the N-by-N similarity allocation.
