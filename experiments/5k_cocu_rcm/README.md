# CoCu-RCM: Talk2DINO Subset-5K Experiment

This is an isolated side-car experiment. It **does not modify** the original
`src/`, `train.py`, `configs/`, or OVSS evaluator.

## 1. Experiment arms

- `raw`: original five-caption training data, global InfoNCE only.
- `cocu`: original five captions plus up to three CoCu concept captions.
- `cocu_same`: CoCu training plus K region positives from the same image.
- `cocu_rcm`: CoCu training plus K same-concept region positives from K distinct
  training images.
- `cocu_rcm_shuffled`: CoCu training plus K cross-image regions assigned to
  the wrong concept as a correspondence control.

The local objective is supervised-contrastive multi-positive learning:

`L_i = - mean_p log( exp(sim(z_i,p)/T) / sum_a exp(sim(z_i,a)/T) )`

The original Talk2DINO `ContrastiveLoss` is unchanged. It divides its
bidirectional InfoNCE by `batch_size ** 2`; the local loss is divided by the
actual batch size squared before applying `lambda_local`. Raw, scaled and
weighted local losses are logged separately.

## 2. Reused repository components

- `src.dataset.DinoClipDataset`
- `src.model.ProjectionLayer` (or the model class named in the YAML config)
- `src.loss.ContrastiveLoss`
- `src.train_util.set_seed`, `const_lr`, `cosine_lr`
- `experiments/local_cocu_talk2dino/common/vision.py::Vision`
- `experiments/local_cocu_talk2dino/common/algorithms.py::bbox_from_map`
- `experiments/local_cocu_talk2dino/common/data.py::normalize`
- `experiments/local_cocu_novel/build_train.py`
- original OVSS configs and `src/open_vocabulary_segmentation/main.py`

No new backbone, LLM, spaCy, FAISS, or model dependency is introduced.

## 3. Expected inputs

Defaults follow the 5K Rich-Diversity experiment and Local-CoCu D-region
discovery. Override them with environment variables if your paths differ.

| Variable | Default | Purpose |
|---|---|---|
| `BASE_TRAIN` | `outputs/5k_rich_diversity/work/raw_train.pth` | 4,500-image train PTH with five raw captions, DINO and CLIP features |
| `BASE_VAL` | `outputs/5k_rich_diversity/work/raw_val.pth` | Shared 500-image validation PTH |
| `COCU_MATCHES` | `outputs/local_cocu_talk2dino/D_region_discovery/matched/matched_concepts_detailed.json` | Local-CoCu detailed concept matches |
| `IMAGE_ROOT` | `data/coco2014_subset_50k` | Root containing the COCO subset images |
| `CONFIG` | `configs/vitb_mlp_infonce.yaml` | Original Talk2DINO config |
| `WORK_DIR` | `outputs/5k_cocu_rcm/work` | New experiment cache |
| `CLIP_MODEL` | `ViT-B/16` | Existing CLIP model |

The concept adapter accepts:
1. `{"image_id": [{"concept": "...", "score": ..., "best_head": ...}]}`
   (`matched_concepts_detailed.json`);
2. `{"image_id": {"concepts": [...]}}` (`concepts_by_image.json`); or
3. a list of records with `image_id` and `concepts`.

It creates an explicit entry for every training image, including images with
zero concepts, as required by the existing `local_cocu_novel/build_train.py`.

**Important:** `BASE_TRAIN` and `BASE_VAL` must be the already prepared matching
4,500/500 split. The Region Gallery uses `BASE_TRAIN` only. Validation images
never enter the training gallery.

## 4. Placement

Unzip at the Talk2DINO repository root:

```bash
cd /path/to/Talk2DINO
unzip 5k_cocu_rcm_experiment.zip -d .
```

It creates only `experiments/5k_cocu_rcm/`. Runtime outputs go to
`outputs/5k_cocu_rcm/` and uniquely named new checkpoints go to `weights/`.
No existing source file is overwritten.

## 5. Environment and cached models

Activate the same environment used for Talk2DINO, then check:

```bash
python -c "import torch, torchvision, clip; print(torch.__version__)"
```

The package checks for the OpenAI CLIP `ViT-B/16` cache in `~/.cache/clip`
before invoking the existing caption-feature builder. It fails clearly rather
than silently downloading CLIP. DINOv2 must already be present in the existing
`torch.hub` cache because the reused Local-CoCu `Vision` class calls
`torch.hub.load`.

## 6. Preprocessing

```bash
BASE_TRAIN=outputs/5k_rich_diversity/work/raw_train.pth \
BASE_VAL=outputs/5k_rich_diversity/work/raw_val.pth \
COCU_MATCHES=outputs/local_cocu_talk2dino/D_region_discovery/matched/matched_concepts_detailed.json \
IMAGE_ROOT=data/coco2014_subset_50k \
bash experiments/5k_cocu_rcm/run_preprocess.sh
```

Run individual stages:

```bash
STAGE=concept bash experiments/5k_cocu_rcm/run_preprocess.sh
STAGE=build   bash experiments/5k_cocu_rcm/run_preprocess.sh
STAGE=region  bash experiments/5k_cocu_rcm/run_preprocess.sh
STAGE=verify  bash experiments/5k_cocu_rcm/run_preprocess.sh
STAGE=sample  bash experiments/5k_cocu_rcm/run_preprocess.sh
```

`STAGE=all` runs:

1. Concept archive normalization and train-ID filtering.
2. Existing `local_cocu_novel/build_train.py` to preserve five raw captions and
   append up to three CoCu concepts.
3. DINO Attention region extraction using existing Local-CoCu `Vision.get()`
   and `bbox_from_map()`. Existing cached `disentangled_self_attn` features are
   used after a cached-vs-extracted consistency check.
4. Native CLIP ViT-B/16 verification of full images and Attention-derived crops.
5. Same-image, cross-image and shuffled K-shot sampling indices.

Default verification thresholds are configurable:
`MIN_CROP_SCORE=0.0`, `MIN_LOCAL_GAIN=-1.0`, `MIN_MARGIN=0.0`.
The full-image baseline is encoded from the same 448-pixel DINO center-crop
view as the candidate regions. The default local-gain threshold is disabled so
that global context does not accidentally eliminate most local candidates in
the first pass. Use
`local_gain` as a diagnostic, inspect `region_verification_report.json`, and
calibrate stricter thresholds after manual crop inspection.

Overrides:

```bash
COCU_MATCHES=/path/to/matched_concepts_detailed.json \
bash experiments/5k_cocu_rcm/run_preprocess.sh

K_SHOT=3 NEGATIVES=16 bash experiments/5k_cocu_rcm/run_preprocess.sh

MIN_CROP_SCORE=0.15 MIN_LOCAL_GAIN=-0.02 MIN_MARGIN=0.01 \
bash experiments/5k_cocu_rcm/run_preprocess.sh

OVERWRITE=1 bash experiments/5k_cocu_rcm/run_preprocess.sh
```

Main preprocessing outputs:

```text
outputs/5k_cocu_rcm/work/
├── concept_archive.json
├── matched_concepts_detailed_train.json
├── concept_statistics.json
├── train_augmented.pth
├── train_augmented_manifest.json
├── region_gallery/
│   ├── region_features.pt
│   ├── region_metadata.json
│   ├── region_feature_index.json
│   └── region_gallery_manifest.json
├── verified_region_concepts.json
├── region_verification_report.json
├── sampling_index_same_k2.json
├── sampling_index_cross_k2.json
└── sampling_index_shuffled_k2.json
```

## 7. Smoke tests

CPU-side unit tests:

```bash
bash experiments/5k_cocu_rcm/run_smoke_test.sh
```

Optional two-step GPU smoke test after preprocessing:

```bash
RUN_TRAIN_SMOKE=1 bash experiments/5k_cocu_rcm/run_smoke_test.sh
```

It checks a real batch, forward/backward, local-loss gradient flow and dataset
compatibility. Smoke training uses `--no-save` and does not create a main arm
checkpoint.

## 8. Training

First run one seed:

```bash
SEEDS="123" \
TRAIN_ARMS="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled" \
bash experiments/5k_cocu_rcm/run_train.sh
```

Defaults:
- 100 epochs, batch size 128, learning rate `1e-4`
- Adam, linear scheduler, warmup 0
- original global InfoNCE
- K=2, 16 negatives, local temperature 0.07
- local loss weight 0.1, seed 123

One initialization checkpoint is created per seed and reused for every arm.
The original `set_seed`, optimizer and scheduler helpers are reused. Validation
uses `avg_self_attn_out`, exactly as original `train.py` does when training
with `disentangled_self_attn`.

| Arm | Train PTH | Local objective |
|---|---|---|
| `raw` | `BASE_TRAIN` | none |
| `cocu` | `train_augmented.pth` | none |
| `cocu_same` | `train_augmented.pth` | K Head regions from the same image |
| `cocu_rcm` | `train_augmented.pth` | same-concept regions from K distinct images |
| `cocu_rcm_shuffled` | `train_augmented.pth` | K distinct-image regions with incorrect concept assignment |

Three seeds:

```bash
SEEDS="123 456 789" \
TRAIN_ARMS="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled" \
bash experiments/5k_cocu_rcm/run_train.sh
```

Change local loss weight:

```bash
LAMBDA_LOCAL=0.05 bash experiments/5k_cocu_rcm/run_train.sh
```

Train only the RCM arm:

```bash
TRAIN_ARMS="cocu_rcm" bash experiments/5k_cocu_rcm/run_train.sh
```

Existing checkpoints are protected. Set `OVERWRITE=1` only deliberately.

Checkpoint names:

```text
weights/vitb_mlp_infonce_5k_cocu_rcm_<arm>_seed<seed>.pth
```

## 9. OVSS evaluation

The wrapper reuses the original Talk2DINO configs and defaults to PAMR
disabled, matching no-PAMR 5K comparisons:

```bash
EVAL_ARMS="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled" \
SEEDS="123" \
bash experiments/5k_cocu_rcm/run_seg_eval.sh
```

Default benchmarks: VOC20, Pascal Context 59, COCO-Stuff and ADE20K.

Three-seed evaluation:

```bash
SEEDS="123 456 789" \
EVAL_ARMS="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled" \
bash experiments/5k_cocu_rcm/run_seg_eval.sh
```

If PAMR is enabled, use `PAMR=1` consistently for every comparison arm.

## 10. One-command run

```bash
bash experiments/5k_cocu_rcm/run_all.sh
```

For the first run, it is safer to execute preprocessing, smoke test, training
and evaluation separately.

## 11. Outputs

```text
outputs/5k_cocu_rcm/
├── initializations/
├── logs/
├── segmentation/
└── summary.csv
```

The package records source hashes, train IDs, DINO configuration, attention
quantile, CLIP model and verification thresholds, K-shot/negative sampling
parameters, and per-epoch global/local/scaled/weighted losses.

## 12. Scientific limitations

- A DINO Attention Head is an **Attention-conditioned Region Candidate**, not
  a ground-truth object mask.
- CLIP crop-text similarity is a weak verification signal, not a calibrated
  probability or detector.
- `local_gain` compares crop similarity with full-image similarity. It is a
  diagnostic, not proof of localization.
- Negative regions are sampled from regions verified for other normalized
  concepts. Lexical overlap is filtered, but semantic false negatives remain
  possible.
- `cocu_same` requires K distinct verified Head regions from the same image;
  its coverage can be lower than cross-image RCM. Report valid-anchor coverage
  alongside mIoU.
- The package does not assume that RCM improves OVSS. That is the hypothesis
  being tested.

## 13. Troubleshooting

- Missing `ann_feats`: rerun the original 5K CLIP text-feature extraction.
- Missing `disentangled_self_attn`: use the prepared original 5K train PTH.
- Image lookup failure: set `IMAGE_ROOT` to a root containing the
  `images[*].file_name` entries from `BASE_TRAIN`.
- CLIP cache missing: ensure `ViT-B/16` exists in `~/.cache/clip`; no download
  is attempted.
- DINO hub cache missing: use the existing Talk2DINO DINO environment cache.
- Low K-shot coverage: inspect verification/sampling reports and crop examples;
  do not silently lower K.
