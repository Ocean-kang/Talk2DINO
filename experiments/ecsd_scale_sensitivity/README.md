# ECSD Data Scale Sensitivity for Talk2DINO

An isolated experiment package comparing **Raw vs ECSD-v3 B4** at nested MSCOCO2014 training scales:

- 5K, 10K, 20K, 40K, Full (82,783 Train2014 images)
- 3 paired seeds by default: 123, 456, 789
- 5 captions per image
- one shared Raw Val2014 validation set
- original Talk2DINO training and OVSS evaluation code

## 1. Design principles

1. No source edits: all new code lives under `experiments/ecsd_scale_sensitivity/`.
2. Reuse first: ECSD v3 candidate extraction and B4 selection functions are imported from the existing v3 package. The original CLIP helper, DINO extractor, CLIP text-feature extractor, `train.py`, `train_seeded.py`, and OVSS evaluator are reused.
3. One variable at a time: nested image subsets are generated once; Raw and ECSD share image IDs, DINO features, validation set, model configuration, and paired random seeds.
4. Deterministic and auditable: subset manifest, caption hashes, selection trace, feature audits, run names, and result tables are saved.
5. Safe by default: existing outputs are not overwritten unless an explicit overwrite flag is passed.

## 2. Important interpretation

This package creates a **new controlled 5K subset of 5,000 Train2014 images**. It is not the old 4,500-train / 500-validation split used in the earlier 5K experiment. The old 5K result can be reported as historical context, but it is not a directly paired point in this new scale curve.

The official Train2014 split has 82,783 images. The official Val2014 split has 40,504 images. This experiment keeps the complete Val2014 split as a single shared validation set.

## 3. Placement

Unzip the archive at the Talk2DINO repository root:

```bash
cd /path/to/Talk2DINO
unzip ecsd_scale_sensitivity.zip -d .
```

Expected new directory:

```text
Talk2DINO/
└── experiments/
    └── ecsd_scale_sensitivity/
        ├── README.md
        ├── configs/
        ├── data/
        ├── analysis/
        ├── scripts/
        └── tests/
```

The package must be placed alongside the existing `experiments/` directory, not inside `experiments/full_mscoco14_ecsd/`.

## 4. Existing repository components required

```text
experiments/5k_ecsd_v3/candidate_quality/build_candidate_pool_v3.py
experiments/5k_ecsd_v3/selection_ablation/selection_methods_v3.py
experiments/5k_ecsd_v3/configs/default.json
experiments/5k_ecsd/clip_utils.py
experiments/full_mscoco14_ecsd/build_raw_captions.py
experiments/full_mscoco14_ecsd/prepare.py
experiments/full_mscoco14_ecsd/train_seeded.py
dino_extraction_v2.py
text_features_extraction.py
train.py
src/dataset.py
src/train_util.py
src/open_vocabulary_segmentation/main.py
configs/vitb_mlp_infonce.yaml
```

Do not delete or rename those directories.

## 5. Environment and input data

Activate the existing Talk2DINO environment:

```bash
conda activate talk2dino
```

Expected COCO directory:

```text
coco/
├── train2014/
├── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

The local CLIP ViT-B/16 checkpoint must already exist:

```bash
test -f ~/.cache/clip/ViT-B-16.pt
```

The ECSD generator does not download CLIP weights.

If your COCO root or annotation paths differ:

```bash
export COCO_ROOT=/path/to/coco
export TRAIN_ANN=$COCO_ROOT/annotations/captions_train2014.json
export VAL_ANN=$COCO_ROOT/annotations/captions_val2014.json
```

## 6. Recommended execution sequence

Run commands from the Talk2DINO repository root.

### Stage 0: validate environment and package

```bash
bash experiments/ecsd_scale_sensitivity/scripts/00_check_environment.sh
bash experiments/ecsd_scale_sensitivity/scripts/validate_package.sh
```

These checks do not launch GPU training.

### Stage 1: build official Raw captions

```bash
bash experiments/ecsd_scale_sensitivity/scripts/01_build_raw.sh
```

Outputs:

```text
data/coco2014_scale/
├── raw_train.json
├── raw_val.json
└── raw_manifest.json
```

The existing `build_raw_captions.py` normalizes every image to exactly five captions using the repository's annotation-ID ordering policy.

### Stage 2: generate ECSD-v3 B4 captions for Train2014

```bash
bash experiments/ecsd_scale_sensitivity/scripts/02_generate_ecsd_b4.sh
```

This chunked adapter imports the existing v3 functions:

- `build_candidates()`
- `choose_anchor()`
- `select_five()`

The B4 weights and candidate filters are read from the existing `experiments/5k_ecsd_v3/configs/default.json`. It uses the same B4 algorithm, one shared anchor per image, and the same fallback behavior as v3. It changes only the execution strategy to process image chunks and persist a resumable trace.

Outputs:

```text
data/coco2014_scale/ecsd_train_b4_v3.json
outputs/ecsd_scale_sensitivity/
├── ecsd_b4_trace_train.jsonl
└── ecsd_b4_generation_summary.json
```

If interrupted, resume with:

```bash
RESUME_ECSD=1 bash experiments/ecsd_scale_sensitivity/scripts/02_generate_ecsd_b4.sh
```

To intentionally regenerate:

```bash
OVERWRITE_ECSD=1 bash experiments/ecsd_scale_sensitivity/scripts/02_generate_ecsd_b4.sh
```

Do not run resume and overwrite together.

For a small smoke test, run the Python script directly with `--max-images 20 --expected-images 0` and separate output/trace paths. Do not use smoke-test output for training.

### Stage 2b: compute full-train CISD once

```bash
bash experiments/ecsd_scale_sensitivity/scripts/02b_cisd_full.sh
```

This reuses the repository's existing `experiments/full_mscoco14_ecsd/evaluate_cisd.py`. It computes per-image Raw/ECSD CISD and source-fact coverage once for all Train2014 images. The later scale analysis filters these per-image values by the nested subset manifest, so CLIP encoding is not repeated five times.

Output:

```text
outputs/ecsd_scale_sensitivity/caption_metrics/full_train/
├── per_image_cisd.csv
├── summary.csv
└── metadata.json
```

### Stage 3: prepare full feature files

```bash
bash experiments/ecsd_scale_sensitivity/scripts/03_prepare_full_features.sh
```

The script:

1. Calls the original `full_mscoco14_ecsd/prepare.py build-base`.
2. Reuses existing compatible DINO features if found.
3. Otherwise calls the original `dino_extraction_v2.py`.
4. Builds three full feature files: Raw Train2014, ECSD Train2014, and shared Raw Val2014.
5. Calls the original `text_features_extraction.py` once per file.
6. Checks that every annotation has a CLIP `ann_feats` tensor.

Default DINO reuse paths:

```text
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
```

Override if needed:

```bash
DINO_TRAIN_SOURCE=/path/to/train_features.pth \
DINO_VAL_SOURCE=/path/to/val_features.pth \
bash experiments/ecsd_scale_sensitivity/scripts/03_prepare_full_features.sh
```

To force fresh DINO extraction:

```bash
FORCE_DINO=1 bash experiments/ecsd_scale_sensitivity/scripts/03_prepare_full_features.sh
```

To force fresh CLIP text-feature extraction:

```bash
FORCE_CLIP=1 bash experiments/ecsd_scale_sensitivity/scripts/03_prepare_full_features.sh
```

### Stage 4: create nested subsets and audit them

```bash
bash experiments/ecsd_scale_sensitivity/scripts/04_build_nested_subsets.sh
```

Default subset seed: `2026`.

Default scales: 5K, 10K, 20K, 40K, 82,783.

The first N IDs of one fixed random permutation are used at every scale. Thus each smaller subset is contained in every larger subset.

Outputs:

```text
outputs/ecsd_scale_sensitivity/work/
├── subset_manifest.json
├── audit_report.json
├── val_shared.pth
├── 5k/
│   ├── train_raw.pth
│   └── train_ecsd.pth
├── 10k/
├── 20k/
├── 40k/
└── full/
```

Each training file contains the original Talk2DINO schema: `images` with DINO features and `annotations` with `caption`, `image_id`, `id`, and `ann_feats`.

### Stage 4b: aggregate caption diagnostics by scale

```bash
bash experiments/ecsd_scale_sensitivity/scripts/08_collect_caption_metrics.sh
```

This creates `outputs/ecsd_scale_sensitivity/summary/caption_scale_metrics.csv`, including CISD, pairwise similarity, source-fact coverage, corpus concept vocabulary, long-tail concept retention, candidate-pool size, candidate-role coverage, selected-role coverage, and fallback rate.

The long-tail threshold is the number of images in which a Raw concept appears. The default threshold is at most five images. Treat this as a transparent corpus statistic, not a semantic ground-truth label.

### Stage 5: run one-seed pilot

After Stages 1 to 4 pass:

```bash
SEEDS="123" bash experiments/ecsd_scale_sensitivity/scripts/05_train.sh
```

This runs 10 models: five scales times Raw/ECSD.

Checkpoint naming:

```text
weights/vitb_mlp_infonce_scale5k_s123_raw.pth
weights/vitb_mlp_infonce_scale5k_s123_ecsd.pth
weights/vitb_mlp_infonce_scale10k_s123_raw.pth
...
weights/vitb_mlp_infonce_scalefull_s123_ecsd.pth
```

The script reuses `experiments/full_mscoco14_ecsd/train_seeded.py`, which wraps the original `train.py` and `src.train_util` without editing them.

Default training configuration is read from the original `configs/vitb_mlp_infonce.yaml`:

- 100 epochs
- batch size 128
- learning rate 1e-4
- Adam
- InfoNCE
- `disentangled_self_attn` image feature
- `ann_feats` text feature
- `save_best_model: false`

All runs use the same shared Raw Val2014 feature file. This keeps the validation signal identical for Raw and ECSD.

### Stage 6: formal three-seed training

After the pilot completes and is checked:

```bash
SEEDS="123 456 789" SKIP_EXISTING=1 \
bash experiments/ecsd_scale_sensitivity/scripts/05_train.sh
```

To intentionally overwrite an existing checkpoint:

```bash
OVERWRITE=1 SEEDS="123 456 789" \
bash experiments/ecsd_scale_sensitivity/scripts/05_train.sh
```

Do not run duplicate training jobs against the same checkpoint names simultaneously.

### Stage 7: OVSS evaluation

```bash
SEEDS="123 456 789" \
bash experiments/ecsd_scale_sensitivity/scripts/06_seg_eval.sh
```

Default benchmarks: `voc20 context59 coco_stuff ade20k`.

PAMR is disabled by using the original non-PAMR evaluation configs. Raw and ECSD use exactly the same evaluation configuration.

Useful overrides:

```bash
BENCHMARKS="voc20 coco_stuff" \
bash experiments/ecsd_scale_sensitivity/scripts/06_seg_eval.sh

NPROC_PER_NODE=2 \
bash experiments/ecsd_scale_sensitivity/scripts/06_seg_eval.sh
```

The original evaluator writes metrics to:

```text
segmentation_results/<projector_name>.json
```

The package's run folders and logs are stored under `outputs/ecsd_scale_sensitivity/segmentation/` and `outputs/ecsd_scale_sensitivity/logs/`.

### Stage 8: collect tables and plots

```bash
SEEDS="123 456 789" \
bash experiments/ecsd_scale_sensitivity/scripts/07_collect_results.sh
```

Outputs:

```text
outputs/ecsd_scale_sensitivity/summary/
├── per_seed_results.csv
├── paired_seed_delta.csv
├── aggregate_results.csv
├── scale_sensitivity_voc20.png
├── scale_delta_voc20.png
├── scale_sensitivity_context59.png
├── scale_delta_context59.png
├── scale_sensitivity_coco_stuff.png
├── scale_delta_coco_stuff.png
├── scale_sensitivity_ade20k.png
└── scale_delta_ade20k.png
```

The main comparison is the paired difference: `ECSD mIoU - Raw mIoU`. Caption diagnostics are supporting mechanism evidence. Do not infer an ECSD benefit from CISD alone.

## 7. Full pipeline commands

One-seed pilot, including preprocessing, training, evaluation, and collection:

```bash
SEEDS="123" bash experiments/ecsd_scale_sensitivity/scripts/run_all.sh
```

Formal three-seed run:

```bash
SEEDS="123 456 789" bash experiments/ecsd_scale_sensitivity/scripts/run_all.sh
```

`run_all.sh` is sequential. It does not launch jobs in parallel.

## 8. Fixed-step control

The primary experiment uses 100 epochs at every scale to preserve the repository's original training setup. Larger datasets therefore receive more optimizer steps.

An exact fixed-step control is deliberately kept out of the primary pipeline so that the original training loop remains untouched and the main experiment stays minimal. If the 100-epoch results suggest optimizer-budget confounding, run a separate fixed-step follow-up rather than mixing protocols.

Approximate optimizer steps with batch size 128 and five captions per image:

| Scale | Steps per epoch | Steps over 100 epochs |
|---|---:|---:|
| 5K | 196 | 19,600 |
| 10K | 391 | 39,100 |
| 20K | 782 | 78,200 |
| 40K | 1,563 | 156,300 |
| Full | 3,234 | 323,400 |

Exact last-batch behavior is determined by the DataLoader. The manifest records theoretical step counts.

## 9. Data integrity rules

Before training, `audit_report.json` must contain `"status": "ok"`.

The audit checks:

- Raw and ECSD train IDs are identical at every scale.
- The same nested manifest is used for Raw and ECSD.
- Every image has exactly five non-empty captions.
- Every annotation has a CLIP `ann_feats` tensor.
- Train features contain `disentangled_self_attn`.
- Validation features contain `avg_self_attn_out`.
- Validation IDs are fixed and disjoint from Train2014.

## 10. Limitations and interpretation

- B4 is imported from the v3.1 code currently in the repository. Record the Git commit and v3 config hash with experiment outputs.
- The new 5K subset is not the previous 4.5K train split.
- The primary experiment uses fixed 100 epochs, not fixed optimizer steps.
- The package does not include COCO images, DINO features, CLIP weights, segmentation datasets, or trained checkpoints. Because nested feature PTH files duplicate image features across scales and arms, reserve at least 30 GB of free disk space; actual usage depends on the stored DINO tensor shapes.
- Full COCO candidate generation and feature extraction require local data, model caches, disk space, and GPU time.
- The v3 role classifier and coverage score are heuristics, not semantic ground truth.

## 11. Troubleshooting

**Missing `experiments/5k_ecsd_v3`:** update the Talk2DINO checkout to the commit containing the v3 package. Do not copy partial files from an older branch.

**Missing CLIP weights:** place `ViT-B-16.pt` under `~/.cache/clip/` or set `CLIP_CACHE`.

**DINO ID mismatch:** the existing DINO feature file is not the matching official split. Set `REUSE_DINO=0` or point `DINO_TRAIN_SOURCE` / `DINO_VAL_SOURCE` to compatible files.

**Missing `ann_feats`:** rerun Stage 3 with `FORCE_CLIP=1`.

**Existing outputs:** use a fresh output directory or explicit overwrite options. Do not overwrite original experiment directories.

**GPU out of memory during ECSD:** lower `ECSD_IMAGE_BATCH_SIZE` and `ECSD_CLIP_BATCH_SIZE`.

**GPU out of memory during DINO extraction:** lower `DINO_BATCH_SIZE`.

## 12. Reproducibility record

Keep the repository Git commit SHA, v3 config, Talk2DINO training config, Raw manifest, ECSD generation summary, subset manifest, audit report, run logs, checkpoint hashes, segmentation JSON files, final CSV tables, and plots.
