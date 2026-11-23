# Full MSCOCO14 ECSD vs Raw for Talk2DINO

This package adds an isolated full-scale experiment under `experiments/full_mscoco14_ecsd/`.
It does **not** modify `train.py`, the Talk2DINO model, the original loss, DINO extraction, CLIP extraction, or the OVSS evaluator.

## 1. Research question and controlled comparison

Compare only two caption arms:

- **Raw**: the five original human-written captions from the official COCO2014 caption annotations.
- **ECSD**: five captions selected from the same five Raw captions using the existing `experiments/5k_ecsd` ECSD-naive algorithm.

The image IDs, DINO features, train/validation split, CLIP encoder, projector configuration, optimizer, learning rate, number of epochs, and segmentation evaluation settings are shared. Raw and ECSD projectors are initialized and trained independently.

The ECSD candidate construction and selection score are reused from `experiments/5k_ecsd/ecsd_core.py` and `generate_ecsd.py`; the score weights are not changed. The new generator only changes the *execution strategy*: it processes image chunks, caches CLIP embeddings per chunk, and writes a resumable JSONL journal.

## 2. Expected data and repository layout

The default COCO root is `coco/` and must contain:

```text
coco/
├── train2014/
├── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

Official COCO2014 counts expected by the scripts:

| Split | Images | Captions |
|---|---:|---:|
| Train2014 | 82,783 | 413,915 |
| Val2014 | 40,504 | 202,520 |

Each image must have exactly five non-empty captions. The scripts deliberately fail on duplicate IDs, missing captions, or unexpected counts instead of silently weakening the comparison.

After extraction, the package creates:

```text
data/coco2014_full/
├── raw_train.json
├── raw_val.json
├── ecsd_train.json
├── ecsd_val.json
└── raw_manifest.json

outputs/full_mscoco14_ecsd/
├── cisd/{train,val}/
├── logs/
├── work/
│   ├── base_train.pth
│   ├── base_val.pth
│   ├── dino_train.pth
│   ├── dino_val.pth
│   ├── raw_train.pth
│   ├── raw_val.pth
│   ├── ecsd_train.pth
│   └── ecsd_val.pth
├── segmentation/
└── summary/
```

The four `*_*.pth` training files contain the original Talk2DINO schema: `images` with DINO features and `annotations` with five captions per image. CLIP `ann_feats` are added by the repository's existing `text_features_extraction.py`.

## 3. Install / placement

Unzip at the **Talk2DINO repository root**, not inside `experiments/`:

```bash
cd /path/to/Talk2DINO
unzip full_mscoco14_ecsd.zip -d .
```

The resulting path must be:

```text
Talk2DINO/experiments/full_mscoco14_ecsd/
```

The package expects the existing `experiments/5k_ecsd/` directory because it imports the original `ecsd_core.py` and `clip_utils.py` from there. Keep that original directory in place.

No dataset or model weights are included in the ZIP. Reserve at least 20-30 GB of free disk space for DINO features, the four assembled feature files, CLIP text features, and logs; actual usage depends on the stored tensor layout and whether compatible DINO features already exist.

## 4. Environment and paths

Activate the existing Talk2DINO environment:

```bash
conda activate talk2dino
```

The repository's existing dependencies must be available, including PyTorch, OpenAI CLIP, torchvision, and the packages required by Talk2DINO. The CLIP ViT-B/16 checkpoint must already be cached at:

```text
~/.cache/clip/ViT-B-16.pt
```

The CLIP helper intentionally refuses to download a missing checkpoint.

If your COCO folder is not `./coco`, override it:

```bash
COCO_ROOT=/path/to/coco \
TRAIN_ANN=/path/to/coco/annotations/captions_train2014.json \
VAL_ANN=/path/to/coco/annotations/captions_val2014.json \
bash experiments/full_mscoco14_ecsd/run_preprocess.sh
```

`COCO_ROOT` must contain `train2014/` and `val2014/`. The annotation files may be elsewhere if `TRAIN_ANN` and `VAL_ANN` are set explicitly.

## 5. Run preprocessing

Recommended first command:

```bash
bash experiments/full_mscoco14_ecsd/run_preprocess.sh
```

It performs:

1. Parse official Train2014 and Val2014 caption annotations into `raw_train.json` and `raw_val.json`.
2. Generate five ECSD captions per image for both splits. CLIP candidate embeddings are computed in image chunks; progress and selected-candidate provenance are saved in JSONL trace files.
3. Compute per-image CISD and lexical source-fact coverage for Raw and ECSD.
4. Build base image manifests from the official COCO image metadata.
5. Reuse compatible DINO features if found, otherwise invoke the repository's existing `dino_extraction_v2.py`.
6. Assemble Raw and ECSD `.pth` files using shared DINO image features.
7. Extract CLIP ViT-B/16 `ann_feats` with the repository's original `text_features_extraction.py`.
8. Audit image IDs, train/val separation, five captions per image, DINO features, and CLIP features.

### DINO feature reuse

By default the script checks these existing files:

```text
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
```

Reuse is allowed only if the source file has exactly the expected image IDs and the required feature field (`disentangled_self_attn` for Train2014, `avg_self_attn_out` for Val2014). A mismatch is a hard error; it will not silently mix feature sets.

If your DINO feature files are stored elsewhere:

```bash
DINO_TRAIN_SOURCE=/path/to/train_features.pth \
DINO_VAL_SOURCE=/path/to/val_features.pth \
bash experiments/full_mscoco14_ecsd/run_preprocess.sh
```

To force a fresh extraction instead of reusing cached features:

```bash
FORCE_DINO=1 bash experiments/full_mscoco14_ecsd/run_preprocess.sh
```

If the existing DINO features are incompatible and you want the script to extract them, use `REUSE_DINO=0`. Fresh extraction uses DINOv2 ViT-B/14-reg at 448 resolution and may require the pretrained DINOv2 weights/repository to be available in the local Torch Hub cache.

### Useful preprocessing overrides

```bash
# Reduce GPU memory use during DINO extraction
DINO_BATCH_SIZE=4 bash experiments/full_mscoco14_ecsd/run_preprocess.sh

# Reduce CLIP candidate/text batch sizes
ECSD_CLIP_BATCH_SIZE=128 CLIP_BATCH_SIZE=128 \
bash experiments/full_mscoco14_ecsd/run_preprocess.sh

# Change the number of images processed per ECSD resume chunk
ECSD_IMAGE_BATCH_SIZE=128 bash experiments/full_mscoco14_ecsd/run_preprocess.sh

# Force regeneration of all ECSD captions and traces
REGENERATE_ECSD=1 bash experiments/full_mscoco14_ecsd/run_preprocess.sh

# Force re-extraction of CLIP text features
FORCE_CLIP=1 bash experiments/full_mscoco14_ecsd/run_preprocess.sh
```

If ECSD generation is interrupted, rerun `run_preprocess.sh`. When the output JSON does not yet exist, the generator resumes from `ecsd_trace_train.jsonl` / `ecsd_trace_val.jsonl`. It verifies a SHA-256 digest of each image's Raw captions before accepting resumed results.

## 6. Train Raw and ECSD

After preprocessing and a successful `work/audit_report.json`, run:

```bash
bash experiments/full_mscoco14_ecsd/run_train.sh
```

Default settings come from the repository's `configs/vitb_mlp_infonce.yaml`:

- 100 epochs
- batch size 128
- learning rate 1e-4
- Adam
- linear scheduler, warmup 0
- InfoNCE
- `disentangled_self_attn` image features and `ann_feats` text features

Default first-stage run uses seed 123 and trains both arms. Checkpoints are saved as:

```text
weights/vitb_mlp_infonce_full14_s123_raw.pth
weights/vitb_mlp_infonce_full14_s123_ecsd.pth
```

`train_seeded.py` is a small adapter around the original `train.py` / `src.train_util.do_train` so seeds can be specified without modifying repository training code. The underlying model, loss, optimizer construction, and training loop are the original Talk2DINO implementation.

For the three-seed experiment after the single-seed pipeline is verified:

```bash
SEEDS="123 42 2026" bash experiments/full_mscoco14_ecsd/run_train.sh
```

To skip checkpoints that already exist:

```bash
SEEDS="123 42 2026" SKIP_EXISTING=1 \
bash experiments/full_mscoco14_ecsd/run_train.sh
```

To intentionally overwrite an existing checkpoint, set `OVERWRITE=1`.

## 7. Evaluate OVSS

Default evaluation runs VOC20, Pascal Context59, COCO-Stuff, and ADE20K with PAMR disabled:

```bash
bash experiments/full_mscoco14_ecsd/run_seg_eval.sh
```

For three seeds:

```bash
SEEDS="123 42 2026" bash experiments/full_mscoco14_ecsd/run_seg_eval.sh
```

Optional settings:

```bash
# Use two GPUs for evaluation
NPROC_PER_NODE=2 bash experiments/full_mscoco14_ecsd/run_seg_eval.sh

# Only evaluate VOC20 and COCO-Stuff
BENCHMARKS="voc20 coco_stuff" bash experiments/full_mscoco14_ecsd/run_seg_eval.sh

# PAMR must be identical across Raw and ECSD
PAMR=1 bash experiments/full_mscoco14_ecsd/run_seg_eval.sh
```

Evaluation uses the original Talk2DINO OVSS evaluator and config files. Each seed/arm receives a unique projector name. The evaluator writes metrics to the repository's `segmentation_results/` directory.

## 8. One-command execution

After checking the data paths and cached model weights:

```bash
bash experiments/full_mscoco14_ecsd/run_all.sh
```

For the formal three-seed run:

```bash
SEEDS="123 42 2026" bash experiments/full_mscoco14_ecsd/run_all.sh
```

The first full run should use the default single seed. Confirm preprocessing, training, and evaluation before launching all seeds.

## 9. Results

Key outputs:

```text
outputs/full_mscoco14_ecsd/
├── ecsd_generation_train.json
├── ecsd_generation_val.json
├── cisd/
│   ├── train/summary.csv
│   └── val/summary.csv
├── work/audit_report.json
├── logs/
└── summary/
    ├── per_seed_results.csv
    ├── paired_seed_delta.csv
    └── aggregate_results.csv
```

- `cisd/*/per_image_cisd.csv`: per-image Raw/ECSD CISD, pairwise similarity, unique fact counts, and ECSD source-fact coverage.
- `summary/per_seed_results.csv`: segmentation metrics and caption diagnostics per seed and arm.
- `summary/paired_seed_delta.csv`: paired `ECSD - Raw` mIoU deltas for each seed.
- `summary/aggregate_results.csv`: mean and standard deviation across seeds.

The principal outcome is the paired segmentation delta, not CISD alone. Higher CISD does not by itself imply better OVSS.

## 10. Tests and safety checks

From the repository root:

```bash
python -m py_compile experiments/full_mscoco14_ecsd/*.py
bash -n experiments/full_mscoco14_ecsd/*.sh
python -m unittest discover -s experiments/full_mscoco14_ecsd/tests -v
```

The smoke tests use synthetic captions and do not train a model or download data.

Before formal training, verify:

- `work/audit_report.json` has `status: ok`.
- Train2014 and Val2014 counts match the official release.
- Raw and ECSD image IDs are identical within each split.
- Train and Val IDs are disjoint.
- Each image has exactly five captions in both arms.
- Both arms have the required DINO and CLIP feature fields.
- `PAMR` and benchmark configs are identical for Raw and ECSD.

## 11. Important scope notes

- The experiment compares **Raw vs ECSD only**. Rich/Rich-Weaken/Rich-Diversity are not retrained here.
- ECSD generation uses only the five original COCO captions for each image. It does not use an LLM, image encoder, CoCu, F6, or bootstrap teacher.
- Train2014 is the training split; Val2014 is the projector validation split. The segmentation benchmarks remain the same held-out OVSS evaluation sets used by the repository.
- No original repository model/training/evaluation files are patched by this package.
