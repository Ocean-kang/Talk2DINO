# Talk2DINO 5K Raw / Rich / Rich-Weaken Caption Comparison

This is an isolated side-car experiment. It does **not** modify the original
Talk2DINO training, model, loss, or segmentation source files.

## 1. Experimental question

Compare three caption conditions on the same 5,000 COCO images:

| Arm | Caption field |
|---|---|
| Raw | `raw_captions` |
| Rich | `rich_captions` |
| Rich-Weaken | `rich_weaken_captions` |

The supplied `5k_rich_weaken_captions.json` was checked before packaging:
5,000 unique `image_id` records, and exactly five nonempty captions in each
of the three fields for every image.

The image IDs, train/validation split, DINO image features, model
initialization seed, training code, training configuration, and segmentation
evaluation configuration are shared across the three arms. Only the caption
texts differ.

## 2. Placement

Unzip this package at the **Talk2DINO repository root**:

```bash
unzip raw_rich_weaken_5k_experiment.zip -d .
```

It creates:

```text
Talk2DINO/
└── experiments/
    └── raw_rich_weaken_5k/
        ├── README.md
        ├── prepare.py
        ├── run_preprocess.sh
        ├── run_train.sh
        ├── run_seg_eval.sh
        └── collect_results.py
```

The input data should be available at:

```text
data/coco2014_subset_50k/
├── 5k_rich_weaken_captions.json
└── images or train2014/val2014/
    ├── COCO_train2014_000000000077.jpg
    └── ...
```

The image resolver also checks `images/train2014/` and `images/val2014/`.
It accepts the usual COCO 2014 train/val filenames. The JSON contains
`image_id` and the three caption fields; image filenames are reconstructed
from each `image_id`.

If the caption file has another name or location, set `CAPTIONS` when running
the preprocessing script. If the image root differs, set `DATA_DIR`.

## 3. Environment

Use the existing Talk2DINO environment and its existing dependencies.
No package installation, package upgrade, new NLP package, or new model
architecture is introduced.

For example:

```bash
cd /path/to/Talking2dino/Talk2DINO
conda activate talk2dino
```

The existing DINOv2 ViT-B/14 reg and CLIP ViT-B/16 model weights should
already be available through the environment's existing caches.

## 4. Preprocess

Run from the repository root:

```bash
bash experiments/raw_rich_weaken_5k/run_preprocess.sh
```

The script performs the following:

1. Validates 5,000 unique image IDs and all 15,000 captions.
2. Stages image symlinks under `outputs/raw_rich_weaken_5k/work/images/train2014/`.
3. Extracts DINOv2 ViT-B/14 reg features once at 448 x 448:
   - `disentangled_self_attn` for training;
   - `avg_self_attn_out` for validation.
4. Creates a deterministic 4,500-image train / 500-image validation split
   using seed 123. `val_ids.json` records the exact validation IDs.
5. Creates six feature files:
   - `raw_train.pth`, `raw_val.pth`;
   - `rich_train.pth`, `rich_val.pth`;
   - `rich_weaken_train.pth`, `rich_weaken_val.pth`.
6. Extracts CLIP ViT-B/16 `ann_feats` for each caption file.
7. Audits image IDs, split consistency, DINO features, and CLIP features.

Each arm has 22,500 training captions and 2,500 validation captions.
DINO image features are extracted once and reused by all three arms.

Useful overrides:

```bash
# If the JSON is stored under another name/path:
CAPTIONS=data/coco2014_subset_50k/my_captions.json \
  bash experiments/raw_rich_weaken_5k/run_preprocess.sh

# If the image root is different:
DATA_DIR=/path/to/coco_subset \
  bash experiments/raw_rich_weaken_5k/run_preprocess.sh

# Re-extract DINO features intentionally:
FORCE_DINO=1 bash experiments/raw_rich_weaken_5k/run_preprocess.sh
```

## 5. Train with the original Talk2DINO recipe

Run:

```bash
bash experiments/raw_rich_weaken_5k/run_train.sh
```

This invokes the repository's original:

```text
train.py
  -> DinoClipDataset
  -> ProjectionLayer.from_config(...)
  -> src.train_util.do_train(...)
  -> src.loss.ContrastiveLoss(..., ltype='infonce')
```

It uses the unchanged `configs/vitb_mlp_infonce.yaml`:

| Parameter | Value |
|---|---:|
| DINO backbone | `dinov2_vitb14_reg` |
| DINO feature extraction | 448 x 448 |
| CLIP text encoder | `ViT-B/16` |
| Projector | original `ProjectionLayer` |
| Learning rate | 1e-4 |
| Epochs | 100 |
| Batch size | 128 |
| Loss | InfoNCE |
| Optimizer | Adam |
| Seed | 123 |
| Scheduler | linear with warmup 0, i.e. no LR decay |

The original config has `save_best_model: false`, so the original training
routine saves the final-epoch model, not a validation-selected checkpoint.
The script does not alter that setting.

The original `train.py` maps training `disentangled_self_attn` features and
validation `avg_self_attn_out` features when
`--feature_name disentangled_self_attn` is supplied. The runner explicitly
uses this original behavior.

The three checkpoints are:

```text
weights/vitb_mlp_infonce_subset5k_raw.pth
weights/vitb_mlp_infonce_subset5k_rich.pth
weights/vitb_mlp_infonce_subset5k_rich_weaken.pth
```

Training logs and a copy/hash of the original config are saved under:

```text
outputs/raw_rich_weaken_5k/logs/
outputs/raw_rich_weaken_5k/config_snapshot/
```

The script refuses to overwrite an existing checkpoint. To intentionally
retrain and replace all three, run:

```bash
OVERWRITE=1 bash experiments/raw_rich_weaken_5k/run_train.sh
```

## 6. Segmentation evaluation

Run:

```bash
bash experiments/raw_rich_weaken_5k/run_seg_eval.sh
```

By default it evaluates all three checkpoints on:

- VOC20;
- Pascal Context 59 (`context59`, also accepted as `pc59`);
- COCO-Stuff;
- ADE20K.

It reuses the repository's original segmentation model configs and evaluation
configs. PAMR defaults to **off**, matching the no-PAMR evaluation configs.
If the baseline you are comparing against was evaluated with PAMR enabled,
use the same setting for all three arms:

```bash
PAMR=1 bash experiments/raw_rich_weaken_5k/run_seg_eval.sh
```

If only some segmentation datasets are installed, choose them explicitly:

```bash
BENCHMARKS="voc20 coco_stuff" \
  bash experiments/raw_rich_weaken_5k/run_seg_eval.sh
```

For a single visible GPU, the default is `NPROC_PER_NODE=1`. You can override
it if your established evaluation setup uses multiple GPUs.

Evaluation logs are saved under `outputs/raw_rich_weaken_5k/logs/`.
The original segmentation evaluator writes per-checkpoint metrics to:

```text
segmentation_results/
  vitb_mlp_infonce_subset5k_raw.json
  vitb_mlp_infonce_subset5k_rich.json
  vitb_mlp_infonce_subset5k_rich_weaken.json
```

## 7. Summary

After training, and again after segmentation evaluation, run:

```bash
python experiments/raw_rich_weaken_5k/collect_results.py
```

The summary is written to:

```text
outputs/raw_rich_weaken_5k/summary.csv
```

It contains final-epoch training/validation losses, checkpoint existence,
and the available segmentation mIoU values for each caption arm.

## 8. Output layout

```text
outputs/raw_rich_weaken_5k/
├── work/
│   ├── images/train2014/        # symlinks only
│   ├── base.pth
│   ├── dino.pth
│   ├── val_ids.json
│   ├── split_metadata.json
│   ├── audit_report.json
│   ├── raw_train.pth
│   ├── raw_val.pth
│   ├── rich_train.pth
│   ├── rich_val.pth
│   ├── rich_weaken_train.pth
│   └── rich_weaken_val.pth
├── config_snapshot/
├── logs/
├── segmentation/
└── summary.csv
```

The original repository files remain unchanged. This experiment only adds
`experiments/raw_rich_weaken_5k/` and writes its generated data/logs under
`outputs/raw_rich_weaken_5k/`, plus the three uniquely named checkpoints
under `weights/`.
