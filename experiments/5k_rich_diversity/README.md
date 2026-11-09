# Talk2DINO 5K Rich-Diversity experiment

An isolated side-car experiment that adds the `rich_diversity_captions` condition to the existing `raw_rich_weaken_5k` workflow. It reuses the original Talk2DINO DINO extractor, CLIP text extractor, `train.py`, model, InfoNCE loss, and segmentation evaluator. It does not patch or replace any original repository source file.

## 1. Data contract

Default input:

```text
data/coco2014_subset_50k/5k_rich_diversity.json
```

The JSON is expected to contain 5,000 unique `image_id` records and exactly five nonempty captions in each field:

- `raw_captions`
- `rich_captions`
- `rich_weaken_captions`
- `rich_diversity_captions`

The preparation script validates all four fields before feature extraction. The supplied JSON is not copied into this ZIP; keep it at the path above.

## 2. Placement

Unzip at the **Talk2DINO repository root**, not inside the `experiments` directory:

```bash
cd /path/to/Talking2dino/Talk2DINO
unzip 5k_rich_diversity_experiment.zip -d .
```

It creates:

```text
Talk2DINO/
└── experiments/
    └── 5k_rich_diversity/
        ├── README.md
        ├── prepare.py
        ├── run_preprocess.sh
        ├── run_train.sh
        ├── run_seg_eval.sh
        └── collect_results.py
```

All generated experiment data is written under `outputs/5k_rich_diversity/`. Checkpoints are written under the repository's existing `weights/` directory, following the original `train.py` naming convention.

## 3. Environment and image files

Use the existing Talk2DINO environment. No new environment, package installation, model download, or language resource is introduced.

```bash
conda activate talk2dino
```

The image resolver checks the following common layouts under `data/coco2014_subset_50k/`:

```text
COCO_train2014_*.jpg
COCO_val2014_*.jpg
images/train2014/COCO_train2014_*.jpg
images/val2014/COCO_val2014_*.jpg
train2014/COCO_train2014_*.jpg
val2014/COCO_val2014_*.jpg
```

If your image root or caption JSON is elsewhere, override `DATA_DIR` or `CAPTIONS` when running preprocessing.

## 4. Preprocessing

Run from the repository root:

```bash
bash experiments/5k_rich_diversity/run_preprocess.sh
```

This performs the following:

1. Validates 5,000 unique image IDs and all four caption fields (five captions per image).
2. Creates symlinks for the 5,000 source images under `outputs/5k_rich_diversity/work/images/train2014/`.
3. Extracts DINOv2 ViT-B/14 reg features once at 448 x 448, including `disentangled_self_attn` and `avg_self_attn_out`.
4. Creates one deterministic 4,500/500 train/validation image split using seed 123. The exact validation IDs are saved in `val_ids.json`.
5. Creates eight COCO-style feature files: train and validation files for Raw, Rich, Rich-Weaken, and Rich-Diversity.
6. Extracts CLIP ViT-B/16 `ann_feats` for all eight files.
7. Audits image IDs, split consistency, DINO features, and CLIP features before training.

Each arm contains 22,500 training captions and 2,500 validation captions. DINO image features are shared across all four arms; only caption text differs.

Useful overrides:

```bash
# Caption file at a different path
CAPTIONS=/path/to/5k_rich_diversity.json bash experiments/5k_rich_diversity/run_preprocess.sh

# Image root at a different path
DATA_DIR=/path/to/coco_subset bash experiments/5k_rich_diversity/run_preprocess.sh

# Intentionally re-extract DINO features
FORCE_DINO=1 bash experiments/5k_rich_diversity/run_preprocess.sh

# Lower extraction batch sizes if GPU memory is insufficient
DINO_BATCH_SIZE=4 CLIP_BATCH_SIZE=128 bash experiments/5k_rich_diversity/run_preprocess.sh
```

## 5. Training

### Default: train only the new Rich-Diversity arm

This is the default to avoid needlessly retraining the three existing controls:

```bash
bash experiments/5k_rich_diversity/run_train.sh
```

It produces:

```text
weights/vitb_mlp_infonce_subset5k_rich_diversity.pth
```

The command uses the repository's original `configs/vitb_mlp_infonce.yaml` and `train.py` path:

- DINO backbone: `dinov2_vitb14_reg`
- DINO feature: `disentangled_self_attn` for train; the original `train.py` maps validation to `avg_self_attn_out`
- CLIP text feature: `ann_feats` from ViT-B/16
- Projector: original `ProjectionLayer`
- Loss: InfoNCE
- Epochs: 100
- Batch size: 128
- Learning rate: 1e-4
- Optimizer: Adam
- Scheduler: linear, warmup 0 (same original recipe)
- Seed: 123 (set by the original `train.py`)

The script refuses to overwrite an existing checkpoint. Use `OVERWRITE=1` only when you intentionally want to replace that checkpoint.

### Optional: retrain all four caption arms from the same split

For the cleanest four-way comparison, retrain all four arms on the current JSON and its shared split. To preserve the previous `subset5k_*` checkpoints, use a new weight tag:

```bash
TRAIN_ARMS="raw rich rich_weaken rich_diversity" \
WEIGHT_TAG=5k_richdiversity \
bash experiments/5k_rich_diversity/run_train.sh
```

This creates `vitb_mlp_infonce_5k_richdiversity_{raw,rich,rich_weaken,rich_diversity}.pth` without replacing the previous experiment's weights.

If you deliberately want to replace checkpoints using the default `subset5k` tag, set `OVERWRITE=1` and back up the old weights first.

## 6. Segmentation evaluation

Default: evaluate only the new Rich-Diversity checkpoint:

```bash
bash experiments/5k_rich_diversity/run_seg_eval.sh
```

Default benchmarks are VOC20, Pascal Context 59, COCO-Stuff, and ADE20K. PAMR defaults to **off**, matching the no-PAMR evaluation configs used by the previous experiment.

To compare against the three checkpoints from `raw_rich_weaken_5k`, first ensure these files exist:

```text
weights/vitb_mlp_infonce_subset5k_raw.pth
weights/vitb_mlp_infonce_subset5k_rich.pth
weights/vitb_mlp_infonce_subset5k_rich_weaken.pth
weights/vitb_mlp_infonce_subset5k_rich_diversity.pth
```

Then run:

```bash
EVAL_ARMS="raw rich rich_weaken rich_diversity" \
bash experiments/5k_rich_diversity/run_seg_eval.sh
```

If you trained all four with the separate `5k_richdiversity` tag, use:

```bash
WEIGHT_TAG=5k_richdiversity \
EVAL_ARMS="raw rich rich_weaken rich_diversity" \
bash experiments/5k_rich_diversity/run_seg_eval.sh
```

Other useful overrides:

```bash
# Evaluate selected datasets only
BENCHMARKS="voc20 coco_stuff" bash experiments/5k_rich_diversity/run_seg_eval.sh

# Enable PAMR only if the comparison baselines also use PAMR
PAMR=1 bash experiments/5k_rich_diversity/run_seg_eval.sh

# Use the established multi-GPU evaluation setup
NPROC_PER_NODE=2 bash experiments/5k_rich_diversity/run_seg_eval.sh
```

The evaluator writes per-checkpoint metrics to the repository's existing `segmentation_results/` directory and run logs under `outputs/5k_rich_diversity/logs/`.

## 7. Important: verify split compatibility before reusing old checkpoints

The new preprocessing uses the same deterministic split rule as `raw_rich_weaken_5k`: sorted image IDs, Python `random.Random(123)`, and 500 validation images. If the previous experiment used the same 5,000 image IDs, its `val_ids.json` should match exactly.

After preprocessing, check:

```bash
python - <<'PY'
import json
from pathlib import Path
old = Path("outputs/raw_rich_weaken_5k/work/val_ids.json")
new = Path("outputs/5k_rich_diversity/work/val_ids.json")
if not old.is_file():
    raise SystemExit("Previous val_ids.json not found; do not assume old checkpoints use the same split.")
a = json.loads(old.read_text())
b = json.loads(new.read_text())
print("same validation IDs:", a == b)
if a != b:
    raise SystemExit("Split mismatch. Retrain all four arms for a strict comparison.")
PY
```

Reusing the previous Raw/Rich/Rich-Weaken checkpoints is appropriate only when the image IDs, validation split, config, and training recipe match. If they do not, use the fresh four-arm training command above.

## 8. Result collection and output layout

Run at any time after training or evaluation:

```bash
python experiments/5k_rich_diversity/collect_results.py
```

For a separately tagged four-arm run:

```bash
python experiments/5k_rich_diversity/collect_results.py --weight-tag 5k_richdiversity
```

The summary CSV is written to `outputs/5k_rich_diversity/summary.csv`. It records checkpoint status, final train/validation losses when logs are available, and the selected segmentation mIoU values. For Raw/Rich/Rich-Weaken it also falls back to the prior experiment's training logs when the new experiment has not retrained those arms.

Expected generated layout:

```text
outputs/5k_rich_diversity/
├── work/
│   ├── images/train2014/       # symlinks only
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
│   ├── rich_weaken_val.pth
│   ├── rich_diversity_train.pth
│   └── rich_diversity_val.pth
├── config_snapshot/
├── logs/
├── segmentation/
└── summary.csv
```

The original Talk2DINO source files remain unchanged. The package only adds `experiments/5k_rich_diversity/` and writes experiment outputs under `outputs/5k_rich_diversity/` plus the new checkpoint(s) under `weights/`.
