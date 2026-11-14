# Talk2DINO Subset5K ECSD-naive experiment

This is an isolated side-car experiment for `Talk2DINO`. It does not modify the
original model, projector, loss, `train.py`, DINO extractor, CLIP feature
extractor, or segmentation evaluator.

## Scope

- Input: `data/coco2014_subset_50k/5k_rich_diversity.json`
- Output: `data/coco2014_subset_50k/5k_ECSD.json`
- Caption arms: `raw`, `rich`, `rich_weaken`, `rich_diversity`, `ecsd`
- Text encoder: the repository's existing OpenAI CLIP `ViT-B/16`
- Training: original Talk2DINO `train.py`, original ViT-B MLP InfoNCE config
- Evaluation: original OVSS evaluation entrypoint and configs
- No LLM/MLLM, CoCu, F6, bootstrap, spaCy, or new model dependency

## ECSD-naive algorithm

For each image, the algorithm only uses its five `raw_captions` as evidence:

1. Candidate construction:
   - retain each complete source caption;
   - split explicit punctuation-delimited clauses;
   - extract supported noun phrases containing at least two content words;
   - wrap noun phrases with the neutral phrase `A photo shows ...`.
2. Evidence tracking:
   - every candidate stores source caption indices and character spans;
   - candidate content tokens must originate from the source captions;
   - no new visual entity, action, attribute, or relation is generated.
3. Set-level selection:
   - choose an anchor caption with strong coverage of recurring source facts;
   - greedily select four additional candidates using weighted fact coverage,
     CLIP embedding distance, role novelty, sentence quality, and weighted
     Jaccard redundancy.
4. Output exactly five `ecsd_captions` per image.

This is deliberately a naive, deterministic, rule-based baseline. It is not a
full dependency parser and should be treated as a first algorithmic baseline.
The lexical rules are in `ecsd_core.py` and can be extended later.

The CISD implementation follows:

`CISD(image) = 1 - mean(cosine(z_i, z_j))`, for all ten pairs among five
L2-normalized CLIP text embeddings.

The evaluator reports per-image CISD for all five caption arms, plus aggregate
mean/std/median/quantiles and the ten pairwise similarities.

## 1. Placement

Unzip the package at the Talk2DINO repository root:

```bash
cd /path/to/Talking2dino/Talk2DINO
unzip 5k_ecsd_experiment.zip -d .
```

It creates:

```text
Talk2DINO/
└── experiments/
    └── 5k_ecsd/
        ├── README.md
        ├── clip_utils.py
        ├── ecsd_core.py
        ├── generate_ecsd.py
        ├── evaluate_cisd.py
        ├── prepare.py
        ├── run_preprocess.sh
        ├── run_train.sh
        ├── run_seg_eval.sh
        ├── run_all.sh
        ├── collect_results.py
        └── tests/
```

No dataset or model weight is included in this ZIP.

## 2. Expected data layout

```text
data/coco2014_subset_50k/
├── 5k_rich_diversity.json
└── images/
    ├── COCO_train2014_000000xxxxxx.jpg
    └── ...
```

The image resolver also supports images directly under the data directory,
`train2014/`, `val2014/`, `images/train2014/`, and `images/val2014/`.

The input JSON must be a list of 5,000 records. Each record must contain
`image_id` and exactly five non-empty strings in each field:

- `raw_captions`
- `rich_captions`
- `rich_weaken_captions`
- `rich_diversity_captions`

The generated JSON preserves all original fields and adds `ecsd_captions`.

## 3. Environment

Activate the existing Talk2DINO environment. No extra Python package is
required beyond the dependencies already used by the repository.

```bash
conda activate talk2dino
```

The scripts reuse the cached OpenAI CLIP `ViT-B/16` weights under
`~/.cache/clip`. They intentionally fail with a clear error if the weights
are not cached, rather than silently downloading a model.

## 4. Generate ECSD captions and compute CISD

From the repository root:

```bash
bash experiments/5k_ecsd/run_preprocess.sh
```

This command:

1. Generates `data/coco2014_subset_50k/5k_ECSD.json` if it does not exist.
2. Computes per-image CISD for Raw, Rich, Rich-Weaken, Rich-Diversity, ECSD.
3. Stages the 5,000 images as symlinks.
4. Reuses `outputs/5k_rich_diversity/work/dino.pth` when available and
   compatible by image IDs; otherwise extracts DINOv2 ViT-B/14-reg features.
5. Creates a deterministic 4,500/500 train/validation split with seed 123.
6. Creates train/val feature files for all five caption arms.
7. Extracts CLIP ViT-B/16 `ann_feats` using the original
   `text_features_extraction.py`.
8. Audits image IDs, split consistency, DINO features, and CLIP features.

Useful overrides:

```bash
# Input and output paths
SOURCE_CAPTIONS=/path/to/5k_rich_diversity.json \
ECSD_CAPTIONS=/path/to/5k_ECSD.json \
bash experiments/5k_ecsd/run_preprocess.sh

# Force regeneration of 5k_ECSD.json
REGENERATE_ECSD=1 bash experiments/5k_ecsd/run_preprocess.sh

# Force a fresh DINO extraction instead of reusing the prior 5K DINO features
FORCE_DINO=1 bash experiments/5k_ecsd/run_preprocess.sh

# Change CLIP cache or extraction batch sizes
CLIP_CACHE=/path/to/clip_cache \
ECSD_BATCH_SIZE=256 \
CLIP_BATCH_SIZE=128 \
bash experiments/5k_ecsd/run_preprocess.sh

# Reduce DINO extraction batch size if GPU memory is insufficient
DINO_BATCH_SIZE=4 bash experiments/5k_ecsd/run_preprocess.sh
```

Generation and CISD outputs:

```text
data/coco2014_subset_50k/5k_ECSD.json
outputs/5k_ecsd/
├── cisd/
│   ├── per_image_cisd.csv
│   ├── summary.csv
│   └── metadata.json
├── logs/
└── work/
    ├── dino.pth
    ├── val_ids.json
    ├── raw_train.pth
    ├── ...
    └── ecsd_val.pth
```

You can regenerate or evaluate CISD independently:

```bash
python experiments/5k_ecsd/generate_ecsd.py \
  --input data/coco2014_subset_50k/5k_rich_diversity.json \
  --output data/coco2014_subset_50k/5k_ECSD.json

python experiments/5k_ecsd/evaluate_cisd.py \
  --captions data/coco2014_subset_50k/5k_ECSD.json \
  --output-dir outputs/5k_ecsd/cisd
```

## 5. Train all five caption arms separately

Default training trains all five arms from the same split and same original
Talk2DINO recipe:

```bash
bash experiments/5k_ecsd/run_train.sh
```

It uses:

- `configs/vitb_mlp_infonce.yaml`
- 100 epochs
- batch size 128
- learning rate 1e-4
- Adam
- linear scheduler, warmup 0
- InfoNCE
- seed 123 (set by the original `train.py`)

To avoid overwriting prior `subset5k_*` weights, the default weight tag is
`ecsd5k`. Checkpoints:

```text
weights/
├── vitb_mlp_infonce_ecsd5k_raw.pth
├── vitb_mlp_infonce_ecsd5k_rich.pth
├── vitb_mlp_infonce_ecsd5k_rich_weaken.pth
├── vitb_mlp_infonce_ecsd5k_rich_diversity.pth
└── vitb_mlp_infonce_ecsd5k_ecsd.pth
```

Train only ECSD if the four baseline checkpoints are already confirmed to use
the same image IDs, validation IDs, config, and training recipe:

```bash
TRAIN_ARMS="ecsd" bash experiments/5k_ecsd/run_train.sh
```

For strict comparison, the recommended first run is to train all five arms:

```bash
TRAIN_ARMS="raw rich rich_weaken rich_diversity ecsd" \
WEIGHT_TAG=ecsd5k \
bash experiments/5k_ecsd/run_train.sh
```

The script refuses to overwrite an existing checkpoint unless
`OVERWRITE=1` is explicitly supplied.

## 6. Evaluate all five arms

Default evaluation runs VOC20, Pascal Context 59, COCO-Stuff, and ADE20K,
with PAMR disabled to match the previous no-PAMR 5K experiments:

```bash
bash experiments/5k_ecsd/run_seg_eval.sh
```

Useful overrides:

```bash
# Evaluate only selected arms
EVAL_ARMS="ecsd raw rich_diversity" bash experiments/5k_ecsd/run_seg_eval.sh

# Evaluate selected benchmarks
BENCHMARKS="voc20 coco_stuff" bash experiments/5k_ecsd/run_seg_eval.sh

# Enable PAMR only when all comparison arms use the same PAMR setting
PAMR=1 bash experiments/5k_ecsd/run_seg_eval.sh

# Use two GPUs for evaluation
NPROC_PER_NODE=2 bash experiments/5k_ecsd/run_seg_eval.sh
```

Evaluation metrics are written by the original evaluator under
`segmentation_results/`; logs and per-arm output folders are under
`outputs/5k_ecsd/`.

## 7. One-command full run

After checking paths and the CLIP cache:

```bash
bash experiments/5k_ecsd/run_all.sh
```

This performs preprocessing, trains all five arms, then evaluates all five.

## 8. Strict-comparison checks

- All five arms use the same 5,000 image IDs.
- All five arms use the same 4,500/500 split.
- DINO image features are shared across arms.
- Each arm has 22,500 train captions and 2,500 validation captions.
- All arms use the same CLIP encoder and original Talk2DINO training code.
- PAMR must be identical across all evaluation arms.
- For old checkpoints to be reused, verify their split IDs and training
  configuration first. Otherwise retrain all five arms with `ecsd5k`.

The split is generated from sorted image IDs with `random.Random(123)`, matching
the split rule in `experiments/5k_rich_diversity/prepare.py`.

## 9. Outputs

```text
outputs/5k_ecsd/
├── cisd/
│   ├── per_image_cisd.csv
│   ├── summary.csv
│   └── metadata.json
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
│   ├── rich_diversity_val.pth
│   ├── ecsd_train.pth
│   └── ecsd_val.pth
├── logs/
├── config_snapshot/
├── segmentation/
└── summary.csv
```
