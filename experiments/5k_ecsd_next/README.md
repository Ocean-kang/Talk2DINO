# ECSD Next Stage: Candidate Quality → Selection Ablation → Multi-Seed OVSS

This is an isolated extension for [Ocean-kang/Talk2DINO](https://github.com/Ocean-kang/Talk2DINO). It creates `experiments/5k_ecsd_next/` and leaves the original `experiments/5k_ecsd/` untouched.

## Research question

Can we construct five complementary, visually meaningful captions from the existing Raw Captions using deterministic non-LLM algorithms, and improve Talk2DINO OVSS?

The experiment follows the requested order:

1. **A · Candidate Quality**: diagnose the existing ECSD-naive pool and selected captions, audit 100 images, then build a traceable improved candidate pool.
2. **B · Selection Ablation**: hold the improved candidate pool, CLIP encoder, candidate filter, and anchor constant; vary only the B1-B4 objective.
3. **C · Multi-Seed OVSS**: compare Raw, Rich, Rich-Diversity, and ECSD v2 across seeds 123/456/789 on VOC20, Pascal Context 59, COCO-Stuff, and ADE20K.

## Constraints enforced

- Only MSCOCO14 Subset5K is used.
- ECSD candidate content is derived only from `raw_captions`.
- No LLM, VLM, external caption API, CoCu, F6, Bootstrap, spaCy, or semantic annotation model.
- CLIP ViT-B/16 is used only for text embeddings and similarity; the existing local cache is required.
- Existing DINOv2 ViT-B/14-reg features are reused. This package does not extract DINO features or download model weights.
- The original Talk2DINO backbone, projector, InfoNCE loss, optimizer recipe, and OVSS inference code are not redesigned.
- Existing Raw/Rich/Rich-Weaken/Rich-Diversity captions, checkpoints, and experiment outputs are not overwritten by default.
- No GPU training or segmentation evaluation is performed by the package validation.

## 1. Placement

Download the ZIP and extract it at the Talk2DINO repository root:

```bash
cd /path/to/Talking2dino/Talk2DINO
unzip 5k_ecsd_next.zip -d .
```

The archive creates:

```text
Talk2DINO/
└── experiments/
    └── 5k_ecsd_next/
        ├── README.md
        ├── configs/default.json
        ├── candidate_quality/
        ├── selection_ablation/
        ├── multiseed/
        ├── scripts/
        ├── patches/
        ├── tests/
        └── docs/
```

No COCO images, caption JSON, checkpoints, DINO features, or CLIP weights are bundled.

## 2. Environment and prerequisites

Activate the existing Talk2DINO environment. The scripts rely on dependencies already used by the repository:

- Python and the standard library
- PyTorch
- OpenAI CLIP package (`clip`)
- Existing `matplotlib` for optional plots
- Bash

No new model dependency is installed.

Check the required local resources:

```bash
# Existing caption file
test -f data/coco2014_subset_50k/5k_rich_diversity.json

# Existing CLIP ViT-B/16 cache (used by the original text feature extractor)
test -f "$HOME/.cache/clip/ViT-B-16.pt"

# Preferred shared DINO feature source
test -f outputs/5k_rich_diversity/work/dino.pth
```

If the DINO file is stored elsewhere, set `DINO_SOURCE=/path/to/dino.pth` when running Stage C. The file must contain exactly the same 5,000 image IDs and both `disentangled_self_attn` and `avg_self_attn_out` features. This package intentionally fails rather than downloading or extracting another model.

The original `text_features_extraction.py` uses the default OpenAI CLIP cache under `~/.cache/clip` and does not expose a custom cache argument. Stage C checks the configured local `ViT-B-16.pt`; if it is stored elsewhere, the script creates a symlink at the upstream default cache path. It refuses to replace an existing link/file that points to a different weight. No download occurs.

## 3. Configuration

Edit `experiments/5k_ecsd_next/configs/default.json` to change:

- dataset and output paths
- DINO feature source and CLIP cache path
- candidate construction thresholds
- candidate validity filters
- B1-B4 score weights
- fixed train/validation split seed and size
- training seeds, caption arms, and evaluation benchmarks

Environment variables can override shell-script defaults for one run. For example:

```bash
DATA_DIR=/data/coco_subset \
SOURCE_CAPTIONS=/data/coco_subset/5k_rich_diversity.json \
bash experiments/5k_ecsd_next/scripts/run_experiment_a.sh
```

The four objective definitions are read from `configs/default.json`; the shipped values are:

| Method | Coverage | CLIP diversity | Role novelty |
|---|---:|---:|---:|
| B1 | 1.0 | 0.0 | 0.0 |
| B2 | 0.0 | 1.0 | 0.0 |
| B3 | 0.5 | 0.5 | 0.0 |
| B4 | 0.4 | 0.4 | 0.2 |

## 4. Lightweight validation

Run this from the repository root:

```bash
bash experiments/5k_ecsd_next/scripts/validate_package.sh
```

It performs Python syntax compilation, unit tests, and Bash syntax checks. It does not load CLIP, run GPU training, or run segmentation evaluation.

After caption generation, validate the produced JSON:

```bash
python experiments/5k_ecsd_next/tests/validate_json_schema.py \
  --input data/coco2014_subset_50k/5k_ECSD_v2.json \
  --expected-images 5000 \
  --require-ecsd-v2
```

## 5. Stage A · Candidate Quality

Run:

```bash
bash experiments/5k_ecsd_next/scripts/run_experiment_a.sh
```

What it does:

1. Reuses `data/coco2014_subset_50k/5k_ECSD.json` if it exists. If not, it calls the existing `experiments/5k_ecsd/generate_ecsd.py` and writes the legacy output under `outputs/5k_ecsd_next/legacy/`.
2. Calls the original `experiments/5k_ecsd/ecsd_core.py::build_candidates()` through the existing module to measure the legacy candidate pool and selected five captions.
3. Reports candidate counts, type distribution, heuristic invalidity, duplicate ratio, lexical unsupported-content proxy, token lengths, category distribution, per-image category coverage, and selection frequency.
4. Builds a new candidate pool from Raw Captions only. Every candidate stores source caption IDs and exact character spans.
5. Runs B4 once to materialize `data/coco2014_subset_50k/5k_ECSD_v2.json`.
6. Exports a fixed-seed 100-image annotation CSV and offline HTML viewer.

### Stage A outputs

```text
outputs/5k_ecsd_next/
├── legacy/
├── candidate_quality/
│   ├── summary.json
│   ├── summary.csv
│   ├── per_candidate.csv
│   └── per_image_category_coverage.csv
├── candidates/
│   ├── ecsd_v2_candidates.jsonl
│   └── candidate_pool_summary.json
├── selection_traces/
│   ├── B4_selection_trace.jsonl
│   └── generation_metadata.json
└── manual_audit/
    ├── manual_annotation.csv
    ├── manual_audit_viewer.html
    └── ANNOTATION_GUIDELINES.md
```

The CSV has blank human-label columns by design. Fill them manually. Do not interpret the rule-based classifier as ground truth. After annotation:

```bash
python experiments/5k_ecsd_next/candidate_quality/analyze_manual_audit.py \
  --annotations outputs/5k_ecsd_next/manual_audit/manual_annotation.csv \
  --output-dir outputs/5k_ecsd_next/manual_audit
```

The resulting summary separates all candidates from selected candidates. It calculates validity, groundedness, completeness, category-label accuracy, and redundancy only from labels that were actually filled.

## 6. Stage B · Selection Ablation

Run:

```bash
bash experiments/5k_ecsd_next/scripts/run_experiment_b.sh
```

All four methods consume the exact same `ecsd_v2_candidates.jsonl` pool. The shared candidate filter and anchor are fixed across B1-B4. The only changing component is the scoring objective after the anchor.

- **B1**: incremental weighted semantic fact coverage.
- **B2**: mean CLIP distance from selected captions.
- **B3**: 0.5 coverage + 0.5 CLIP diversity.
- **B4**: 0.4 coverage + 0.4 CLIP diversity + 0.2 role novelty.

The script writes the four requested caption JSON files, selection traces, per-image metrics, aggregate metrics, and plots. The `5k_ECSD_v2.json` file is checked against B4; if it already exists and differs, the script stops unless `OVERWRITE=1` is explicitly set.

### Stage B outputs

```text
data/coco2014_subset_50k/
├── 5k_ECSD_v2.json
├── 5k_ECSD_B1_Coverage.json
├── 5k_ECSD_B2_Diversity.json
├── 5k_ECSD_B3_Coverage_Diversity.json
└── 5k_ECSD_B4_Coverage_Diversity_Role.json

outputs/5k_ecsd_next/
├── selection_traces/
├── ablation_metrics/
│   ├── per_image_metrics.csv
│   ├── summary.csv
│   └── metadata.json
└── ablation_analysis/
    ├── REPORT.md
    ├── ranking_summary.json
    └── metric plots
```

CISD and semantic coverage remain separate metrics. The lexical coverage metric is a proxy based on Raw Caption facts, not a human-verified concept inventory.

## 7. Stage C · Multi-Seed OVSS

### 7.1 Apply the backward-compatible seed patch

The current repository version hardcodes `set_seed(123)` inside `train_and_eval()`. The existing `do_train()` already accepts a seed. Apply the minimal patch:

```bash
bash experiments/5k_ecsd_next/patches/apply_seed_patch.sh
python train.py --help | grep seed
```

The patch adds `--seed` (default `123`) and deterministic DataLoader worker seeding. It does not change the model, loss, training configuration, or evaluation logic. The original default seed remains 123.

### 7.2 Prepare four arms on one shared split

Run the preparation-only mode first:

```bash
RUN_TRAIN=0 RUN_EVAL=0 \
bash experiments/5k_ecsd_next/scripts/run_experiment_c.sh
```

It assembles Raw, Rich, Rich-Diversity, and ECSD v2 from the same DINO features and one fixed 4,500/500 split. It then calls the original `text_features_extraction.py` for CLIP `ann_feats` and audits all four arms.

The split seed is `123` by default and is independent of the training seeds. The same `val_ids.json` is used for all caption arms and all training seeds.

### 7.3 Train 12 runs

```bash
bash experiments/5k_ecsd_next/multiseed/run_train.sh
```

This runs:

- Raw × seeds 123, 456, 789
- Rich × seeds 123, 456, 789
- Rich-Diversity × seeds 123, 456, 789
- ECSD v2 × seeds 123, 456, 789

The original `configs/vitb_mlp_infonce.yaml` is used unchanged. Each checkpoint is unique:

```text
weights/vitb_mlp_infonce_raw_seed123.pth
weights/vitb_mlp_infonce_rich_seed123.pth
weights/vitb_mlp_infonce_rich_diversity_seed123.pth
weights/vitb_mlp_infonce_ecsdv2_seed123.pth
```

The same naming convention applies to seeds 456 and 789. Existing checkpoints are never overwritten unless `OVERWRITE=1` is supplied.

### 7.4 Evaluate OVSS with PAMR disabled

```bash
bash experiments/5k_ecsd_next/multiseed/run_seg_eval.sh
```

The script uses the repository's existing evaluation configs, each of which explicitly sets `pamr: false`. It evaluates VOC20, Pascal Context 59, COCO-Stuff, and ADE20K for every checkpoint. Existing benchmark results for a checkpoint are skipped; set `OVERWRITE=1` to intentionally rerun all benchmarks for those checkpoints.

### 7.5 Aggregate

```bash
python experiments/5k_ecsd_next/multiseed/aggregate_results.py \
  --root outputs/5k_ecsd_next/multiseed \
  --weights-dir weights \
  --seg-results-dir segmentation_results \
  --metrics-summary outputs/5k_ecsd_next/ablation_metrics/summary.csv
```

Outputs:

- `per_run_summary.csv`
- `aggregate_summary.csv`
- `learning_curves.csv` and `convergence_summary.csv`
- `training_stability.csv`
- `cisd_coverage_ovss_join.csv`
- `aggregation_metadata.json`
- `CORRELATION_NOTE.md`

The aggregate reports mean, population standard deviation, best, worst, and relative differences against Raw, Rich, and Rich-Diversity. Three seeds are used to describe stability, not to claim statistical significance.

## 8. One-command orchestration

Stage A and B can be run together:

```bash
bash experiments/5k_ecsd_next/scripts/run_all.sh
```

The script pauses before GPU work. To continue to all 12 training runs and up to 48 segmentation evaluations:

```bash
CONFIRM_FULL_RUN=YES \
bash experiments/5k_ecsd_next/scripts/run_all.sh
```

## 9. Safe overwrite policy

- Caption outputs and checkpoints: refuse overwrite by default.
- Candidate pool: refuse overwrite by default.
- Manual annotation CSV/HTML: refuse overwrite by default because they may contain human labels.
- Evaluation metrics: resume missing benchmarks; `OVERWRITE=1` removes the per-checkpoint metric JSON before rerunning.
- `OVERWRITE=1` should be used only when intentionally regenerating the corresponding artifacts.

## 10. Important interpretation limits

1. Lexical role classification is a transparent heuristic, not semantic ground truth.
2. Unsupported-content is estimated by comparing normalized candidate tokens against source-span tokens. It does not prove visual grounding.
3. Candidate validity flags are surface-form heuristics. Human annotation is required to estimate actual grammatical and semantic quality.
4. CISD measures text embedding dispersion, not concept coverage.
5. Semantic coverage measures lexical fact coverage from Raw Captions, not image-grounded object recall.
6. OVSS differences across three seeds are descriptive. Do not claim statistical significance.
7. No GPU training or segmentation evaluation was executed during package construction.
