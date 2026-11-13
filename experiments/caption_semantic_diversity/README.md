# Talk2DINO: CLIP Caption Semantic Diversity Evaluation

This is a **standalone, evaluation-only** experiment for the four caption groups in
`5k_rich_diversity.json`:

- `raw_captions`
- `rich_captions`
- `rich_weaken_captions`
- `rich_diversity_captions`

It does not modify Talk2DINO training, segmentation evaluation, model configs, or existing weights.
It does not train a model, call an LLM, or download images/weights automatically.

## 1. Why this implementation reuses the repository

The repository's `text_features_extraction.py` already uses the OpenAI CLIP interface:

```python
model, _ = clip.load(model_name)
inputs = clip.tokenize(texts, truncate=True)
outputs = model.encode_text(inputs)
```

This experiment reuses the same `clip.load`, `clip.tokenize`, `encode_text`, and
`encode_image` APIs. It uses the same default text model, `ViT-B/16`, and frozen weights.
The original training/evaluation implementation is left untouched.

## 2. Placement

Unzip the package at the root of the Talk2DINO repository. The final layout should be:

```text
Talk2DINO/
├── text_features_extraction.py
├── src/
├── data/
│   └── coco2014_subset_50k/
│       └── 5k_rich_diversity.json
└── experiments/
    └── caption_semantic_diversity/
        ├── eval_caption_diversity.py
        ├── README.md
        ├── run_text_only.sh
        └── run_with_grounding.sh
```

If the JSON is stored elsewhere, pass its path through `--captions-json`.

## 3. Environment

Run this experiment in the same Talk2DINO environment used by the original repository.
The environment should already contain:

- PyTorch
- OpenAI CLIP package (`import clip`)
- NumPy
- Pillow
- tqdm
- matplotlib
- SciPy (used for Wilcoxon tests)

Quick check:

```bash
python -c "import torch, clip, numpy, PIL, tqdm, matplotlib, scipy; print('environment OK')"
```

### CLIP weights

The script intentionally **does not download weights**. It first looks for the cached
OpenAI CLIP ViT-B/16 file, normally:

```text
~/.cache/clip/ViT-B-16.pt
```

If your existing CLIP checkpoint is elsewhere, pass it explicitly:

```bash
--clip-checkpoint /absolute/path/to/ViT-B-16.pt
```

Accepted local checkpoint formats are an OpenAI CLIP TorchScript archive or a plain
OpenAI CLIP state dictionary. A Talk2DINO projector checkpoint is **not** a CLIP
backbone checkpoint and must not be passed here.

## 4. Stage A: text-only evaluation (recommended first run)

From the Talk2DINO repository root:

```bash
bash experiments/caption_semantic_diversity/run_text_only.sh
```

Or run directly:

```bash
python experiments/caption_semantic_diversity/eval_caption_diversity.py \
  --captions-json data/coco2014_subset_50k/5k_rich_diversity.json \
  --output-dir experiments/caption_semantic_diversity/outputs \
  --clip-model ViT-B/16 \
  --batch-size 256 \
  --bootstrap-iterations 10000 \
  --seed 42
```

If the JSON is not at the default path:

```bash
python experiments/caption_semantic_diversity/eval_caption_diversity.py \
  --captions-json /absolute/path/to/5k_rich_diversity.json \
  --output-dir experiments/caption_semantic_diversity/outputs
```

CPU fallback:

```bash
python experiments/caption_semantic_diversity/eval_caption_diversity.py \
  --captions-json data/coco2014_subset_50k/5k_rich_diversity.json \
  --device cpu --batch-size 32
```

## 5. Stage B: optional image-text grounding

This stage needs the actual COCO images. It is optional and is not required for the
text-only diversity metrics.

For COCO train2014, if the images are stored in
`/path/to/coco/train2014/COCO_train2014_000000000077.jpg`, run:

```bash
python experiments/caption_semantic_diversity/eval_caption_diversity.py \
  --captions-json data/coco2014_subset_50k/5k_rich_diversity.json \
  --output-dir experiments/caption_semantic_diversity/outputs_with_grounding \
  --image-dir /path/to/coco/train2014 \
  --image-pattern 'COCO_train2014_{image_id:012d}.jpg'
```

For a COCO validation directory, change the pattern:

```bash
--image-pattern 'COCO_val2014_{image_id:012d}.jpg'
```

If image filenames are not standard COCO names, provide `--image-manifest`.
It accepts either a COCO JSON containing an `images` list with `id` and `file_name`,
or a JSON mapping from image IDs to relative/absolute image paths.

Example:

```bash
--image-dir /path/to/coco/train2014 --image-manifest /path/to/instances_train2014.json
```

## 6. Metrics

For each image and each caption type, the script computes:

### Primary: CLIP Intra-set Semantic Diversity (CISD)

For the five normalized text embeddings `z_1 ... z_5`:

```text
CISD = 1 - mean(cos(z_i, z_j)), for all 10 unordered pairs i < j
```

Higher CISD means lower average pairwise semantic similarity. It is **not** a
bounded [0, 1] quality score: cosine similarity can be negative.

### Complementary metrics

- `nn_redundancy`: mean, over the five captions, of its maximum cosine similarity
  to any other caption. Lower means less nearest-neighbor duplication.
- `effective_rank_norm`: effective rank of the centered 5 x 512 feature matrix,
  normalized by the maximum possible centered rank (4). Higher suggests the set
  occupies more independent semantic directions.
- `grounding_mean`: average CLIP image-caption cosine similarity (only with images).
- `grounding_min`: minimum image-caption cosine similarity in the five captions
  (only with images).

Length diagnostics:

- `word_count`: regex-based word count.
- `clip_token_count`: CLIP BPE content-token count after the same 77-token truncation
  used by the CLIP encoder.
- `clip_truncated`: flags captions that reach CLIP's 75 content-token limit.

Length-adjusted diagnostic:

- `length_adjusted_ols.csv` fits a within-image fixed-effect regression:
  `CISD ~ caption_type + mean_CLIP_token_length + image_fixed_effect`.
  Its method coefficients estimate differences versus Raw after controlling for
  mean CLIP token length. Treat this as a diagnostic association, not causal proof.

## 7. Statistical comparison

All comparisons are paired by `image_id`, never by treating the 10 caption pairs
within an image as independent samples.

For Rich-Diversity vs Raw / Rich / Rich-Weaken, the script reports:

- Mean paired difference
- 95% paired-bootstrap confidence interval (default 10,000 resamples)
- Wilcoxon signed-rank p-value
- Holm-adjusted p-value within each metric's three comparisons

A positive difference means Rich-Diversity is higher for that metric. For
`nn_redundancy`, lower is generally preferable; interpret the sign accordingly.

## 8. Outputs

The output directory contains:

| File | Meaning |
|---|---|
| `per_image_metrics.csv` | One row per image x caption type; primary analysis table |
| `per_caption_metrics.csv` | One row per individual caption, with position and token lengths |
| `summary_metrics.csv` | Mean / standard deviation / median by caption type |
| `paired_statistics.csv` | Paired bootstrap CIs, Wilcoxon and Holm-adjusted p-values |
| `length_adjusted_ols.csv` | Within-image length-adjusted regression coefficients |
| `rich_diversity_position_similarity.csv` | 5 x 5 mean similarity matrix by Rich-Diversity position |
| `rich_diversity_position_similarity.npy` | Numeric matrix for downstream analysis |
| `run_config.json` | Input hash, model, weight source, settings and metric definitions |
| `figures/cisd_boxplot.png` | CISD distributions |
| `figures/nn_redundancy_boxplot.png` | Nearest-neighbor redundancy distributions |
| `figures/effective_rank_boxplot.png` | Effective-rank distributions |
| `figures/rich_diversity_position_similarity.png` | Five-position similarity heatmap |
| `figures/diversity_vs_grounding.png` | Only generated when image grounding is enabled |

## 9. Interpretation guardrails

1. Do not interpret higher CISD alone as better captions. A caption set can become
   diverse by introducing irrelevant or hallucinated content.
2. Read CISD together with nearest-neighbor redundancy and image grounding.
3. Effective rank is based on only five points, so it is a supporting descriptor,
   not a standalone proof of semantic coverage.
4. Global CLIP image-text similarity is only a proxy for visual grounding. It can
   under-represent small objects, local details, attributes, and spatial relations.
5. Raw captions are human-written and may already be diverse. Rich-Diversity does
   not need to beat Raw on every metric to be useful.
6. Keep the same CLIP checkpoint, image preprocessing, input records, and seed
   across comparisons.

## 10. Sanity checks performed by the script

- JSON must be a list with unique image IDs.
- Each of the four caption fields must contain exactly five non-empty strings.
- The script checks that CLIP ViT-B/16 returns 512-dimensional text embeddings.
- All metrics are computed per image before cross-image aggregation.
- Missing image files cause a clear error only when image grounding is requested.
