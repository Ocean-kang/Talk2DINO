# ECSD v3: deterministic candidate deduplication, shared-anchor B1-B4, and final-output metrics

This is an **independent v3 experiment package** for Talk2DINO. It creates `experiments/5k_ecsd_v3/` and uses separate output paths under `outputs/5k_ecsd_v3/` and separate caption filenames `5k_ECSD_v3_*.json`. It does not overwrite or modify `experiments/5k_ecsd/`, `experiments/5k_ecsd_next/`, v2 captions, existing checkpoints, or old manual annotations.

The v3 scope is deliberately A/B only. It does not launch GPU training or segmentation evaluation.

## What changed from v2

1. **Strict Clause candidates**: a Clause must be a strict source-caption subspan. A complete Raw Caption is never duplicated under the Clause type.
2. **Text deduplication**: candidates are deduplicated by lowercase alphanumeric surface text while preserving stopwords and word order. Duplicate source evidence is merged into `source_spans` and `source_caption_ids`.
3. **Explicit type priority**: `Original > Clause > Noun Phrase`. If the same visible text is produced by multiple extractors, the representative type is chosen by this priority, while all origins/spans remain recorded.
4. **Stable candidate IDs**: `candidate_id = image_id + SHA1(canonical surface text) prefix`. IDs are deterministic for the same image/text and unique within the image.
5. **One anchor computed once per image**: B1-B4 receive the same precomputed anchor ID. Candidate ordering and tie-breaking are deterministic.
6. **Manual audit uses IDs**: `Selected` is based only on exact `candidate_id` membership in the B4 trace, never on caption text.
7. **Safe writes**: caption JSON and traces are written to temporary files and atomically replaced. Existing invalid JSON is never silently overwritten. `--overwrite` preserves a timestamped backup before replacement.
8. **Metrics use final artifacts**: CISD is recomputed from normalized CLIP ViT-B/16 embeddings of the final JSON caption strings. The evaluator cross-checks final JSON against exact trace order and candidate IDs.
9. **Embedding cache**: a hash-addressed `.npz` cache avoids re-encoding the same text set. No model download is performed.

## 1. Placement

From the Talk2DINO repository root:

```bash
unzip 5k_ecsd_v3.zip -d .
```

The archive creates:

```text
Talk2DINO/
└── experiments/
    └── 5k_ecsd_v3/
        ├── candidate_quality/
        ├── selection_ablation/
        ├── utils/
        ├── configs/default.json
        ├── scripts/
        ├── tests/
        └── README.md
```

Keep the old `experiments/5k_ecsd_next/` directory. v3 is not an in-place patch.

## 2. Prerequisites

Activate the existing Talk2DINO environment. No new model or NLP dependency is introduced. The v3 scripts reuse:

- Python standard library
- NumPy
- the existing Talk2DINO CLIP helper at `experiments/5k_ecsd/clip_utils.py`
- the existing local CLIP ViT-B/16 weight

Required local files:

```bash
test -f data/coco2014_subset_50k/5k_rich_diversity.json
test -f experiments/5k_ecsd/clip_utils.py
test -f "$HOME/.cache/clip/ViT-B-16.pt"
```

The CLIP cache path can be changed in `configs/default.json` or via `CLIP_CACHE`. The script checks for an existing `ViT-B-16.pt` before calling the original CLIP helper. It does **not** download weights.

## 3. Validate the package first

```bash
bash experiments/5k_ecsd_v3/scripts/validate_package_v3.sh
```

This runs Python compilation, unit tests, and `bash -n` over v3 shell scripts. It does not load CLIP, use the GPU, or access COCO images.

## 4. Experiment A: rebuild the candidate pool

```bash
bash experiments/5k_ecsd_v3/scripts/run_experiment_a_v3.sh
```

The builder reads only `raw_captions` from `5k_rich_diversity.json` and writes:

```text
outputs/5k_ecsd_v3/
├── candidates/
│   ├── ecsd_v3_candidates.jsonl
│   ├── candidate_pool_summary_v3.json
│   └── candidate_pool_per_image_v3.csv
└── logs/build_candidate_pool_v3.log
```

Important candidate-pool fields:

- `candidate_id`: stable image/text identifier
- `candidate_type`: one of `original`, `clause`, `noun_phrase`
- `candidate_origins`: all extraction types merged into this unique surface text
- `source_caption_ids`: all source captions supporting this text
- `source_spans`: exact character offsets and source text for each origin
- `facts`, `categories`, `category_rules`, `quality_flags`

The summary distinguishes raw extraction records from unique candidates after deduplication and reports the number of full-caption Clause candidates rejected.

If the v3 pool already exists, the script refuses to replace it. To intentionally rebuild it:

```bash
OVERWRITE=1 bash experiments/5k_ecsd_v3/scripts/run_experiment_a_v3.sh
```

The previous v3 pool and summary are preserved as timestamped `.bak.*` files.

## 5. Experiment B: generate B1-B4

```bash
bash experiments/5k_ecsd_v3/scripts/run_experiment_b_v3.sh
```

This step encodes the unique candidate strings using the existing CLIP ViT-B/16 model, then computes one anchor per image before running any selection arm. All B1-B4 use the same filtered candidate list and anchor.

B1-B4 objectives:

| Method | Objective |
|---|---|
| B1 | Coverage only |
| B2 | CLIP diversity only |
| B3 | 0.5 Coverage + 0.5 Diversity |
| B4 | 0.4 Coverage + 0.4 Diversity + 0.2 Role Novelty |

The weights are read from `configs/default.json`.

### Caption outputs

The default data directory receives new, versioned files:

```text
data/coco2014_subset_50k/
├── 5k_ECSD_v3_B1_Coverage.json
├── 5k_ECSD_v3_B2_Diversity.json
├── 5k_ECSD_v3_B3_Coverage_Diversity.json
├── 5k_ECSD_v3_B4_Coverage_Diversity_Role.json
└── 5k_ECSD_v3.json                 # B4 alias, field ecsd_v3_captions
```

These do not overwrite `5k_ECSD_v2.json` or any v2 B1-B4 file.

### Traces, cache, and logs

```text
outputs/5k_ecsd_v3/
├── selection_traces/
│   ├── B1_selection_trace.jsonl
│   ├── B2_selection_trace.jsonl
│   ├── B3_selection_trace.jsonl
│   ├── B4_selection_trace.jsonl
│   └── generation_metadata.json
├── clip_cache/
│   └── clip_text_embeddings_<content-hash>.npz
├── integrity_audit.json
└── logs/
```

The incremental embedding store is model-specific and keyed by exact text strings. Stage B first stores candidate-pool embeddings; metric evaluation then reuses those vectors and encodes only any additional baseline caption strings.

### Safe overwrite behavior

By default, existing outputs are not overwritten. If a target JSON exists but is invalid, the script stops and leaves it untouched. To intentionally replace an existing v3 output:

```bash
OVERWRITE=1 bash experiments/5k_ecsd_v3/scripts/run_experiment_b_v3.sh
```

Existing files are copied to timestamped `.bak.*` backups before replacement. This applies to the v3 caption JSONs and traces. It does not automatically overwrite an existing manual annotation CSV.

## 6. Integrity validation

Generation automatically validates:

- source and candidate-pool image IDs are unique and identical
- candidate pool Raw Captions exactly match the source file
- candidate IDs and canonical surface texts are unique within each image
- each B1-B4 trace has exactly five unique selected candidate IDs per image
- each trace ID exists in the candidate pool and its text matches the pool
- each final caption JSON contains five non-empty captions per image
- final JSON caption order exactly equals the corresponding trace order
- B1-B4 anchor IDs are identical for every image
- `5k_ECSD_v3.json` equals B4 trace output

The report is written to `outputs/5k_ecsd_v3/integrity_audit.json`.

You can rerun validation separately:

```bash
python experiments/5k_ecsd_v3/selection_ablation/validate_generation_v3.py \
  --source data/coco2014_subset_50k/5k_rich_diversity.json \
  --candidate-pool outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl \
  --caption-dir data/coco2014_subset_50k \
  --trace-dir outputs/5k_ecsd_v3/selection_traces \
  --expected-images 5000
```

## 7. Recompute B1-B4 metrics from final JSON and Trace

The Stage B script runs this automatically. To run separately:

```bash
python experiments/5k_ecsd_v3/selection_ablation/evaluate_metrics_v3.py \
  --source data/coco2014_subset_50k/5k_rich_diversity.json \
  --candidate-pool outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl \
  --caption-dir data/coco2014_subset_50k \
  --trace-dir outputs/5k_ecsd_v3/selection_traces \
  --output-dir outputs/5k_ecsd_v3/ablation_metrics \
  --expected-images 5000
```

Outputs:

- `per_image_metrics_v3.csv`: one row per image, all baseline and B1-B4 metrics
- `summary_v3.csv` / `summary_v3.json`: mean, population standard deviation, median
- `pairwise_selection_consistency_v3.csv`: exact same selected-ID set count and average shared selected captions
- `selection_consistency_v3.json`: pairwise comparisons and anchor invariance
- `data_integrity_audit_v3.json`: schema/trace/source checks
- `metrics_metadata_v3.json`: exact metric definitions and CLIP cache metadata

### Metric definitions

- **CISD**: `1 - mean cosine similarity` over the ten unordered pairs of L2-normalized CLIP ViT-B/16 text embeddings. It is recomputed from final caption strings, not selector-internal scores.
- **Weighted fact coverage**: inverse-frequency weighted union of selected content tokens intersected with Raw Caption content tokens, divided by total Raw Caption fact weight.
- **Role coverage**: number of unique rule-classified roles among Entity / Attribute / Action / Relation / Context divided by five.
- **Weighted Jaccard redundancy**: mean weighted Jaccard overlap over the ten unordered selected-caption pairs.
- **Raw exact-match rate**: fraction of the five selected captions whose lowercase alphanumeric surface token sequence exactly matches one of the five Raw Captions.
- **Candidate type distribution**: Original / Clause / Noun Phrase counts from selected trace entries, not inferred from text.
- **Candidate validity proxy**: fraction without configured lexical surface-form flags. This is not human ground truth.

## 8. Manual audit and annotation migration

The v3 exporter uses the B4 Trace's exact candidate IDs:

```bash
python experiments/5k_ecsd_v3/candidate_quality/export_manual_audit_v3.py \
  --source data/coco2014_subset_50k/5k_rich_diversity.json \
  --candidate-pool outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl \
  --trace outputs/5k_ecsd_v3/selection_traces/B4_selection_trace.jsonl \
  --output-dir outputs/5k_ecsd_v3/manual_audit \
  --sample-size 100 --seed 42
```

It creates `manual_annotation_v3.csv`, `manual_audit_viewer_v3.html`, and annotation guidelines. Each sampled image must have exactly five selected rows. Duplicate visible text cannot accidentally mark another candidate ID as selected.

The exporter refuses to replace an existing v3 annotation CSV by default. If you already annotated v2, first create the v3 audit file, then optionally migrate labels:

```bash
python experiments/5k_ecsd_v3/candidate_quality/migrate_manual_annotations_v3.py \
  --old-csv outputs/5k_ecsd_next/manual_audit/manual_annotation.csv \
  --new-csv outputs/5k_ecsd_v3/manual_audit/manual_annotation_v3.csv \
  --output-csv outputs/5k_ecsd_v3/manual_audit/manual_annotation_v3_migrated.csv \
  --report outputs/5k_ecsd_v3/manual_audit/annotation_migration_report.json
```

Migration uses `(image_id, normalized candidate text)`, transfers only human label fields/notes, never transfers `selection_status`, and leaves conflicting categorical labels blank. The original v2 CSV is not modified. The uploaded v2 audit file currently has blank human label columns, so it contains no completed labels to migrate.

## 9. Configuration and reruns

Edit `experiments/5k_ecsd_v3/configs/default.json` for dataset size, file paths, candidate thresholds, and B1-B4 weights. Environment variables can override the main paths (`SOURCE_CAPTIONS`, `CANDIDATE_POOL`, `DATA_DIR`, `OUTPUT_ROOT`, `CLIP_CACHE`, `EMBED_CACHE`).

For a clean reproducible run:

```bash
bash experiments/5k_ecsd_v3/scripts/run_experiment_a_v3.sh
bash experiments/5k_ecsd_v3/scripts/run_experiment_b_v3.sh
```

Do not run the v2 and v3 scripts against the same output root. v3 uses its own root and versioned caption names.

## 10. Scope and limitations

- v3 does not run the 12/15 training jobs or OVSS evaluation. First establish trustworthy candidate and selection results.
- The original Talk2DINO `clip_utils.py` is not bundled; it must exist in the repository at `experiments/5k_ecsd/clip_utils.py`.
- Full COCO generation, CLIP encoding, and GPU execution were not performed while constructing this ZIP. See `VALIDATION_REPORT.md` for the checks actually run.
- Rule-based roles, lexical fact coverage, and validity flags are transparent proxies, not semantic ground truth.
