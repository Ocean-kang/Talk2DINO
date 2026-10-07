# CaptionsConcept_Completion

This folder implements three isolated Talk2DINO experiments for testing whether CoCu-style concepts are useful **after semantic completion**, rather than being inserted as isolated `a photo of a {concept}` training samples.

It follows the repository's existing experiment pattern: **no file under `src/`, `train.py`, or `configs/` is modified**. Each experiment only builds a new pre-extracted `train_*.pth`, then reuses the original Talk2DINO training pipeline.

## Layout

```text
experiments/CaptionsConcept_Completion/
├── README.md
├── common/
│   └── completion_utils.py
├── inputs/
│   ├── clip_vitb16_5plus3_manifest.json
│   └── local_cocu_compare_5plus3_manifest.json
├── A_raw_caption_anchored/
│   ├── build.py
│   ├── run.sh
│   ├── clip/config.json
│   └── local_talk2dino/config.json
├── B_filtered_semantic_residual/
│   ├── build.py
│   ├── run.sh
│   ├── clip/config.json
│   └── local_talk2dino/config.json
└── D_concept_set_supplementary/
    ├── build.py
    ├── run.sh
    ├── clip/config.json
    └── local_talk2dino/config.json
```

The two large concept manifests are stored **once** in `inputs/` to avoid six duplicated copies. Every A/B/D experiment has explicit `clip/` and `local_talk2dino/` configurations pointing to those inputs.

## Common prerequisites

Run from the Talk2DINO repository root. These original files must already exist:

```text
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
configs/vitb_mlp_infonce.yaml
```

Use the same environment as normal Talk2DINO training, including the repository's `import clip` dependency.

## Scheme A — Raw Caption Anchored Concept Completion

For each image:

1. keep the original five COCO captions;
2. encode each curated concept with CLIP ViT-B/16;
3. compare the concept against the five existing raw-caption `ann_feats`;
4. assign the concept to the most similar raw caption;
5. append the concept as a controlled natural-language completion;
6. re-encode the resulting five captions.

The output remains **exactly 5 captions per image**, so this is the cleanest experiment for testing semantic completeness without increasing sample count.

Build:

```bash
bash experiments/CaptionsConcept_Completion/A_raw_caption_anchored/run.sh build clip 0
bash experiments/CaptionsConcept_Completion/A_raw_caption_anchored/run.sh build local 1
```

Train:

```bash
bash experiments/CaptionsConcept_Completion/A_raw_caption_anchored/run.sh train clip 0
bash experiments/CaptionsConcept_Completion/A_raw_caption_anchored/run.sh train local 1
```

Weights:

```text
weights/vitb_mlp_infonce_ccA_clip.pth
weights/vitb_mlp_infonce_ccA_local_talk2dino.pth
```

## Scheme B — Filtered Semantic Residual Completion

B adds a semantic-residual filter before A-style fusion. Each concept is checked using:

- extraction confidence relative to the best concept of that image;
- lexical redundancy with the five raw captions;
- CLIP semantic consistency with the raw-caption set;
- semantic outlier filtering;
- high-similarity redundancy filtering.

Only retained concepts are matched back to raw captions. Images with no accepted residual concept simply keep their original five captions. The training set therefore remains **exactly 5 captions per image**.

Default thresholds are intentionally source-scale independent because visual confidence is normalized **within each image**:

```text
min_visual_ratio   = 0.85
semantic_floor     = 0.18
consistency_margin = 0.20
redundancy_ceiling = 0.90
max_concepts       = 3
```

The generated `*_audit.jsonl` and `*_summary.json` should be inspected before training. If B removes too many or too few concepts, change thresholds in `run.sh` and rebuild; do not tune them after looking at segmentation test results.

Build:

```bash
bash experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/run.sh build clip 0
bash experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/run.sh build local 1
```

Train:

```bash
bash experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/run.sh train clip 0
bash experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/run.sh train local 1
```

Weights:

```text
weights/vitb_mlp_infonce_ccB_clip.pth
weights/vitb_mlp_infonce_ccB_local_talk2dino.pth
```

## Scheme D — Concept-Set Supplementary Caption

D preserves all five raw captions unchanged. It turns the curated concept set into **one contextual supplementary caption**:

1. deduplicate concepts;
2. keep concepts with extraction score >= 0.85 of that image's best concept (Top-3 maximum);
3. if all are removed, fall back to the highest-ranked concept so every image stays strict 5+1;
4. CLIP-encode the retained concepts and average them into a concept-set centroid;
5. choose the raw caption most semantically compatible with that centroid as the scene anchor;
6. create one natural supplementary sentence from the anchor + concept set;
7. encode that one new sentence.

The output is therefore **strictly 5 raw + 1 supplementary = 6 samples per image**.

Build:

```bash
bash experiments/CaptionsConcept_Completion/D_concept_set_supplementary/run.sh build clip 0
bash experiments/CaptionsConcept_Completion/D_concept_set_supplementary/run.sh build local 1
```

Train:

```bash
bash experiments/CaptionsConcept_Completion/D_concept_set_supplementary/run.sh train clip 0
bash experiments/CaptionsConcept_Completion/D_concept_set_supplementary/run.sh train local 1
```

Weights:

```text
weights/vitb_mlp_infonce_ccD_clip.pth
weights/vitb_mlp_infonce_ccD_local_talk2dino.pth
```

## Expected output files

```text
outputs/CaptionsConcept_Completion/
├── A/
│   ├── train_clip.pth
│   ├── train_local_talk2dino.pth
│   ├── ccA_clip_audit.jsonl
│   ├── ccA_clip_summary.json
│   ├── ccA_local_talk2dino_audit.jsonl
│   └── ccA_local_talk2dino_summary.json
├── B/
│   └── ...
└── D/
    └── ...
```

## Recommended experimental comparison

Use the same seed/config/validation set for all models and compare against the original Talk2DINO baseline plus your previous `5 raw + 3 isolated concept captions` result.

The key interpretation is:

- **A > raw baseline**: concept semantic completion itself helps, even at fixed sample count.
- **B > A**: concept usefulness depends on identifying true missing semantic residuals rather than inserting every prediction.
- **D > previous 5+3 isolated concepts**: the concept information was useful, but Talk2DINO needed scene-level language context instead of isolated noun prompts.
- **Local > CLIP within the same scheme**: local/head-aware concept extraction provides better semantic additions than global CLIP curation.
