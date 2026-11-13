# RA-C2C: Role-Aware Concept-to-Caption Composition

Version: `0.1.0`

A deterministic, non-LLM preprocessing experiment that converts existing per-image Concepts into exactly five role-aware captions for Talk2DINO.

## 1. What this package tests

> Given the same Concept candidates, does assigning them to semantic roles with evidence-gated fallbacks produce better Talk2DINO supervision than ordinary Concept-to-caption assignment?

This v0.1 deliberately isolates the composition algorithm:

- Reuses existing F5, CoCu, or Bootstrap candidate manifests. It does not re-mine concepts.
- Extracts action, attribute, and spatial-relation evidence from the five raw COCO captions using small deterministic lexicons and phrase matching.
- Normalizes ranks separately inside each source. F5, CoCu, and Bootstrap raw scores are never directly added because they are not calibrated to a shared scale.
- Uses bounded Beam Search to choose five caption slots, rewarding new concept coverage and penalizing lexical overlap.
- Emits a spatial-relation caption only when the same object-relation-object triple is supported by at least `--min-relation-support` distinct raw captions. Default: 2.
- Uses residual concepts or untouched raw captions when a role lacks evidence.
- Re-encodes the five final captions with the existing CLIP ViT-B/16 checkpoint. The resulting `ann_feats` are 512-D and compatible with the original Talk2DINO dataset loader.
- Does not modify `src/`, `train.py`, `configs/`, the original loss, DINO image features, or segmentation evaluation code.

**Scope boundary:** this first package does not recompute P1 Concept Propagation or train a teacher. If a candidate manifest already contains `grounding_score` or `p1_score`, that evidence is consumed by the confidence fusion. This keeps the first comparison focused on composition. A separate P1 export can be added after the F5-only composition baseline is validated.

## 2. Where to put the ZIP

Unzip into the Talk2DINO repository root. The final layout is:

```text
Talk2DINO/
├── experiments/
│   ├── role_aware_concept_composition/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── io_utils.py
│   │   ├── build_ra_c2c.py
│   │   ├── tests/
│   │   └── ...
│   ├── f5_ponytail/
│   ├── local_cocu_talk2dino_new_train/
│   └── ...
├── src/
├── train.py
├── configs/
└── outputs/
```

From the repository root:

```bash
unzip /path/to/role_aware_concept_composition.zip
```

The ZIP creates only `experiments/role_aware_concept_composition/`. It does not overwrite existing experiment or core files.

## 3. Required input

### Talk2DINO training cache

Default:

```text
outputs/coco2014_b14/train.pth
```

Expected structure:

- `images`: each image record has `id` and the original DINO features.
- `annotations`: at least five annotations per image.
- Each of the first five annotations has `image_id`, `caption`, and `ann_feats`.
- `ann_feats` must contain 512 values, the original CLIP ViT-B/16 text feature.

The builder keeps the first five annotations per image, preserving annotation IDs and all existing annotation fields. Image-side DINO features are copied unchanged. The builder refuses to write into a non-empty output directory; use a fresh run directory for every experiment.

### Concept candidates

Pass one or more sources using repeatable `--concept-source NAME=PATH`.

**Start with F5 only:**

```bash
--concept-source f5=outputs/f5_ponytail/run1/audit.jsonl
```

**Then test F5 + CoCu:**

```bash
--concept-source f5=outputs/f5_ponytail/run1/audit.jsonl
--concept-source cocu=outputs/local_cocu_talk2dino_new_train/manifest.json
```

**Optional Bootstrap:**

```bash
--concept-source bootstrap=/path/to/bootstrap_candidates.json
```

Supported repository-style input schemas:

1. F5 `audit.jsonl`: one JSON object per image with `image_id` and `candidates: [{concept, score/target_score, best_head, ...}]`.
2. CoCu manifest: `{"meta": {...}, "images": [{"image_id": 1, "curated": [...]}]}`.
3. Local-CoCu manifest: `{"meta": {...}, "images": {"1": {"selected_concepts": [...]}}}`.
4. Image-ID keyed JSON: `{"1": {"concepts": [...]}, "2": {"candidates": [...]}}`.
5. Bootstrap manifest using `concepts`, `candidates`, `curated`, or `selected_concepts`.

A candidate may be a string or a dictionary containing `concept` / `name` / `text`, and optional `score`, `rank`, `best_head`, `grounding_score`, or `p1_score`.

All source manifests must refer to the same COCO train split as `train.pth`. By default, incomplete source coverage is an error. `--missing-policy keep_raw` explicitly keeps the original five captions for images missing from all candidate sources.

### Gallery

The default is the existing repository gallery:

```text
experiments/local_cocu/MSCOCO14_train_gallery.json
```

It is used for noun-phrase matching in raw captions. If the file is absent, the code continues with terms from candidate sources, but raw-object extraction may be less complete.

## 4. Execution: two-stage workflow

Activate the existing Talk2DINO environment and run commands from the repository root. No new package installation is needed. No model or dataset download is initiated by this package.

### Stage A: tests and 32-image composition smoke test

This stage does not load CLIP and does not create a training PTH.

```bash
python -m unittest discover -s experiments/role_aware_concept_composition/tests -v

python -m experiments.role_aware_concept_composition.build_ra_c2c \
  --base-train outputs/coco2014_b14/train.pth \
  --concept-source f5=outputs/f5_ponytail/run1/audit.jsonl \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --output-dir outputs/role_aware_concept_composition/smoke_plan \
  --limit 32 \
  --plan-only \
  --missing-policy keep_raw
```

Inspect:

```text
outputs/role_aware_concept_composition/smoke_plan/
├── generated_captions.json
├── composition_audit.jsonl
└── summary.json
```

Manually review at least 20 images before a full build:

- C1: main object, plus an action only if the raw captions support it.
- C2: attribute and appearance, otherwise a secondary-object fallback.
- C3: a distinct local/secondary object.
- C4: spatial relation only if at least two raw captions support the same relation triple. Otherwise residual-object fallback.
- C5: scene/background, otherwise a remaining high-confidence concept.

### Stage B: build the Talk2DINO training PTH

The full build requires a **local existing CLIP ViT-B/16 checkpoint file**. The code passes the local file path to `clip.load`, not the model name. It will not download weights.

Find an already cached checkpoint:

```bash
find ~/.cache/clip -maxdepth 2 -type f
```

Set the exact `.pt` path:

```bash
export CLIP_CKPT=/absolute/path/to/ViT-B-16.pt
test -f "$CLIP_CKPT"
```

Build F5-only first:

```bash
python -m experiments.role_aware_concept_composition.build_ra_c2c \
  --base-train outputs/coco2014_b14/train.pth \
  --concept-source f5=outputs/f5_ponytail/run1/audit.jsonl \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --clip-checkpoint "$CLIP_CKPT" \
  --device cuda \
  --batch-size 512 \
  --output-dir outputs/role_aware_concept_composition/f5_only \
  --missing-policy error
```

Build the hybrid version separately:

```bash
python -m experiments.role_aware_concept_composition.build_ra_c2c \
  --base-train outputs/coco2014_b14/train.pth \
  --concept-source f5=outputs/f5_ponytail/run1/audit.jsonl \
  --concept-source cocu=outputs/local_cocu_talk2dino_new_train/manifest.json \
  --gallery experiments/local_cocu/MSCOCO14_train_gallery.json \
  --clip-checkpoint "$CLIP_CKPT" \
  --device cuda \
  --batch-size 512 \
  --output-dir outputs/role_aware_concept_composition/f5_cocu_hybrid \
  --missing-policy keep_raw
```

If your CoCu manifest has another filename, replace only that path. Never use a val/test candidate manifest to construct train captions.

### Stage C: original Talk2DINO training

The generated training file is:

```text
outputs/role_aware_concept_composition/f5_only/train_ra_c2c.pth
```

Reuse the original training entrypoint and change only the training PTH and run name. Example:

```bash
python train.py \
  --model_config configs/vitb_mlp_infonce.yaml \
  --train_dataset outputs/role_aware_concept_composition/f5_only/train_ra_c2c.pth \
  --val_dataset outputs/coco2014_b14/val.pth \
  --feature_name disentangled_self_attn \
  --text_features ann_feats \
  --name_pedix ra_c2c_f5
```

The current repository's `train.py` reads `--model_config` as a YAML file path, so pass `configs/vitb_mlp_infonce.yaml` rather than only the config basename. It creates a fresh projector and uses the original Talk2DINO training loop. The expected weight name is `weights/vitb_mlp_infonce_ra_c2c_f5.pth`, but verify the actual save path printed by your run.

For segmentation, reuse the same evaluation script and settings as the F5 comparator. Keep PAMR, datasets, evaluation splits, and seeds identical.

## 5. Recommended first experiment matrix

| Run | Candidate source | Caption construction |
|---|---|---|
| Raw baseline | None | Original five captions |
| F5 baseline | F5 | Existing F5 assignment |
| RA-C2C-F5 | Same F5 candidate file | Role-aware composition |
| RA-C2C-Hybrid | Same F5 + CoCu | Role-aware composition |

The key comparison is **F5 baseline vs RA-C2C-F5**. This holds the candidate source fixed and isolates the composition strategy.

Do not add Bootstrap, P1 recomputation, architecture changes, loss changes, or PAMR changes in this first comparison.

## 6. Algorithm summary

### Source-local confidence fusion

Each source is normalized independently to rank scores in `[0,1]`. This avoids treating F5 and CoCu raw scores as calibrated probabilities.

For a candidate concept:

- `rank_score`: weighted average of its normalized rank within each source.
- `source_agreement`: fraction of supplied sources that contain the concept.
- `raw_support`: fraction of the five raw captions that mention it.
- `grounding_score`: optional precomputed P1/grounding score.

Without grounding metadata:

```text
confidence = 0.62 * rank_score
           + 0.20 * source_agreement
           + 0.18 * raw_support
```

With grounding metadata, 0.15 weight is allocated to grounding and the other terms are adjusted.

### Five roles

- **C1 Main:** raw-caption noun evidence plus candidate confidence. Attach an action only when an action token and nearby subject/target phrases are found.
- **C2 Attribute:** match a small color/size/material/texture lexicon to a nearby noun. If absent, fall back to a secondary object.
- **C3 Local:** select a high-confidence object distinct from the C1 main object and concepts already assigned.
- **C4 Relation:** use explicit spatial prepositions between two object mentions. The same relation triple must occur in multiple raw captions by default. Otherwise use a residual object, never invent a relation.
- **C5 Scene:** use a scene/environment concept. If absent, use a remaining high-confidence concept.

### Adaptive Beam Search

A beam of at most `--beam-width` partial caption sets is expanded through C1 to C5. Each assignment rewards role confidence and new concept coverage, and penalizes lexical overlap and repeated concepts. Sorting and tie-breaking are deterministic.

Exactly five caption slots are retained. On fallback, the slot's actual semantic role is recorded separately from its slot ID.

## 7. Outputs and audit

```text
outputs/role_aware_concept_composition/f5_only/
├── generated_captions.json
├── composition_audit.jsonl
├── summary.json
└── train_ra_c2c.pth
```

Each training annotation preserves its original annotation ID, image ID, and other metadata, and adds:

- `ra_c2c_role`
- `ra_c2c_assigned_role`
- `ra_c2c_concepts`
- `ra_c2c_fallback`
- `ra_c2c_fallback_reason`

All five final captions are re-encoded into 512-D `ann_feats`. The original image-side `disentangled_self_attn` is copied unchanged.

## 8. Limitations and next extension

- The lexicons are intentionally small and auditable. Unlisted actions, attributes, and relations will be missed.
- Relation evidence is extracted from raw text. Repeated textual support is not independent visual verification. The code does not infer depth, `holding`, or `riding` from 2D proximity.
- P1 is not recomputed in v0.1. The candidate adapter can consume existing `grounding_score` / `p1_score` fields. The next extension should export compact per-image/per-concept P1 statistics from `Concept_Guided_Head_Patch_Visualization`, then compare with and without those statistics.
- Bootstrap is an optional candidate source, not a teacher in student training. Keep the student freshly initialized and do not pass teacher features into the student training loop.
- This is a caption data-construction algorithm, not a new Talk2DINO model architecture.

## 9. Reproducibility checklist

- [ ] Candidate manifests and base PTH use the same COCO train image pool.
- [ ] Source image-ID coverage is checked and recorded.
- [ ] Keep the exact command and local CLIP checkpoint path.
- [ ] Review at least 200 generated images before full training.
- [ ] Compare F5 and RA-C2C-F5 with identical seeds, config, validation data, and PAMR.
- [ ] Report mean ± std over five seeds before claiming an improvement.
