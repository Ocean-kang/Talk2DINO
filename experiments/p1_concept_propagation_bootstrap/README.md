# P1 Concept Propagation Bootstrap for Talk2DINO

Independent side-car experiment. It implements:

**Original Teacher → P1 Concept Propagation → Pseudo Head Feature Generation → Reinitialized Student → Original Talk2DINO Training**

It does not modify `src/`, `train.py`, the original configs, existing weights, or existing experiment folders. It reuses the current Talk2DINO environment and code interfaces.

## 1. Exact experimental definition

For one COCO image \(I_i\), let its original annotations/captions be
\(T_i=\{t_{i,1},\ldots,t_{i,C}\}\). Each complete raw caption is one concept unit.
There is **no noun parsing, phrase extraction, gallery retrieval, or caption rewriting**.

### Teacher seed response

The source PTH already contains original CLIP ViT-B/16 text features
`ann_feats` with dimension 512. The frozen original Teacher projector maps them:

\[
z_t=P_T(e_{\mathrm{CLIP}}(t))\in\mathbb R^{768}.
\]

For DINOv2 patch features \(X\in\mathbb R^{1024\times768}\):

\[
r_t(p)=\cos(X_p,z_t).
\]

This deliberately reuses the exact `concept_patch_cosine()` function in
`experiments/Concept_Guided_Head_Patch_Visualization/core.py`.

**Space-compatibility boundary:** the existing visualization code uses this
768-D comparison because `W_projector` maps CLIP text features into the DINO
feature dimension. This is the repository's existing operational convention,
not a theorem that the Teacher was explicitly trained for every individual
Patch-Caption pair. The seed-response distribution and its per-caption/head
diagnostics are saved so the assumption can be inspected. A 768-D shape check
alone is not treated as evidence of semantic correctness.

### P1 Patch-to-Patch propagation

The reused DINO extractor returns the last-block QKV and `x_norm_patchtokens`.
For each of the 12 heads, the existing implementation constructs:

\[
A_h=\operatorname{softmax}_{key}
\left(\frac{Q_h^P(K_h^P)^\top}{\sqrt{64}}\right)
\in\mathbb R^{1024\times1024}.
\]

The implementation scales Q by \(1/\sqrt{64}\) before multiplication.
Rows are query patches; columns are key patches. CLS and four register
tokens are excluded from the P1 patch matrix.

The response is propagated exactly as the existing P1 visualization does:

\[
r'_{h,t}=A_h r_t.
\]

### Pseudo Head Feature generation

For each head and each original caption:

\[
w_{h,t}(p)=\operatorname{softmax}_{p}(r'_{h,t}(p)/\tau),
\qquad \tau=0.1\ \text{by default}.
\]

\[
H^{P1}_{h,t}=
\frac{1}{1024}\sum_{p=1}^{1024}w_{h,t}(p)X_p
\in\mathbb R^{768}.
\]

The final cache row for each annotation is `[12,768]`. Pooling uses the same
`weighted_patch_features.mean(patch_dim)` convention as the original
`disentangled_self_attn` extraction in `dino_extraction_v2.py`.

The temperature is configurable and recorded in cache metadata. Do not compare
absolute feature magnitudes across different temperatures without checking
normalization; the original projector normalizes features when computing cosine
similarity.

## 2. Source audit and reused repository interfaces

The implementation was checked against the repository's current `main` code:

| Existing file | Reused behavior |
|---|---|
| `experiments/Concept_Guided_Head_Patch_Visualization/core.py` | `concept_patch_cosine`, `propagate_concept_scores` |
| `experiments/Concept_Guided_Head_Patch_Visualization/vision.py` | `DinoRelationExtractor`, final-block QKV hook, 448 input, ViT-B/14-reg |
| `experiments/Caption_Head_Visualization/data.py` | `caption_rows`, `resolve_image`, `select_images`, trusted PTH loading helper |
| `experiments/Caption_Head_Visualization/run.py` | `load_projector()` for the original Teacher |
| `src/model.py` | Original `ProjectionLayer` |
| `src/loss.py` | Original `ContrastiveLoss` |
| `src/train_util.py` | Original `do_train()` and seed behavior |
| `train.py` | Original train/validation feature-shape convention |
| `configs/vitb_mlp_infonce.yaml` | Original model and training hyperparameters |

The original `train.py` uses `disentangled_self_attn` `[12,768]` for training
and `avg_self_attn_out` `[768]` for validation. This package preserves that
input contract:

- Student training dataset: one sample per annotation; `image=[12,768]`.
- Student validation dataset: one sample per annotation; `image=[768]`, computed
  as the mean of the 12 P1 pseudo heads.
- Text input: original `ann_feats=[512]`.
- The original `ProjectionLayer` retains its configured `max_score` behavior.
- `src.train_util.do_train()` retains the original InfoNCE, optimizer,
  scheduler, DataLoader and validation code.

The Student is initialized with `ProjectionLayer.from_config()` after setting
the seed. **No Teacher weights are copied into the Student.** The Teacher is
used only during pseudo-feature construction.

## 3. Install the ZIP

Extract the ZIP into the Talk2DINO repository root:

```bash
cd /path/to/Talk2DINO
unzip /path/to/p1_concept_propagation_bootstrap.zip
```

Expected placement:

```text
Talk2DINO/
└── experiments/
    └── p1_concept_propagation_bootstrap/
        ├── __init__.py
        ├── core.py
        ├── dataset.py
        ├── prepare_features.py
        ├── train_student.py
        ├── run_smoke.sh
        ├── run_full.sh
        ├── eval_p1.sh
        ├── README.md
        └── tests/
            └── test_core.py
```

No existing repository file is replaced.

## 4. Required existing assets

The default paths follow the existing COCO14 Talk2DINO setup:

```text
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
weights/vitb_mlp_infonce.pth
configs/vitb_mlp_infonce.yaml
data/coco2014/train2014/
data/coco2014/val2014/
```

The source PTH must contain:

- `images`: records with `id` and `file_name`.
- `annotations`: records with `id`, `image_id`, `caption`, and `ann_feats`.
- `ann_feats` must be the original raw CLIP ViT-B/16 feature `[512]`.

The DINOv2 ViT-B/14-reg source and weights must already exist in the local
Torch Hub cache. The script invokes the repository's offline guard before
loading DINOv2; it will not download models or install packages.

Activate the existing Talk2DINO environment. No dependency installation or
version changes are included.

## 5. First run: smoke test

Set the COCO root to the parent directory of `train2014/` and `val2014/`:

```bash
conda activate talk2dino
cd /path/to/Talk2DINO
export COCO_ROOT=/path/to/data/coco2014
```

Run a two-image train/validation preprocessing smoke test and one Student epoch:

```bash
bash experiments/p1_concept_propagation_bootstrap/run_smoke.sh 0
```

This writes only to:

```text
outputs/p1_concept_propagation_bootstrap/smoke_cache/
outputs/p1_concept_propagation_bootstrap/smoke_student/
```

The smoke-test caches are intentionally partial and must not be used for the
full training run.

## 6. Full preprocessing and Student training

Run:

```bash
bash experiments/p1_concept_propagation_bootstrap/run_full.sh 0
```

Equivalent explicit commands:

### 6.1 Build train pseudo features

```bash
python -m experiments.p1_concept_propagation_bootstrap.prepare_features \
  --split train \
  --features outputs/coco2014_b14/train.pth \
  --image-root "$COCO_ROOT" \
  --teacher-weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --output outputs/p1_concept_propagation_bootstrap/cache/train \
  --device cuda \
  --temperature 0.1 \
  --limit 0
```

### 6.2 Build validation pseudo features

```bash
python -m experiments.p1_concept_propagation_bootstrap.prepare_features \
  --split val \
  --features outputs/coco2014_b14/val.pth \
  --image-root "$COCO_ROOT" \
  --teacher-weights weights/vitb_mlp_infonce.pth \
  --config configs/vitb_mlp_infonce.yaml \
  --output outputs/p1_concept_propagation_bootstrap/cache/val \
  --device cuda \
  --temperature 0.1 \
  --limit 0
```

### 6.3 Train the fresh Student

```bash
python -m experiments.p1_concept_propagation_bootstrap.train_student \
  --train-features outputs/coco2014_b14/train.pth \
  --val-features outputs/coco2014_b14/val.pth \
  --train-cache outputs/p1_concept_propagation_bootstrap/cache/train \
  --val-cache outputs/p1_concept_propagation_bootstrap/cache/val \
  --config configs/vitb_mlp_infonce.yaml \
  --output outputs/p1_concept_propagation_bootstrap/student_seed123 \
  --device cuda \
  --seed 123
```

The original config is used without changing its training values:

- Epochs: 100.
- Batch size: 128.
- Learning rate: `1e-4`.
- Loss: InfoNCE.
- Optimizer: Adam.
- Scheduler: linear, no warmup.
- Seed: 123.
- `save_best_model`: follows the original YAML (`False`).

The optimizer/scheduler/warmup defaults match `train.py`.

## 7. Resume and cache behavior

`prepare_features.py` writes a memory-mapped FP16 array, annotation/image ID
indices, and a per-image completion bitmap. It flushes features before marking
an image complete. Re-running the same command resumes completed images.

If the source PTH, Teacher checkpoint, projector config, image root, DINO
configuration, or temperature changes, the fingerprint check rejects the
existing cache. Use a new output directory for a different setting.

Useful partial run:

```bash
python -m experiments.p1_concept_propagation_bootstrap.prepare_features \
  --split train \
  --features outputs/coco2014_b14/train.pth \
  --image-root "$COCO_ROOT" \
  --output outputs/p1_concept_propagation_bootstrap/cache/train \
  --limit 20
```

Then run the same command with `--limit 0` to continue through the remaining
images. A partial cache is rejected by full Student training unless
`--allow-partial-cache` is explicitly supplied. That flag is only for smoke
tests.

Approximate feature storage is proportional to
`number_of_annotations × 12 × 768 × 2 bytes`. Full COCO14 train+val caches
can require around 10–11 GB, excluding source PTH and diagnostics.

## 8. Outputs

```text
outputs/p1_concept_propagation_bootstrap/
├── cache/
│   ├── train/
│   │   ├── metadata.json
│   │   ├── pseudo_features.npy
│   │   ├── annotation_ids.npy
│   │   ├── image_ids.npy
│   │   ├── done_images.npy
│   │   ├── caption_head_diagnostics.jsonl
│   │   ├── errors.jsonl
│   │   └── summary.json
│   └── val/
│       └── ...
└── student_seed123/
    ├── student_projector.pth
    ├── training_results.json
    └── model_config.yaml
```

The diagnostic JSONL stores the original caption, image/annotation IDs,
seed-response statistics, and per-head propagated-response/weight entropy/
maximum patch weight. It does not store the full `1024×1024` affinity matrix
for every image.

## 9. Segmentation evaluation

`DINOText` loads projector checkpoints from `weights/{model.proj_name}.pth`.
The trained Student is saved under `outputs/` to avoid touching existing
weights. The evaluation script copies it to a **new unique name** under
`weights/` and refuses to overwrite a different file.

Run the original four benchmark evaluations with the repository's PAMR
evaluation configs:

```bash
bash experiments/p1_concept_propagation_bootstrap/eval_p1.sh 0 123
```

This uses the same evaluation config pattern as
`experiments/CoCu_Bootstrap_Talk2dino/eval_all.sh`:
VOC20, Context59, ADE20K, and COCO-Stuff.

## 10. Quick tensor unit test

From the Talk2DINO repository root:

```bash
python -m unittest discover -s experiments/p1_concept_propagation_bootstrap/tests -v
```

This checks the pseudo-pooling tensor shapes and patch-weight normalization.
It does not replace the one-image DINO smoke test above.

## 10. Interpretation boundary

This experiment tests whether caption-conditioned P1 propagation creates
training features that help the Student projector align captions to local
DINO patch evidence. It does not prove that each Head is an object mask.
The P1 relation can propagate context/background evidence, and a full caption
can describe multiple objects or scene context. Inspect:

- `caption_head_diagnostics.jsonl` for response concentration and head diversity.
- Student train/validation InfoNCE losses.
- The four standard segmentation results against the original Teacher/Baseline.
- Multiple seeds before making a stability claim.

The experiment deliberately changes the image-side training representation
while retaining the original text supervision, Student architecture, loss,
and evaluation protocol.
