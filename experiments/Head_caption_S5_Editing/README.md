# Head–Caption S5 Editing: teacher-head Top/Bottom-10 + Policy + S5 + Original Talk2DINO

**Version 1.0 — independent Ponytail experiment, no modifications to original `src/`, `train.py`, or segmentation.**

## 0. Reproducibility / boundaries

This experiment is a *conservative hypothesis test*, not ground-truth concept label construction. Teacher/projection and CLIP ViT-B/16 are frozen during discovery. Edited caption embeddings are computed with the ORIGINAL CLIP text encoder. Student is always freshly trained with original `train.py`, original loss/optimizer, original head selection and original segmentation scripts. Every original COCO image retains its actual number of captions (5, 6, 7, etc.).

Sources of truth (present in the Oct 27, 2026 GitHub `main`):

- `experiments.Head_Match.common.teacher.TeacherBank`: Teacher weights + raw CLIP text feature -> projection. **Do not normalize CLIP raw features before projection.**
- `experiments.Head_caption_completeness_diagnosis.run.FrozenData`: original-caption frozen Teacher argmax Head, head/image evidence, calibrated single-head S5 visual support.
- `experiments.Head_caption_completeness_diagnosis.text_evidence.TextScorer.score(...,'S3')`: S5 phrase-level text evidence and UNKNOWN; CLIP-based.
- `experiments.Head_caption_completeness_diagnosis.core`: comparison_mask, coverage, drift_decomposition (S5 uses these unchanged).
- `experiments.Head_caption_completeness.train_seed` and `.pipeline`: original train.py wrapper and conflict-resilient segmentation launcher.
- `experiments.Head_caption_completeness.core.BENCHMARKS` and `BASELINES`: the requested five evaluation tasks, same config/PAMR settings as the previous baseline evaluations.

**Scope restriction:** Only frozen CLIP, pretrained Teacher, MSCoco14 gallery, MSCOCO14 cached head/text/image vectors. No LLM or language parser. `S5` scores are proxies rather than class existence probabilities or image-level ground truth. An object's low *specific-head* score does not imply its absence from the image.

## 1. Installation

The ZIP contains `experiments/Head_caption_S5_Editing/`. Unzip directly into the Talk2DINO repository **root**; no original file should be overwritten.

```bash
cd /mnt/sda/master/code/oymk/Talking2dino/Talk2DINO
unzip -o /path/to/Head_caption_S5_Editing.zip
conda activate talk2dino   # use the existing environment that runs original Talk2DINO
python -m unittest discover -s experiments/Head_caption_S5_Editing/tests -v
```

If you download the parent project manually, ensure the ZIP's root is `experiments/` (NOT `Talk2DINO/experiments/`). The new experiment has no new pip dependencies beyond the existing Talk2DINO and S5 implementations. In particular, do not randomly upgrade CUDA/PyTorch/mmcv. The test machine used for authoring did not have your project dataset, image cache, GPU or pretrained weights; only unit tests / compile checks were run, not a full train/OVSS.

Required artifacts:

```text
Talk2DINO/
├── outputs/coco2014_b14/train.pth
├── outputs/coco2014_b14/val.pth
├── outputs/Head_caption_completeness/clip_images.pth
├── weights/vitb_mlp_infonce.pth
├── configs/vitb_mlp_infonce.yaml
├── experiments/local_cocu/MSCOCO14_train_gallery.json
├── experiments/Head_Match/common/teacher.py
├── experiments/Head_caption_completeness_diagnosis/{run,text_evidence,core}.py
├── experiments/Head_caption_completeness/{pipeline,train_seed,core}.py
└── experiments/Head_caption_S5_Editing/  <-- new files
```

Run dependency and data integrity checks:

```bash
python -m experiments.Head_caption_S5_Editing.preflight --deep
```

`--deep` may load large `.pth` tensors and requires adequate RAM. To recreate the CLIP image feature cache when missing (requires train2014 JPEGs):

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_completeness.prepare \
  --base-train outputs/coco2014_b14/train.pth \
  --image-root data/coco2014/train2014 \
  --cache outputs/Head_caption_completeness/clip_images.pth \
  --neighbors outputs/Head_caption_completeness/clip_neighbors.json \
  --gpu-device cuda --batch-size 96 --workers 8 --topk 10
```

This is the *old* `prepare.py` invoked as is. It generates an image neighbor file too, although the S5-Editing logic only uses the CLIP image vectors. Adjust `--image-root` for your installed COCO JPEG path. It must cover ALL training images before a full run.

## 2. Algorithm (first-principles control variables)

For every original training caption `c` of image `i`:

1. Read raw cached caption feature `ann_feats[c]` (512D) and its image's 12 DINOv2 Head Features (12×768). Select fixed head `h*=argmax cosine(h, Teacher(ann_feats[c]))` via original `FrozenData.evidence`.
2. Compute cosine similarity of this **one fixed head** and projected prompts `Teacher(CLIP('a photo of a {concept}'))`. Retrieve highest **10** and lowest **10** concepts from the **full Gallery** (stable tie-break). Global Bottom-10 are negative controls, *not direct deletions*. For Remove, rank **concepts explicitly mentioned in the caption** by ascending head similarity and take their bottom 10, because global Bottom-10 usually do not appear in the caption. Optional `--remove-pool global_low10` enforces the strict global-bottom intersection, typically causing near-zero Remove edits.
3. Object-only eligibility: curated, conservative concrete noun vocabulary in `editing.py`, plus optional text-file `--object-allowlist`. Person/action/attribute/abstract terms excluded. This is an explicit *type* filter, not visual-presence verification. Global Top/Bottom-10 still log all gallery concepts, including ineligible ones.
4. Proposed **ADD** requires: object not expressed by that raw caption (lexical/alias), S5 `V_head >= 0.60`, S5 whole-image CLIP support `V_image >=0.30`, and *known* baseline text `T <=0.25`. Apply `with_also_visible_v1`: `{caption}, with a/an <concept> also visible.`.
5. Proposed **REMOVE** requires: existing, literal object-noun mention, `V_head <=0.12`; by default also `V_image <=0.18` and `V_other_heads <=0.30`, to avoid deleting concepts that belong to other Heads. Replace exact singular noun -> `thing`, plural -> `things` with all exact occurrences; no sentence regeneration. For less conservative explorations, `--remove-guard head_only` disables the image/other-head conditions, but it has higher false deletion risk.
6. For every proposed edit, `TextScorer.score(...,'S3')` recomputes `T_after`, with `V_head` and chosen head fixed. CLIP **non-truncating** tokenizer must accept the edited text. `comparison_mask` demands target known pre **and** post, and uses same comparable concept set for S5 `D_add`, `D_remove`.
7. **ADD admission**: target `ΔT >=0.20`, `ΔD_add <= -0.001`, non-target mean abs T drift <=0.10, target D_add absolute contribution >= full absolute off-target contribution. **REMOVE admission**: target `ΔT <=-0.20`, `ΔD_remove <=-0.001`, non-target mean abs T drift <=0.10, and any increase in D_add <=0.01. Both additionally require Teacher fixed-head similarity drop >= -0.03. These values are **precommitted heuristic gates**, not validation-fitted scientific constants.
8. Pick at most **one ADD and one REMOVE per original caption**, ranking passing candidates by corresponding reduction in S5 deficiency. Emit E0 raw, E1 remove-only, E2 add-only, E3 remove-first-then-add. For E3 both operations together are re-scored against original caption with a common fixed Head; if joint direction fails, fall back to best individually accepted variant rather than silently accepting a broken combination. All selected choices/abstentions/intermediate diagnostics are preserved in JSONL logs.
9. Re-encode only edited caption rows using the original CLIP ViT-B/16 text encoder, retain original image/annotation IDs, all source image Head tensors and exactly the original per-image number of annotations. Launch original `train.py` via the repo's unmodified existing `train_seed.py` wrapper; then call original segmentation main.py with identical settings and tasks.

**Experimental limitations:** The calibration null uses sampled random single Heads rather than caption-conditioned selected Heads (selection bias); S5 `UNKNOWN` cannot be forced to 0. `thing` may still carry nontrivial text semantics. The natural-language `with ... also visible` suffix does not assert physical truth. S5 gain alone does not guarantee VOC/ADE/COCO mIoU improvement.

## 3. Smoke discovery (first, no training)

```bash
# 20 images / all original captions; this uses only the FIRST 20 images; no .pth train created.
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_S5_Editing.select \
  --device cuda --limit 20 --captions-per-image 0 --seed 123 \
  --topk 10 --max-evals-per-operation 3 \
  --output outputs/Head_caption_S5_Editing/smoke20
```

Review the outputs `smoke20/candidates.jsonl`, `smoke20/E0.jsonl`, ..., `smoke20/E3.jsonl`, `smoke20/metadata.json`. For smoke training **only**, explicitly supply `--selection-dir outputs/Head_caption_S5_Editing/smoke20` and `--subset-train`; do **not** accidentally train a 20-image model and compare its mIoU against a full-data baseline.

## 4. Full discovery / build / training / OVSS

```bash
cd /mnt/sda/master/code/oymk/Talking2dino/Talk2DINO
conda activate talk2dino

# 0 = full image split; captions-per-image = 0 means ALL original captions, topk = 10
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_S5_Editing.select \
  --device cuda --limit 0 --captions-per-image 0 \
  --seed 123 --topk 10 --max-evals-per-operation 10 \
  --remove-pool caption_low10 --remove-guard all_views \
  --output outputs/Head_caption_S5_Editing/discovery

# Build E1, E2, E3 training .pth; E0 trains on unmodified raw train.pth.
for EXP in E1 E2 E3; do
  CUDA_VISIBLE_DEVICES=0 bash experiments/Head_caption_S5_Editing/run.sh "$EXP" 123 build 0
done

# Fresh original Talk2DINO training / same val for all 4 conditions
for EXP in E0 E1 E2 E3; do
  CUDA_VISIBLE_DEVICES=0 bash experiments/Head_caption_S5_Editing/run.sh "$EXP" 123 train 0
done

# Run all 5 segmentation benchmarks for every trained checkpoint
for EXP in E0 E1 E2 E3; do
  CUDA_VISIBLE_DEVICES=0 bash experiments/Head_caption_S5_Editing/run.sh "$EXP" 123 eval 0
done

# CSV report; missing experiments marked not_run (never fabricated)
python -m experiments.Head_caption_S5_Editing.summarize --seeds 123
```

Instead of three separate calls, `bash experiments/Head_caption_S5_Editing/run.sh E3 123 all 0` calls build/train/eval for that experiment. `--stage build|train|eval|all` works directly from Python too.

**Compute note**: Full 80k+ COCO image × variable-count captions × S5 phrase re-scoring is costly. `--max-evals-per-operation 10` respects the full candidate budget, but do not assume it is fast; start with 20 then 800 images and inspect selection acceptance rate, GPU memory, and runtime, and use `--max-evals-per-operation 3` when a predeclared computational cap is scientifically acceptable. Do **not** compare performance of different discovery settings without naming them as separate experimental conditions. For a real full-data model, never set `--limit` to 20/800.

## 5. Segmentation task mapping / baselines

| key | baseline mIoU | model config | evaluation config |
|---|---:|---|---|
| voc20 | 88.54 | `voc/dinotext_voc_vitb_mlp_infonce.yml` | `voc/eval_voc_pamr.yml` |
| voc | 65.77 | `voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml` | `voc_bg/eval_voc_bg_pamr.yml` |
| coco_stuff | 30.23 | `stuff/dinotext_stuff_vitb_mlp_infonce.yml` | `stuff/eval_stuff_pamr.yml` |
| ade20k | 22.53 | `ade/dinotext_ade_vitb_mlp_infonce.yml` | `ade/eval_ade_pamr.yml` |
| context59 | 42.42 | `context/dinotext_context_vitb_mlp_infonce.yml` | `context/eval_context_pamr.yml` |

All evaluation configuration paths above are under `src/open_vocabulary_segmentation/configs/`. The launcher uses `MASTER_ADDR=127.0.0.1`, a dynamically OS-selected free `MASTER_PORT`, `RANK=0`, `WORLD_SIZE=1`, `LOCAL_RANK=0`, with port-collision retry from the existing experiment. This is not a guarantee against unrelated distributed-process environment or missing dataset/config errors. `segmentation_results/<checkpoint_name>.json` is produced by the original implementation; detailed results and logs are also saved per variant/seed by the new orchestration.

## 6. Output files

```text
outputs/Head_caption_S5_Editing/
├── discovery/
│   ├── metadata.json                  # full source coverage, settings and counts
│   ├── candidates.jsonl               # per annotation/head, highest/lowest 10, every S5 score and rejection
│   ├── E0.jsonl                       # no-op baseline
│   ├── E1.jsonl                       # remove
│   ├── E2.jsonl                       # add
│   └── E3.jsonl                       # both + joint fallback reason
├── training/E{1,2,3}/
│   ├── train.pth                      # cached 12-head DINO + edited CLIP 512-d captions
│   └── manifest.json                  # edited annotation IDs and manifest SHA256
├── runs/E{0,1,2,3}/seed_123/
│   ├── train.log
│   ├── build.log
│   └── eval/{voc20,voc,coco_stuff,ade20k,context59}/metrics.json
└── segmentation_comparison.csv

weights/Head_caption_S5_Editing/E{0,1,2,3}/seed_123/
└── vitb_mlp_infonce_s5edit_E*_seed123.pth

weights/vitb_mlp_infonce_s5edit_E*_seed123.pth   # symlink required by original eval
```

The experiment refuses missing inputs, invalid image/annotation IDs, partial discovery silently used as full, unexpected token truncation, changed per-image caption counts, stale weights/datasets, missing segmentation task outputs, and mismatched raw-vs-recomputed CLIP embeddings (unless explicitly bypassed with `--allow-feature-drift`).

## 7. Ablation suggestions

Primary: E0 raw vs E1 remove vs E2 add vs E3 combined. Secondary controls: `--remove-pool global_low10` (strict proposed bottom10), `--remove-guard head_only` (safety ablation), template `with_visible_v0` (not mixed with default), `--max-evals-per-operation`, and curated object allowlist. Freeze all thresholds **before** looking at segmentation test mIoU; evaluate at least 3–5 training seeds if initial effect warrants.

## 8. What has and has not been validated

- Source path names and launcher conventions were checked against the connected GitHub repo on Oct 27, 2026.
- `compileall` passed, and six unit tests for editing, S5 score direction, candidate ordering, and variable 5/6/7-caption structural consistency passed with minimal pure-module mocks in the packaging environment.
- **Not verified here:** full GPU feature loading, all dependencies installed on the user's server, `train.py` execution end-to-end, CLIP cache completeness, and actual segmentation mIoUs. `preflight --deep` and smoke20 are designed to detect those integration issues on the real server. Do not treat the distributed ZIP as proof of a successful full-scale run.


## Variable captions per image (5/6/7) — hotfix

- Read each image's **actual** original caption count in `train.pth`.
- **Full discovery** must use `--captions-per-image 0` (0 means all); `--limit 0` means all images.
- `--captions-per-image N > 0` is only a per-image truncation for fast diagnostic experiments; it is not a full training split.
- Every E0/E1/E2/E3 condition must preserve **all original annotation IDs and the original per-image count**, whether 5, 6 or 7 (or another positive count).
- `preflight --deep` reports the distribution and checks for missing, duplicate, or orphan IDs.
- Existing `Head_caption_completeness_diagnosis.run` CLI still has an independent 1..5 cap. Do not edit upstream S5 to fix this experiment: our `select.py` now overrides that default and iterates all annotations itself.

Full discovery command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.Head_caption_S5_Editing.select \
  --device cuda --limit 0 --captions-per-image 0 --seed 123 \
  --topk 10 --max-evals-per-operation 10 \
  --output outputs/Head_caption_S5_Editing/discovery
```

If you previously generated truncated manifests or training `.pth` files, **rerun discovery and rebuild** `E1/E2/E3` with `--force-build`; do not reuse old generated datasets.
