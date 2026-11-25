# Repository Audit: Full MSCOCO14 CoCu-RCM

## Repository snapshot

- Repository: `Ocean-kang/Talk2DINO`
- Default branch: `main`
- Audited Git tree SHA: `442648db0a69e059d931a275e60f23ec58a1fb99`
- This package was not committed to GitHub; it is delivered as an independent ZIP.

## Required files

| Path | Status | Audit result |
|---|---|---|
| `AGENTS.md` | Not found | Root file returns 404 on current `main`. |
| `PROJECT_CONTEXT.md` | Not found | Root file returns 404 on current `main`. |
| `experiments/5k_cocu_rcm/README.md` | Present | Documents five arms, inputs, local loss scale, evaluation. |
| `experiments/5k_cocu_rcm/` | Present | Contains config, utility, loss, training Dataset/trainer, tests and shell wrappers. |
| `experiments/full_mscoco14_ecsd/` | Present | Provides full MSCOCO14 raw-caption/data preparation pipeline and evaluation wrappers. |
| `experiments/local_cocu_talk2dino/` | Present | Provides Local-CoCu A-H pipeline, `Vision`, `Teacher`, `mine('D')`, `bbox_from_map`. |
| `experiments/local_cocu_novel/` | Present | Provides lexical novelty filtering and `build_train.py` for 5 raw + up-to-3 CoCu captions. |
| `src/dataset.py` | Present | `DinoClipDataset` reads `images` and `annotations` PTH schema. |
| `src/model.py` | Present | `ProjectionLayer` supports the original config. |
| `src/loss.py` | Present | `ContrastiveLoss` includes original bidirectional InfoNCE. |
| `src/train_util.py` | Present | Original seed, scheduler, training helpers. |
| `train.py` | Present | Original training entry and Val feature selection. |
| OVSS evaluator | Present | `src/open_vocabulary_segmentation/main.py` and original benchmark configs. |

## Critical 5K package discrepancy

The current Git tree for `experiments/5k_cocu_rcm/` contains:
- `README.md`, `PACKAGE_MANIFEST.json`
- `common/utils.py`
- `configs/rcm.yaml`
- `models/rcm_loss.py`
- `training/rcm_dataset.py`, `training/train_cocu_rcm.py`
- shell wrappers, tests, result collector

However, it does **not** contain `experiments/5k_cocu_rcm/data/`. The committed `run_preprocess.sh` calls:
- `data/build_concept_archive.py`
- `data/extract_region_gallery.py`
- `data/verify_regions.py`
- `data/build_sampling_index.py`

These four scripts are missing from the audited repository tree. The Full package therefore implements full-scale versions of these stages in its own side-car directory. It does not claim those missing 5K modules were directly reused.

## Confirmed training/loss contract

### Talk2DINO PTH schema

`src.dataset.DinoClipDataset` expects:
- top-level `images` and `annotations`
- image-level feature field, default `disentangled_self_attn`
- annotation-level CLIP feature field, default `ann_feats`
- metadata `image_id` and annotation `id`

The original `train.py` selects `avg_self_attn_out` for validation when the train feature is `disentangled_self_attn`.

### Global InfoNCE

`src.loss.ContrastiveLoss.compute_contrastive_loss`:
- computes bidirectional cross entropy,
- averages the image-to-text and text-to-image CE,
- divides by `scores.shape[0] ** 2`.

The `logit_scale` is a parameter owned by the loss module and initialized from `log(1/0.07)`. The 5K trainer constructs the criterion separately and optimizes `model.parameters()`, so this logit-scale parameter is not included in that optimizer. Full CoCu-RCM preserves the same construction and behavior rather than silently changing it.

### Local Loss

`experiments/5k_cocu_rcm/models/rcm_loss.py::multi_positive_infonce`:
- L2-normalizes projected text, positive region and negative region features,
- computes one multi-positive log-softmax loss per active anchor,
- averages over active anchors,
- returns a differentiable zero if no valid anchor exists.

The 5K trainer then computes:

`local_scaled = local_raw / (batch_size ** 2)`

`weighted_local = lambda_local * local_scaled`

`total_loss = global_loss + weighted_local`

The Full trainer imports this exact 5K loss implementation and preserves this normalization.

## Original `train.py` protocol cross-check

The current `train.py` calls `src.train_util.do_train()` with defaults `optimizer="adam"`, `scheduler="linear"`, and `warmup=0`. `do_train()` creates `optim.Adam(model.parameters(), lr=lr)`; the `weight_decay=0.05` argument is only used in the `AdamW` branch, so the default Adam path has no weight decay. It uses `num_workers=8`, `shuffle` from the config (default true), and the original validation feature switch to `avg_self_attn_out` when training with `disentangled_self_attn`. The Full trainer preserves these effective defaults.

## Confirmed 5K training protocol

`experiments/5k_cocu_rcm/configs/rcm.yaml` records:
- DINOv2 ViT-B/14-reg, 448 resolution
- 12 heads, feature dim 768
- Attention quantile 0.90, bbox expansion 0.15
- CLIP ViT-B/16
- top 3 regions
- K=2, negatives=16, sampling seed=123
- local temperature 0.07, lambda 0.1

`configs/vitb_mlp_infonce.yaml` records:
- DINO embed dim 768
- hidden MLP enabled, tanh
- LR 1e-4
- InfoNCE
- 100 epochs
- batch size 128

The 5K trainer reads epoch/batch/LR from the original config and uses Adam. The Full trainer uses the same original config and optimizer choice. It adds epoch-boundary resume checkpoints but does not change the model or loss definition.

## Confirmed Local-CoCu implementation

`experiments/local_cocu_talk2dino/common/vision.py`:
- loads `dinov2_vitb14_reg`,
- registers the final attention qkv hook,
- resizes/center-crops to 448,
- returns 12 head pooled features, attention maps and transformed image.

`common/algorithms.py::bbox_from_map`:
- thresholds each head attention map at a quantile,
- obtains the attended patch rectangle,
- expands the rectangle,
- returns a bbox in 448 center-crop coordinates.

`common/teacher.py::Teacher`:
- loads OpenAI CLIP ViT-B/16,
- loads existing Talk2DINO `ProjectionLayer` and teacher checkpoint,
- caches prompt embeddings,
- maps text concepts into the 768-D DINO space.

`common/experiment.py::mine('D')`:
- retrieves candidate concepts using the teacher and DINO head features,
- derives an attention bbox for the selected head,
- encodes the crop with the existing DINO Vision,
- scores local and global concept evidence,
- keeps concepts meeting the existing local threshold.

Full CoCu discovery calls this same `mine('D')` implementation and only adds chunked output/resume around it.

## Full MSCOCO14 reference pipeline

`experiments/full_mscoco14_ecsd/README.md` and `run_preprocess.sh` establish:
- Train2014 expected 82,783 images / 413,915 captions
- Val2014 expected 40,504 images / 202,520 captions
- exactly five captions per image
- existing DINO feature reuse from `outputs/coco2014_b14/train.pth` and `val.pth`
- original `dino_extraction_v2.py` fallback
- original `text_features_extraction.py` for CLIP `ann_feats`
- original Talk2DINO training config and OVSS evaluation

The Full CoCu-RCM adapter consumes the same raw PTH schema and checks these counts and fields.

## Isolation guarantee

New package only:
`experiments/full_mscoco14_cocu_rcm/`

Runtime output only:
`outputs/full_mscoco14_cocu_rcm/`

New checkpoints only:
`weights/vitb_mlp_infonce_full_mscoco14_cocu_rcm_*.pth`

No files under `src/`, `train.py`, original `configs/`, original OVSS configs, original weights or datasets are modified by this package.
