# Local-CoCu Novel Concepts → Original Talk2DINO (5+up-to-3)

**Single-variable Ponytail experiment** relative to the existing Local-CoCu 5+3:
**remove concepts already present in the target image's original five captions
BEFORE Local-CoCu's head matching**. All original `src/`, `train.py`, configs,
weights, and `experiments/local_cocu/` remain unchanged.

## What exactly is modified?

1. Unchanged original Local-CoCu: DINOv2 ViT-B/14-reg 224px CLS features → cosine Top-50 COCO image neighbors → concepts mined from neighboring original captions and the original gallery. The included **chunked retrieval** produces the same cosine Top-K as the original full-matrix script, without holding the full N×N matrix.
2. **NEW `filter_novel.py`**: for each target image, compare each candidate concept against **its own first five raw captions**. Remove already-mentioned concepts using case-insensitive token-boundary/contiguous-phrase matching plus conservative plural/irregular noun normalization. For strict lexical matching use `--exact-only`. The original Local-CoCu candidate extractor is reused unchanged, **including its known inability to extract gallery multiword phrases**. This controls the experiment to the requested intervention.
3. Unchanged original Local-CoCu teacher matcher: remaining candidates → CLIP ViT-B/16 text features → pretrained `vitb_mlp_infonce.pth` CLIP→DINO projector → 12-head maximum cosine score → Top-10 with `--threshold=None` unless supplied. Concepts are ranked via detailed matching scores.
4. **NEW `build_train.py`**: retain the first five original `ann_feats` and `disentangled_self_attn` image vectors; revalidate every chosen concept against the same five captions stored in base `train.pth`; append **up to 3 separate** new captions, e.g. `cat` → `a photo of a cat`, encoding only added text through CLIP ViT-B/16. Never invent replacements. If an image has fewer than 3 novel concepts, it receives 5+0/1/2, not fake concepts. `--strict` instead aborts unless every image gets 3.
5. **Original `python train.py`** trains a **new randomly initialized Student** with the repository's original InfoNCE / `disentangled_self_attn` max-head alignment. Teacher is **not** loaded into Student and **not** used during training. Validation is unchanged `outputs/coco2014_b14/val.pth`.
6. Evaluate Student with the repository's four evaluation configs (VOC20, Context59, ADE20K, COCO-Stuff), using `model.proj_name=vitb_mlp_infonce_local_cocu_novel_5plus3`.

## Directory and installation

ZIP contains top-level `local_cocu_novel/`. From repository root:

```bash
cd /home/master/code/oymk/Talking2dino/Talk2DINO
unzip local_cocu_novel.zip -d experiments/
```

Result:

```
experiments/local_cocu_novel/
  README.md
  pipeline.py                 # orchestration / stage resume
  run.sh                      # shell entry
  chunked_retrieval.py       # memory-bounded retrieval
  filter_novel.py            # new filtering step
  build_train.py             # new 5+up-to-3 dataset builder
  eval.sh                    # original 4 segmentation datasets
  tests/
```

## Requirements

Run in the existing Talk2DINO Python 3.10 / CUDA environment. Reuse the project's installed PyTorch, torchvision, `clip`, PIL, etc. Existing assets:

```
data/coco2014/train2014/
data/coco2014/annotations/captions_train2014.json
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
weights/vitb_mlp_infonce.pth
configs/vitb_mlp_infonce.yaml
experiments/local_cocu/MSCOCO14_train_gallery.json
```

The ZIP does **not** include COCO images, model weights, pretrained DINOv2 assets, or cached feature files. The original Local-CoCu extraction scripts load DINOv2 via `torch.hub`: make sure model weights/hub code are locally cached if the host is offline.

## One-command run — reuse your old Local-CoCu feature caches

Recommended when your previous extraction is complete and located under `outputs/local_cocu/`:

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/local_cocu_novel/run.sh all 1 --reuse-local
```

This reuses ONLY the old `candidate/candidate_concepts.json` and `head_features/head_features.pt`, then recomputes novelty filtering, head matching, augmented training dataset, and **original** Talk2DINO training. The original matched concepts output cannot be reused, because filtering must happen **before** the Top-K head selection.

If old output uses another base directory:

```bash
bash experiments/local_cocu_novel/run.sh all 1 --reuse-local \
  --source-dir outputs/CoCu_Bootstrap_Talk2dino/concepts/local
```

## One-command run — extract again from COCO (slow / costly)

```bash
bash experiments/local_cocu_novel/run.sh all 1
```

This stores independent source features under `outputs/local_cocu_novel/source/` and will reuse each completed extraction stage (unless `--force-extract` is supplied). DINO CLS: 224×224; DINO Head: 448×448; image neighbor Top-50; matcher projector from pretrained `weights/vitb_mlp_infonce.pth`.

## Stage-by-stage

Use `--reuse-local` on each stage if using the existing Local-CoCu caches:

```bash
# 1: Remove original-caption concepts, save stats/audit
bash experiments/local_cocu_novel/run.sh filter 1 --reuse-local

# 2: Run original head matching on the filtered candidates
bash experiments/local_cocu_novel/run.sh match 1 --reuse-local

# 3: Build 5+up-to-3 dataset, with CLIP text embeddings
bash experiments/local_cocu_novel/run.sh build 1 --reuse-local

# 4: Train via ORIGINAL train.py (not Bootstrap)
bash experiments/local_cocu_novel/run.sh train 1 --reuse-local

# 5: Evaluate on VOC20, Context59, ADE20K, COCO-Stuff
bash experiments/local_cocu_novel/run.sh eval 1
```

For fresh source extraction only:

```bash
bash experiments/local_cocu_novel/run.sh extract 1
```

Print commands without running/allocating GPU memory:

```bash
bash experiments/local_cocu_novel/run.sh all 1 --reuse-local --dry-run
```

Optional settings:

```bash
# Require exactly 3 *novel* concepts per image; abort if impossible
bash experiments/local_cocu_novel/run.sh all 1 --reuse-local --strict

# Strict literal words/phrases, no plural normalizer
bash experiments/local_cocu_novel/run.sh all 1 --reuse-local --exact-only

# Filter low-scoring head matches (default is NO threshold)
bash experiments/local_cocu_novel/run.sh all 1 --reuse-local --threshold 0.2
```

## Outputs

```
outputs/local_cocu_novel/
  novel_candidates.json                     # candidates AFTER subtraction
  novel_filter_audit.json                   # removed and kept counts, removed concepts
  matched/matched_concepts.json              # head-matched names
  matched/matched_concepts_detailed.json     # score, best_head
  train_local_cocu_novel_5plus3.pth          # original 5 + at most 3 new anns
  train_manifest.json                        # per-image concept details and histogram

weights/vitb_mlp_infonce_local_cocu_novel_5plus3.pth
output/eval/local_cocu_novel/vitb_mlp_infonce_local_cocu_novel_5plus3/
```

Important: the `5plus3` suffix identifies the **up-to-3 concept variant**; it is **not strict 8 samples/image** unless `--strict` succeeds. Read `train_manifest.json -> meta.histogram_added` before comparing results with a fixed-5+3 experiment. Different numbers of annotations/image imply a different optimization budget per epoch. For a controlled comparison, record sample counts and ideally use an equal-step follow-up.

## Scientific limitations

- This detects explicit lexical duplicates, including simple singular/plural variants, **not all semantic synonyms**. e.g., `bike` vs `bicycle` can still coexist. This is intentional: it avoids model-score thresholds becoming an additional experimental factor.
- A concept absent from all five captions is not necessarily **visually present** in the target image. Teacher head scores are an approximate relevance proxy, not visual ground truth.
- The original gallery extraction code uses `[a-z]+` single-token exact matches. Multiword entries in the gallery are not recovered by that original code; fixing that is a **separate ablation** to avoid confounding the novelty-filter experiment.
- The original `match_head_concepts.py` allows multiple concepts per head; this experiment retains that original behavior.
- This is **not Teacher-Guided Bootstrap training**. The pretrained projector only selects concepts; the downstream Student training uses original Talk2DINO dynamic head selection.

## Sanity tests

```bash
python -m unittest discover -s experiments/local_cocu_novel/tests -v
```
