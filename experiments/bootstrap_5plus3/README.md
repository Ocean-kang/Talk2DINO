# bootstrap_5plus3

This experiment uses the three **already generated JSON files** and trains three
new Talk2DINO projectors with a strict:

**5 original COCO captions + 3 curated captions = 8 samples per image**

No file outside `experiments/bootstrap_5plus3/` needs to be modified.

## 1. Directory

Put these files together:

```text
Talk2DINO/
└── experiments/
    └── bootstrap_5plus3/
        ├── build_5plus3.py
        ├── run_5plus3.sh
        ├── clip_vitb16.json
        ├── talk2dino_image_cls_imagelevel.json
        └── local_cocu_compare.json
```

The builder expects the original pre-extracted Talk2DINO files:

```text
outputs/coco2014_b14/train.pth
outputs/coco2014_b14/val.pth
```

`train.pth` must already contain:
- `images[*].disentangled_self_attn`
- `annotations[*].ann_feats`
- exactly 5 original annotations per image

## 2. Concept selection

### clip_vitb16.json

For every `images[]` record:
1. use `new_curated_concepts` first;
2. if fewer than 3 unique concepts remain, supplement from `curated_concepts`;
3. take exactly 3.

### talk2dino_image_cls_imagelevel.json

Same policy as CLIP.

### local_cocu_compare.json

For each image:
1. read `concept_head_scores`;
2. sort by score descending;
3. deduplicate concept names;
4. take Top-3.

`best_head` is retained as audit metadata, but the student still uses the
original Talk2DINO dynamic head matching during training.

## 3. Caption construction

Each selected concept is one independent training caption:

```text
furniture    -> a photo of a furniture
dining table -> a photo of a dining table
chair        -> a photo of a chair
```

Thus one image becomes:

```text
5 original COCO captions
+ 3 individual curated captions
= 8 training annotations
```

The builder encodes only the three new captions with CLIP ViT-B/16 and writes
their embedding to `ann_feats`. Original `ann_feats` and all DINO image
features are reused unchanged.

You can change the template if needed:

```bash
python experiments/bootstrap_5plus3/build_5plus3.py \
  --method all \
  --template "a photo containing {}"
```

For the main experiment, keep the same template for all three methods.

## 4. Build all three training files

From repository root:

```bash
cd /home/master/code/oymk/Talking2dino/Talk2DINO

CUDA_VISIBLE_DEVICES=0 \
python experiments/bootstrap_5plus3/build_5plus3.py \
  --method all \
  --base-train outputs/coco2014_b14/train.pth \
  --output-dir outputs/bootstrap_5plus3 \
  --num-curated 3 \
  --clip-model ViT-B/16 \
  --batch-size 512
```

Or:

```bash
bash experiments/bootstrap_5plus3/run_5plus3.sh build 0
```

Outputs:

```text
outputs/bootstrap_5plus3/
├── train_clip_vitb16_5plus3.pth
├── train_talk2dino_image_cls_imagelevel_5plus3.pth
├── train_local_cocu_compare_5plus3.pth
├── clip_vitb16_5plus3_manifest.json
├── talk2dino_image_cls_imagelevel_5plus3_manifest.json
└── local_cocu_compare_5plus3_manifest.json
```

The builder fails instead of silently continuing if any image cannot satisfy
strict 5+3.

## 5. Train three new projectors

CLIP-CoCu student:

```bash
bash experiments/bootstrap_5plus3/run_5plus3.sh train-clip 0
```

Image-level Talk2DINO bootstrap student:

```bash
bash experiments/bootstrap_5plus3/run_5plus3.sh train-image 1
```

Local Talk2DINO bootstrap student:

```bash
bash experiments/bootstrap_5plus3/run_5plus3.sh train-local 2
```

Equivalent direct commands are inside `run_5plus3.sh`.

Expected weights:

```text
weights/vitb_mlp_infonce_clip_vitb16_5plus3.pth
weights/vitb_mlp_infonce_talk2dino_image_cls_imagelevel_5plus3.pth
weights/vitb_mlp_infonce_local_cocu_compare_5plus3.pth
```

The three students use the same original:
- `train.py`
- `src/dataset.py`
- `src/model.py`
- `src/loss.py`
- `configs/vitb_mlp_infonce.yaml`

Only the three added captions differ.
