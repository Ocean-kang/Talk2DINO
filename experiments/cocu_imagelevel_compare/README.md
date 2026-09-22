# CoCu × CLIP / Talk2DINO-image curated concept comparison

Place this whole directory at:

```text
Talk2DINO/
└── experiments/
    └── cocu_imagelevel_compare/
        ├── build_curated_concepts.py
        ├── run_all.sh
        └── MSCOCO14_train_gallery_exactmatch.json
```

This experiment does **not** modify base Talk2DINO code.

## What is compared

### CLIP baseline

```text
COCO image
   ↓
CLIP ViT-B/16 image encoder
   ↓
image embedding
```

and

```text
"a photo of a {concept}"
   ↓
CLIP ViT-B/16 text encoder
   ↓
concept embedding
```

Both image-image retrieval and concept-image scoring happen in CLIP space.

### Talk2DINO-image

```text
COCO image
   ↓
DINOv2 ViT-B CLS
   ↓
image embedding in DINO space
```

and

```text
"a photo of a {concept}"
   ↓
CLIP ViT-B/16 text encoder
   ↓
image-level Talk2DINO projector
   ↓
concept embedding in DINO space
```

Both image-image retrieval and concept-image scoring happen in DINO space.

## Gallery policy

`MSCOCO14_train_gallery_exactmatch.json` is used exactly as provided.

- no plural generation
- no TextBlob
- no NLTK
- no stemming
- no lemmatization
- no extra concept mining

Captions are only lowercased and punctuation-normalized so exact one/two/three-word
gallery phrases can be matched.

## CoCu logic kept

For each anchor image:

1. Retrieve top-L visually similar images (`L=8` by default).
2. Union their caption-derived concepts into a mini concept archive.
3. For each candidate concept:
   - `global = sim(concept, anchor)`
   - `local = global / mean(sim(concept, reference_images + anchor))`
   - `score = global / max_prob + local`
4. Rank concepts by the CoCu score.

The default keeps **self retrieval**, because original CoCu searches the same
feature index and does not explicitly remove the anchor.

No custom k-means is applied.

## COCO-specific adaptation

MSCOCO2014 has multiple captions per image. The source semantic set for one
image is therefore the union of exact gallery concepts found across all original
captions of that image.

Each output image record contains:

```json
{
  "image_id": 123,
  "file_name": "COCO_train2014_000000000123.jpg",
  "captions": ["...", "...", "...", "...", "..."],
  "original_concepts": ["person", "bike"],
  "curated_concepts": [
    {"concept": "helmet", "score": 1.82},
    {"concept": "road", "score": 1.61},
    {"concept": "rider", "score": 1.55}
  ],
  "new_curated_concepts": [
    {"concept": "helmet", "score": 1.82},
    {"concept": "rider", "score": 1.55},
    {"concept": "street", "score": 1.42}
  ]
}
```

`curated_concepts` follows source ranking and may repeat a concept already
present in the original captions.

`new_curated_concepts` is an additional convenience field that removes those
already-present concepts. Use this field if the downstream experiment is
specifically "semantic expansion only".

## DINO CLS feature PTH

For Talk2DINO-image, use the **same DINO CLS feature PTH used to train the
image-level projector**. This is preferred over re-extracting images because it
guarantees the same DINO model, resolution, preprocessing, and CLS definition.

The script accepts image features stored under any of:

- `dino_features`
- `x_norm_clstoken`
- `cls_token`
- `cls`

If you do not yet have a CLS PTH, generate one with the same settings used in
your image-level training. In this Talk2DINO fork, a typical route is:

```bash
python experiments/cocu_talk2dino/build_cocu.py init-pth \
  --captions-json data/coco2014/annotations/captions_train2014.json \
  --out outputs/coco2014_cls/train_init.pth

CUDA_VISIBLE_DEVICES=2 \
python dino_extraction_v2.py \
  --ann_path outputs/coco2014_cls/train_init.pth \
  --out_path outputs/coco2014_cls/train_cls.pth \
  --data_dir data/coco2014 \
  --model dinov2_vitb14_reg \
  --resize_dim 448 \
  --crop_dim 448 \
  --extract_cls
```

Only use the command above if those are also the settings used by your
image-level projector training. Otherwise point `DINO_PTH` to the actual
training PTH.

## Run three methods

From the Talk2DINO repository root:

```bash
GPU=2 \
DINO_PTH=/path/to/the/exact/train_cls.pth \
bash experiments/cocu_imagelevel_compare/run_all.sh
```

Outputs:

```text
outputs/cocu_imagelevel_compare/
├── clip_vitb16.json
├── talk2dino_image_cls_imagelevel.json
├── talk2dino_image_cls_imagelevel_epoch1000.json
└── cache/
```

## Useful ablations

Exclude self from image retrieval:

```bash
--exclude-self
```

Save the complete CoCu scored archive instead of only the top-N fields:

```bash
--save-all-scored
```

Keep all ranked concepts in `curated_concepts`:

```bash
--num-curated 0
```

Smoke test first 100 images:

```bash
--limit 100
```
