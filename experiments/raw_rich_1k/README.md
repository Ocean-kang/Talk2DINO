# 1000-image raw vs rich caption experiment

This directory is a side-car experiment. Run each command from the **Talk2DINO repository root** on the Linux GPU machine. Keep the original repository files unchanged. Source captions and images are read only. `work/` holds generated links, cached features, checkpoints, and metrics.

## Design

- Exactly 1000 COCO image IDs from `captions_raw_rich_split.json`.
- Same deterministic 900 training / 100 validation image split for both arms, seed 123. `work/val_ids.json` records it.
- First five nonempty raw and rich captions for every image. Four source entries have a sixth raw caption; it is omitted for equal supervision.
- DINOv2 ViT-B/14 register backbone, 448 pixel preprocessing. Extracted DINO image features are reused for both arms. CLIP ViT-B/16 text features are extracted per arm.
- Original `ProjectionLayer` and ViT-B config are used. Training uses `disentangled_self_attn`; validation uses `avg_self_attn_out`, matching the original `train.py` feature choice. Each training batch contains unique images and one of each image's five captions, rotated across epochs, to avoid same-image false negatives.
- 50 epochs, Adam, learning rate 1e-4, batch size 64. Choose best checkpoint by validation loss. `metrics.csv` also reports caption-to-image R@1/R@5 over all 500 validation captions. These are within-subset retrieval measures, **not** segmentation mIoU.

## Input layout

The existing `data/coco2014_subset_10k/` can contain:

```
data/coco2014_subset_10k/
  captions_raw_rich_split.json
  images/
    COCO_train2014_000000000681.jpg
    ...
```

`images/train2014/` and `images/val2014/` are also accepted. Every ID in the JSON must have an image. `captions_subset.json` is unused.

## Commands

Use the repository's working training environment with CUDA, PyTorch, and `requirements.txt` installed. The first DINOv2 and CLIP run downloads model weights, so network access is needed once.

```
python experiments/raw_rich_1k/prepare.py init --captions data/coco2014_subset_10k/captions_raw_rich_split.json --images data/coco2014_subset_10k/images --work experiments/raw_rich_1k/work
```

The init step creates symlinks to images in `work/images/train2014/` and a `base.pth` annotation file. It never edits the source images.

```
python dino_extraction_v2.py --ann_path experiments/raw_rich_1k/work/base.pth --data_dir experiments/raw_rich_1k/work/images --out_path experiments/raw_rich_1k/work/dino.pth --model dinov2_vitb14_reg --resize_dim 448 --crop_dim 448 --batch_size 8 --extract_avg_self_attn --extract_disentangled_self_attn
python experiments/raw_rich_1k/prepare.py assemble --captions data/coco2014_subset_10k/captions_raw_rich_split.json --work experiments/raw_rich_1k/work
python text_features_extraction.py --ann_path experiments/raw_rich_1k/work/raw_train.pth --out_path experiments/raw_rich_1k/work/raw_train.pth --model ViT-B/16 --batch_size 128
python text_features_extraction.py --ann_path experiments/raw_rich_1k/work/raw_val.pth --out_path experiments/raw_rich_1k/work/raw_val.pth --model ViT-B/16 --batch_size 128
python text_features_extraction.py --ann_path experiments/raw_rich_1k/work/rich_train.pth --out_path experiments/raw_rich_1k/work/rich_train.pth --model ViT-B/16 --batch_size 128
python text_features_extraction.py --ann_path experiments/raw_rich_1k/work/rich_val.pth --out_path experiments/raw_rich_1k/work/rich_val.pth --model ViT-B/16 --batch_size 128
python experiments/raw_rich_1k/train_pair.py --work experiments/raw_rich_1k/work --arm raw --epochs 50 --batch-size 64
python experiments/raw_rich_1k/train_pair.py --work experiments/raw_rich_1k/work --arm rich --epochs 50 --batch-size 64
```

Compare the best epoch's `t2i_R@1`, `t2i_R@5`, and `val_loss` in `work/results/raw/metrics.csv` and `work/results/rich/metrics.csv`. The saved `best.pth` files are projection-layer state dictionaries for the original `ProjectionLayer` ViT-B architecture.

For a final open-vocabulary segmentation comparison, point the repository's existing segmentation evaluation configs at each checkpoint and use a **separate labeled dataset**, such as VOC or COCO-Stuff. The 1000 caption pairs alone cannot yield segmentation mIoU.
