# Image-level CLS Projector

New experiment only. No original Talk2DINO files need to be modified.

Files:
- extract_cls_features.py: build compact COCO feature files with DINOv2 CLS + existing CLIP text ann_feats.
- train_cls_projector.py: one unique image per sample, random 1-of-5 COCO caption during training, bidirectional InfoNCE.
- vitb_cls_infonce.yaml: ViT-B/16 CLIP text (512) -> DINOv2 ViT-B (768) projector config.

Run from the Talk2DINO repository root.
