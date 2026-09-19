"""Side-car dataset for CoCu-Talk2DINO; original src/dataset.py remains untouched."""
from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class CocuDinoClipDataset(Dataset):
    """Reuse base DINO features while sampling original or curated text for each image.

    repeats_per_image defaults to 5 so one epoch has about the same number of samples
    as COCO's original ~5 captions/image, avoiding an 8/5 increase in optimization steps
    when 3 curated captions are added.
    """

    def __init__(self, base_features_file, curated_file=None, curated_prob=0.5,
                 repeats_per_image=5, features_name="disentangled_self_attn",
                 text_features="ann_feats", load_attn_maps=False):
        if not 0.0 <= curated_prob <= 1.0:
            raise ValueError("curated_prob must be in [0, 1]")
        if repeats_per_image < 1:
            raise ValueError("repeats_per_image must be >= 1")

        data = torch.load(base_features_file, map_location="cpu")
        self.images = {int(im["id"]): im for im in data["images"]}
        self.image_ids = list(self.images)
        self.original = defaultdict(list)
        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            if text_features not in ann:
                raise KeyError(f"Base annotation missing '{text_features}'. Run text_features_extraction.py first.")
            self.original[image_id].append(ann)
        missing = [x for x in self.image_ids if not self.original[x]]
        if missing:
            raise ValueError(f"{len(missing)} images have no original captions; first id={missing[0]}")

        self.curated = defaultdict(list)
        if curated_file:
            sidecar = torch.load(curated_file, map_location="cpu")
            for ann in sidecar["annotations"]:
                if "ann_feats" not in ann:
                    raise KeyError("Curated annotation missing 'ann_feats'. Rebuild sidecar.")
                self.curated[int(ann["image_id"])].append(ann)
        if curated_prob > 0 and not curated_file:
            raise ValueError("curated_file is required when curated_prob > 0")

        self.curated_prob = curated_prob
        self.repeats = repeats_per_image
        self.features_name = features_name
        self.text_features = text_features
        self.load_attn_maps = load_attn_maps

    def __len__(self):
        return len(self.image_ids) * self.repeats

    def __getitem__(self, idx):
        n = len(self.image_ids)
        image_id = self.image_ids[idx % n]
        slot = idx // n
        original_pool = self.original[image_id]
        curated_pool = self.curated.get(image_id, [])
        use_curated = bool(curated_pool) and random.random() < self.curated_prob
        pool = curated_pool if use_curated else original_pool
        ann = pool[slot % len(pool)]
        image = self.images[image_id]

        if self.features_name not in image:
            raise KeyError(f"Image {image_id} missing feature '{self.features_name}'")
        text_key = "ann_feats" if use_curated else self.text_features
        result = {
            "annotation": ann[text_key],
            "image": image[self.features_name],
            "metadata": {
                "annotation_id": int(ann["id"]),
                "image_id": image_id,
            },
        }
        if self.load_attn_maps:
            result["self_attn_maps"] = image["self_attn_maps"]
            result["dino_features"] = image["dino_features"]
        return result
