from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.Caption_Head_Visualization.data import torch_load_legacy


class P1PseudoFeatureDataset(Dataset):
    """
    Adapter matching the item contract consumed by src.train_util.do_train.

    Training items return image=[12,768], preserving original max_score head
    selection. Validation items return image=[768], matching train.py's
    original avg_self_attn_out validation path.
    """

    def __init__(
        self,
        source_pth: str | Path,
        cache_dir: str | Path,
        *,
        training: bool,
        allow_partial: bool = False,
        max_samples: int = 0,
    ):
        self.source_pth = Path(source_pth)
        self.cache_dir = Path(cache_dir)

        if not self.source_pth.is_file():
            raise FileNotFoundError(f"Missing source feature PTH: {self.source_pth}")

        meta_path = self.cache_dir / "metadata.json"
        feature_path = self.cache_dir / "pseudo_features.npy"
        ann_ids_path = self.cache_dir / "annotation_ids.npy"
        image_ids_path = self.cache_dir / "image_ids.npy"
        done_path = self.cache_dir / "done_images.npy"

        for path in (meta_path, feature_path, ann_ids_path, image_ids_path, done_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing P1 cache file: {path}")

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not self.meta.get("complete", False) and not allow_partial:
            raise RuntimeError(
                f"P1 cache is incomplete: {self.cache_dir}. "
                "Finish prepare_features.py first, or use --allow-partial-cache "
                "only for a smoke test."
            )

        source = torch_load_legacy(self.source_pth)
        if not isinstance(source, dict) or not {"images", "annotations"} <= set(source):
            raise ValueError("Expected Talk2DINO COCO PTH with images/annotations")

        annotations = source["annotations"]
        source_ann_ids = np.asarray([int(a["id"]) for a in annotations], dtype=np.int64)
        cache_ann_ids = np.load(ann_ids_path)
        if not np.array_equal(source_ann_ids, cache_ann_ids):
            raise ValueError(
                "Annotation order/IDs in cache do not match source PTH. "
                "Rebuild the P1 cache from this exact source file."
            )

        self.features = np.load(feature_path, mmap_mode="r")
        if self.features.ndim != 3 or tuple(self.features.shape) != (
            len(annotations), 12, 768
        ):
            raise ValueError(
                f"Expected pseudo feature array [{len(annotations)},12,768], "
                f"got {tuple(self.features.shape)}"
            )

        image_ids = np.load(image_ids_path)
        done_images = np.load(done_path)
        if len(image_ids) != len(done_images):
            raise ValueError("done_images.npy does not match image_ids.npy")
        done_ids = set(int(x) for x, done in zip(image_ids.tolist(), done_images.tolist()) if done)

        self.entries = []
        for row, ann in enumerate(annotations):
            image_id = int(ann["image_id"])
            if allow_partial and image_id not in done_ids:
                continue
            if "ann_feats" not in ann:
                raise KeyError(
                    f"Annotation {ann.get('id')} has no ann_feats. "
                    "The original Talk2DINO CLIP text features are required."
                )
            self.entries.append((row, ann))

        if max_samples and max_samples > 0:
            self.entries = self.entries[:int(max_samples)]
        if not self.entries:
            raise RuntimeError("No annotation samples are available in this P1 cache")

        self.training = bool(training)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row, ann = self.entries[index]
        # Copy the memmap row into writable float32 memory for PyTorch/DataLoader.
        value = np.array(self.features[row], dtype=np.float32, copy=True)
        if self.training:
            image_feature = torch.from_numpy(value)  # [12,768]
        else:
            image_feature = torch.from_numpy(value.mean(axis=0))  # [768]

        text_feature = torch.as_tensor(ann["ann_feats"]).float().reshape(-1)
        if text_feature.numel() != 512:
            raise ValueError(
                f"Annotation {ann.get('id')} CLIP ann_feats must be 512-D, "
                f"got {tuple(text_feature.shape)}"
            )

        return {
            "annotation": text_feature,
            "image": image_feature,
            "metadata": {
                "annotation_id": int(ann["id"]),
                "image_id": int(ann["image_id"]),
            },
        }
