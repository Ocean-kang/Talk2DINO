from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import normalize_concept, read_json, read_jsonl, sha256_file, torch_load
from src.dataset import DinoClipDataset


class FullRCMTrainingDataset(Dataset):
    """Talk2DINO dataset wrapper with lazy CPU loading of region feature shards."""

    def __init__(
        self,
        train_path,
        sampling_dir,
        region_feature_dir,
        region_index_path,
        features_name="disentangled_self_attn",
        text_features="ann_feats",
        max_cached_shards=2,
        expected_arm=None,
    ):
        self.base = DinoClipDataset(
            train_path, features_name=features_name, text_features=text_features
        )
        raw = torch_load(train_path)
        self.ann_by_id = {
            int(a["id"]): {
                "image_id": int(a["image_id"]),
                "novel_cocu_concept": a.get("novel_cocu_concept"),
            }
            for a in raw["annotations"]
        }
        if len(self.ann_by_id) != len(raw["annotations"]):
            raise ValueError("Duplicate annotation IDs in augmented training PTH")
        del raw

        sampling_dir = Path(sampling_dir)
        manifest_path = sampling_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = read_json(manifest_path)
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Sampling index is not complete: {manifest_path}")
        if expected_arm is not None and manifest.get("arm") != expected_arm:
            raise ValueError(
                f"Sampling arm mismatch: requested {expected_arm}, found {manifest.get('arm')}"
            )
        if expected_arm in ("cross", "shuffled") and not manifest.get("paired_rcm_shuffled"):
            raise RuntimeError("Cross-image RCM/Shuffled index lacks the paired-control contract")
        source_train_hash = manifest.get("source", {}).get("train_augmented_sha256")
        if source_train_hash and source_train_hash != sha256_file(train_path):
            raise RuntimeError("Sampling index was built from a different augmented training PTH")
        if int(manifest.get("k", 0)) < 1 or int(manifest.get("negatives", 0)) < 1:
            raise ValueError("Sampling manifest has invalid K-shot/negative counts")
        self.sampling = {}
        shard_hashes = manifest.get("shard_sha256", {})
        for shard_name in manifest.get("shards", []):
            shard_path = sampling_dir / shard_name
            if shard_name in shard_hashes and sha256_file(shard_path) != shard_hashes[shard_name]:
                raise RuntimeError(f"Sampling shard hash mismatch: {shard_path}")
            for row in read_jsonl(shard_path):
                key = row["key"]
                if key in self.sampling:
                    raise ValueError(f"Duplicate sampling anchor: {key}")
                self.sampling[key] = row

        self.region_feature_dir = Path(region_feature_dir)
        region_manifest_path = self.region_feature_dir.parent / "region_gallery_manifest.json"
        if not region_manifest_path.is_file():
            raise FileNotFoundError(region_manifest_path)
        region_manifest = read_json(region_manifest_path)
        if region_manifest.get("status") != "complete":
            raise RuntimeError(f"Region Gallery is not complete: {region_manifest_path}")
        self.region_shard_hashes = {
            Path(item["path"]).name: item["sha256"]
            for item in region_manifest.get("feature_shards", [])
            if isinstance(item, dict) and item.get("path") and item.get("sha256")
        }
        expected_index_hash = region_manifest.get("feature_index_sha256")
        if expected_index_hash and sha256_file(region_index_path) != expected_index_hash:
            raise RuntimeError(f"Region feature index hash mismatch: {region_index_path}")
        self.region_index = read_json(region_index_path)
        sampling_source = manifest.get("source", {})
        if sampling_source.get("region_index_sha256") and sampling_source["region_index_sha256"] != sha256_file(region_index_path):
            raise RuntimeError("Sampling index was built from a different Region feature index")
        if sampling_source.get("region_gallery_manifest_sha256") and sampling_source["region_gallery_manifest_sha256"] != sha256_file(region_manifest_path):
            raise RuntimeError("Sampling index was built from a different Region Gallery manifest")
        self.k = int(manifest["k"])
        self.m = int(manifest["negatives"])
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._shard_cache = OrderedDict()

    def __len__(self):
        return len(self.base)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        return state

    def _load_shard(self, shard_name):
        if shard_name in self._shard_cache:
            self._shard_cache.move_to_end(shard_name)
            return self._shard_cache[shard_name]
        path = self.region_feature_dir / shard_name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = self.region_shard_hashes.get(shard_name)
        if expected_hash and sha256_file(path) != expected_hash:
            raise RuntimeError(f"Region feature shard hash mismatch: {path}")
        payload = torch_load(path)
        features = torch.as_tensor(payload["features"]).float()
        self._shard_cache[shard_name] = features
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return features

    def _gather(self, region_ids, count):
        out = torch.zeros(count, 768, dtype=torch.float32)
        mask = torch.zeros(count, dtype=torch.bool)
        for i, region_id in enumerate(region_ids[:count]):
            try:
                image_text, head_text = str(region_id).rsplit(":", 1)
                image_id, head_id = int(image_text), int(head_text)
            except Exception as exc:
                raise ValueError(f"Invalid region_id {region_id!r}; expected image_id:head_id") from exc
            loc = self.region_index.get(str(image_id))
            if loc is None:
                raise KeyError(f"Unknown image_id in region feature index: {image_id}")
            shard = self._load_shard(loc["shard"])
            row = int(loc["row"])
            if not (0 <= head_id < shard.shape[1]):
                raise IndexError(f"head_id={head_id} outside region shard shape {tuple(shard.shape)}")
            out[i] = shard[row, head_id]
            mask[i] = True
        return out, mask

    def __getitem__(self, idx):
        sample = self.base[idx]
        ann_id = int(sample["metadata"]["annotation_id"])
        ann = self.ann_by_id[ann_id]
        concept = ann.get("novel_cocu_concept")
        record = None
        if concept:
            image_id = int(ann["image_id"])
            key = f"{image_id}::{normalize_concept(concept)}"
            record = self.sampling.get(key)

        if record is None:
            pos = torch.zeros(self.k, 768, dtype=torch.float32)
            neg = torch.zeros(self.m, 768, dtype=torch.float32)
            pos_mask = torch.zeros(self.k, dtype=torch.bool)
            neg_mask = torch.zeros(self.m, dtype=torch.bool)
        else:
            pos, pos_mask = self._gather(record["positive_region_ids"], self.k)
            neg, neg_mask = self._gather(record["negative_region_ids"], self.m)

        sample["local_positive_features"] = pos
        sample["local_negative_features"] = neg
        sample["local_positive_mask"] = pos_mask
        sample["local_negative_mask"] = neg_mask
        return sample
