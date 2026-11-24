from __future__ import annotations
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import torch_load, read_json, normalize_concept
from src.dataset import DinoClipDataset


class RCMTrainingDataset(Dataset):
    def __init__(
        self, train_path, sampling_path, region_features_path, region_index_path,
        features_name="disentangled_self_attn", text_features="ann_feats",
    ):
        self.base = DinoClipDataset(
            train_path, features_name=features_name, text_features=text_features
        )
        raw = torch_load(train_path)
        self.ann_by_id = {int(a["id"]): a for a in raw["annotations"]}
        sampling = read_json(sampling_path)
        self.sampling = sampling["anchors"]
        self.region_features = torch_load(region_features_path).float()
        self.region_index = read_json(region_index_path)
        self.k = int(sampling["meta"]["k"])
        self.m = int(sampling["meta"]["negatives"])

    def __len__(self):
        return len(self.base)

    def _gather(self, region_ids, count):
        out = torch.zeros(count, 768, dtype=torch.float32)
        mask = torch.zeros(count, dtype=torch.bool)
        for i, rid in enumerate(region_ids[:count]):
            if rid not in self.region_index:
                raise KeyError(f"Unknown region_id in sampling index: {rid}")
            row, head = self.region_index[rid]
            out[i] = self.region_features[int(row), int(head)].float()
            mask[i] = True
        return out, mask

    def __getitem__(self, idx):
        sample = self.base[idx]
        ann_id = int(sample["metadata"]["annotation_id"])
        image_id = int(sample["metadata"]["image_id"])
        ann = self.ann_by_id[ann_id]
        concept = ann.get("novel_cocu_concept")
        if concept:
            key = f"{image_id}::{normalize_concept(concept)}"
            record = self.sampling.get(key)
        else:
            record = None
        if record is None:
            pos = torch.zeros(self.k, 768)
            neg = torch.zeros(self.m, 768)
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
