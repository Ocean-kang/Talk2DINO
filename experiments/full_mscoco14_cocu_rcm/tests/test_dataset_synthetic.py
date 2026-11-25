import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))


class DatasetSmokeTests(unittest.TestCase):
    def test_lazy_region_feature_loading(self):
        old_src = sys.modules.get("src")
        old_src_dataset = sys.modules.get("src.dataset")
        src_mod = types.ModuleType("src")
        src_mod.__path__ = []
        dataset_mod = types.ModuleType("src.dataset")

        class DummyDinoClipDataset:
            def __init__(self, *args, **kwargs):
                self.sample = {
                    "image": torch.zeros(12, 768),
                    "annotation": torch.zeros(512),
                    "metadata": {"annotation_id": 1},
                }

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return dict(self.sample)

        dataset_mod.DinoClipDataset = DummyDinoClipDataset
        sys.modules["src"] = src_mod
        sys.modules["src.dataset"] = dataset_mod
        sys.modules.pop("training.rcm_dataset", None)

        try:
            module = importlib.import_module("training.rcm_dataset")
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                train_path = tmp / "train.pth"
                torch.save({
                    "images": [],
                    "annotations": [{
                        "id": 1, "image_id": 101, "novel_cocu_concept": "dog"
                    }],
                }, train_path)

                region_root = tmp / "region_gallery"
                feature_dir = region_root / "features"
                feature_dir.mkdir(parents=True)
                features = torch.arange(2 * 12 * 768, dtype=torch.float32).reshape(2, 12, 768)
                shard_path = feature_dir / "shard_00000.pt"
                torch.save({"image_ids": [101, 102], "features": features}, shard_path)

                region_index_path = region_root / "region_feature_index.json"
                region_index = {
                    "101": {"shard": "shard_00000.pt", "row": 0},
                    "102": {"shard": "shard_00000.pt", "row": 1},
                }
                region_index_path.write_text(json.dumps(region_index), encoding="utf-8")
                from common.io import sha256_file, atomic_json, atomic_jsonl
                atomic_json(region_root / "region_gallery_manifest.json", {
                    "status": "complete",
                    "feature_index_sha256": sha256_file(region_index_path),
                    "feature_shards": [{
                        "path": "features/shard_00000.pt",
                        "sha256": sha256_file(shard_path),
                    }],
                })

                sampling_dir = tmp / "sampling_index_cross_k2"
                sampling_dir.mkdir()
                rows = [{
                    "key": "101::dog",
                    "positive_region_ids": ["101:0", "102:1"],
                    "negative_region_ids": ["102:2"],
                    "source_image_ids": [101, 102],
                }]
                atomic_jsonl(sampling_dir / "shard_00000.jsonl", rows)
                atomic_json(sampling_dir / "manifest.json", {
                    "status": "complete", "arm": "cross",
                    "source": {"train_augmented_sha256": sha256_file(train_path)},
                    "k": 2, "negatives": 1,
                    "shards": ["shard_00000.jsonl"],
                    "shard_sha256": {
                        "shard_00000.jsonl": sha256_file(sampling_dir / "shard_00000.jsonl")
                    },
                })

                dataset = module.FullRCMTrainingDataset(
                    train_path=train_path,
                    sampling_dir=sampling_dir,
                    region_feature_dir=feature_dir,
                    region_index_path=region_index_path,
                    max_cached_shards=1,
                )
                sample = dataset[0]
                self.assertEqual(tuple(sample["local_positive_features"].shape), (2, 768))
                self.assertEqual(tuple(sample["local_negative_features"].shape), (1, 768))
                self.assertTrue(sample["local_positive_mask"].all())
                self.assertTrue(sample["local_negative_mask"].all())
                self.assertTrue(torch.equal(sample["local_positive_features"][0], features[0, 0]))
                self.assertTrue(torch.equal(sample["local_positive_features"][1], features[1, 1]))
        finally:
            sys.modules.pop("training.rcm_dataset", None)
            if old_src is None:
                sys.modules.pop("src", None)
            else:
                sys.modules["src"] = old_src
            if old_src_dataset is None:
                sys.modules.pop("src.dataset", None)
            else:
                sys.modules["src.dataset"] = old_src_dataset


if __name__ == "__main__":
    unittest.main()
