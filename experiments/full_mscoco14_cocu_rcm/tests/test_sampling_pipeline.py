import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

from common.io import atomic_json, sha256_file


class SamplingPipelineTests(unittest.TestCase):
    def test_paired_rcm_shuffled_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_path = tmp / "train_augmented.pth"
            torch.save({
                "images": [],
                "annotations": [{
                    "id": 11, "image_id": 1, "novel_cocu_concept": "dog"
                }],
            }, train_path)

            region_root = tmp / "region_gallery"
            region_root.mkdir()
            region_index_path = region_root / "region_feature_index.json"
            region_index = {
                "1": {"shard": "shard_00000.pt", "row": 0},
                "2": {"shard": "shard_00000.pt", "row": 1},
                "3": {"shard": "shard_00000.pt", "row": 2},
            }
            region_index_path.write_text(json.dumps(region_index), encoding="utf-8")
            atomic_json(region_root / "region_gallery_manifest.json", {
                "status": "complete",
                "feature_index_sha256": sha256_file(region_index_path),
            })

            verified_dir = tmp / "region_verification" / "verified_region_concepts"
            verified_dir.mkdir(parents=True)
            shard = [
                {"image_id": 1, "verified_concepts": [
                    {"normalized_concept": "dog", "regions": [
                        {"region_id": "1:0"}, {"region_id": "1:1"}]},
                    {"normalized_concept": "cat", "regions": [{"region_id": "1:2"}]},
                ]},
                {"image_id": 2, "verified_concepts": [
                    {"normalized_concept": "dog", "regions": [{"region_id": "2:0"}]},
                    {"normalized_concept": "cat", "regions": [{"region_id": "2:1"}]},
                    {"normalized_concept": "horse", "regions": [{"region_id": "2:2"}]},
                ]},
                {"image_id": 3, "verified_concepts": [
                    {"normalized_concept": "dog", "regions": [{"region_id": "3:0"}]},
                    {"normalized_concept": "bird", "regions": [{"region_id": "3:1"}]},
                    {"normalized_concept": "horse", "regions": [{"region_id": "3:2"}]},
                ]},
            ]
            shard_path = verified_dir / "shard_00000.json"
            atomic_json(shard_path, shard)
            atomic_json(verified_dir.parent / "verification_manifest.json", {
                "status": "complete",
                "shard_sha256": {"shard_00000.json": sha256_file(shard_path)},
            })

            out_dir = tmp / "work"
            command = [
                sys.executable, str(EXP / "data" / "build_sampling_index.py"),
                "--train-augmented", str(train_path),
                "--verified-dir", str(verified_dir),
                "--region-index", str(region_index_path),
                "--out-dir", str(out_dir),
                "--k", "2", "--negatives", "2", "--seed", "123",
                "--shard-size", "1",
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(EXP) + os.pathsep + str(ROOT)
            first = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)

            cross_dir = out_dir / "sampling_index_cross_k2"
            shuffled_dir = out_dir / "sampling_index_shuffled_k2"
            cross = json.loads((cross_dir / "shard_00000.jsonl").read_text().strip())
            shuffled = json.loads((shuffled_dir / "shard_00000.jsonl").read_text().strip())
            self.assertEqual(cross["key"], shuffled["key"])
            self.assertEqual(cross["source_image_ids"], shuffled["source_image_ids"])
            self.assertEqual(cross["negative_region_ids"], shuffled["negative_region_ids"])
            self.assertNotEqual(cross["positive_region_ids"], shuffled["positive_region_ids"])
            self.assertEqual(len(set(cross["source_image_ids"])), 2)
            self.assertNotIn(1, cross["source_image_ids"])

            stats = json.loads((out_dir / "sampling_statistics.json").read_text())
            self.assertTrue(stats["rcm_shuffled_anchor_set_identical"])
            self.assertTrue(stats["rcm_shuffled_source_ids_identical"])
            self.assertTrue(stats["rcm_shuffled_negative_ids_identical"])
            self.assertEqual(stats["cross_eligible"], 1)
            self.assertEqual(stats["shuffled_eligible"], 1)

            # A second invocation resumes by validating and skipping the completed shard.
            second = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(second.returncode, 0, msg=second.stdout + second.stderr)
            self.assertIn("[RESUME]", second.stdout)


if __name__ == "__main__":
    unittest.main()
