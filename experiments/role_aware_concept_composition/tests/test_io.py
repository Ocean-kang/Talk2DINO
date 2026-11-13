import json
import tempfile
import unittest
from pathlib import Path

from experiments.role_aware_concept_composition.io_utils import load_source

class SourceAdapterTests(unittest.TestCase):
    def test_cocu_list_manifest(self):
        payload = {
            "meta": {},
            "images": [{
                "image_id": 101,
                "curated": [{"concept": "helmet", "score": 0.8}]
            }]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = load_source(path, "cocu")
        self.assertEqual(result[101][0]["concept"], "helmet")

    def test_local_cocu_image_keyed_manifest(self):
        payload = {
            "meta": {},
            "images": {
                "202": {
                    "selected_concepts": [{"concept": "tree", "score": 0.7}]
                }
            }
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = load_source(path, "local_cocu")
        self.assertEqual(result[202][0]["concept"], "tree")

    def test_f5_jsonl_audit(self):
        rows = [
            {"image_id": 303, "candidates": [{"concept": "road", "target_score": 0.6}]},
            {"image_id": 304, "candidates": [{"concept": "tree", "target_score": 0.5}]},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "audit.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            result = load_source(path, "f5")
        self.assertEqual(result[303][0]["concept"], "road")
        self.assertEqual(result[304][0]["concept"], "tree")

if __name__ == "__main__":
    unittest.main()
