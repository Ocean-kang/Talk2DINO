import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]


class FullArchiveGuardTests(unittest.TestCase):
    def test_explicit_smoke_population_passes_but_default_full_guard_rejects_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = tmp / "train.pth"
            matches = tmp / "matches.json"
            torch.save({
                "images": [{"id": 101, "file_name": "COCO_train2014_000000000101.jpg"}],
                "annotations": [
                    {"id": i, "image_id": 101, "caption": f"caption {i}"}
                    for i in range(1, 6)
                ],
            }, base)
            matches.write_text(json.dumps({"101": []}), encoding="utf-8")
            command = [
                sys.executable, str(EXP / "data" / "build_full_concept_archive.py"),
                "--base-train", str(base), "--matches", str(matches),
                "--out-dir", str(tmp / "archive"), "--check-only",
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(EXP)
            smoke = subprocess.run(command + ["--expected-images", "1"], capture_output=True, text=True, env=env)
            self.assertEqual(smoke.returncode, 0, msg=smoke.stdout + smoke.stderr)
            full_guard = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(full_guard.returncode, 0)
            self.assertIn("Expected full Train2014 population", full_guard.stderr)


if __name__ == "__main__":
    unittest.main()
