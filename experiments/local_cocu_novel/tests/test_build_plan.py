import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_train import plan


class BuildPlanTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "images": [{"id": 42, "disentangled_self_attn": "dummy-feature"}],
            "annotations": [
                {"id": i, "image_id": 42, "caption": cap, "ann_feats": "dummy-feature"}
                for i, cap in enumerate(["a dog plays", "grass", "a blue sky", "park", "a person"], 1)
            ],
        }

    def test_skip_known_concepts_and_preserve_raw(self):
        matches = {"42": [
            {"concept": "dog", "score": .99, "best_head": 0},
            {"concept": "cat", "score": .82, "best_head": 1},
            {"concept": "duck", "score": .72, "best_head": 2},
            {"concept": "horse", "score": .52, "best_head": 3},
        ]}
        raw, supplemental, hist, deficits = plan(self.base, matches, 3)
        self.assertEqual(len(raw), 5)
        self.assertEqual([x["concept"] for x in supplemental], ["cat", "duck", "horse"])
        self.assertEqual(hist[3], 1)

    def test_less_than_three_is_explicit(self):
        matches = {"42": [{"concept": "dog", "score": .9, "best_head": 0},
                          {"concept": "cat", "score": .5, "best_head": 1}]}
        _, extra, hist, deficits = plan(self.base, matches, 3)
        self.assertEqual(len(extra), 1)
        self.assertEqual(hist[1], 1)
        with self.assertRaises(RuntimeError):
            plan(self.base, matches, 3, strict=True)


if __name__ == "__main__":
    unittest.main()
