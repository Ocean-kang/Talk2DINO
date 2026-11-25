import importlib.util
import sys
import unittest
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

path = EXP / "data" / "build_sampling_index.py"
spec = importlib.util.spec_from_file_location("build_sampling_index", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SamplingTests(unittest.TestCase):
    def test_negative_sampler_excludes_anchor_and_duplicates(self):
        regions = ["1:0", "1:1", "2:0", "2:1", "3:0"]
        tags = defaultdict(set, {
            "1:0": {"cat"}, "1:1": {"dog"},
            "2:0": {"cat"}, "2:1": {"bird"}, "3:0": {"horse"},
        })
        result = module.sample_negative_regions(
            __import__("random").Random(7), regions, tags, "cat", {"1:1"}, 2
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(set(result)), 2)
        self.assertTrue(all("cat" not in tags[x] for x in result))
        self.assertNotIn("1:1", result)

    def test_source_images_are_distinct_and_deterministic(self):
        candidates = [
            (10, ["10:0"], ["10:1"]),
            (11, ["11:0"], ["11:1"]),
            (12, ["12:0"], ["12:1"]),
        ]
        first = module.sample_source_images(__import__("random").Random(99), candidates, 10, 2)
        second = module.sample_source_images(__import__("random").Random(99), candidates, 10, 2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len({x[0] for x in first}), 2)
        self.assertNotIn(10, [x[0] for x in first])

    def test_frequency_buckets(self):
        self.assertEqual(module.frequency_bucket(1), "rare")
        self.assertEqual(module.frequency_bucket(10), "medium")
        self.assertEqual(module.frequency_bucket(100), "frequent")


if __name__ == "__main__":
    unittest.main()
