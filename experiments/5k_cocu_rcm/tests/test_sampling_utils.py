import sys
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from data.build_sampling_index import similar_concept, stable_rng


class TestSamplingHelpers(unittest.TestCase):
    def test_overlap_filter(self):
        self.assertTrue(similar_concept("traffic light", "traffic light"))
        self.assertFalse(similar_concept("dog", "bicycle"))

    def test_stable_rng(self):
        self.assertEqual(
            stable_rng(123, "42::bicycle").random(),
            stable_rng(123, "42::bicycle").random()
        )


if __name__ == "__main__":
    unittest.main()
