import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

from common.io import normalize_concept, stable_seed


class ConceptTests(unittest.TestCase):
    def test_plural_normalization(self):
        self.assertEqual(normalize_concept("two red dogs"), "two red dog")
        self.assertEqual(normalize_concept("people"), "person")

    def test_stable_seed(self):
        self.assertEqual(stable_seed(123, "17::dog"), stable_seed(123, "17::dog"))
        self.assertNotEqual(stable_seed(123, "17::dog"), stable_seed(124, "17::dog"))


if __name__ == "__main__":
    unittest.main()
