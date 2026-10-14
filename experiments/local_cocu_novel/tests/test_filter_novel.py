import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from filter_novel import in_raw_captions, filter_one, run_filter


class NovelConceptTests(unittest.TestCase):
    def test_boundary_and_phrase(self):
        raw = ["A man has a brown dog near the swimming pool."]
        self.assertTrue(in_raw_captions("dog", raw))
        self.assertTrue(in_raw_captions("swimming pool", raw))
        self.assertFalse(in_raw_captions("cat", raw))
        self.assertFalse(in_raw_captions("swimming tree", raw))
        self.assertFalse(in_raw_captions("man", ["A woman is walking."]))
        self.assertFalse(in_raw_captions("car", ["A carpet is red."]))

    def test_plural_and_irregular(self):
        self.assertTrue(in_raw_captions("dog", ["two dogs on grass"]))
        self.assertTrue(in_raw_captions("woman", ["several women stand"]))
        self.assertTrue(in_raw_captions("person", ["some people are outside"]))
        self.assertTrue(in_raw_captions("puppy", ["the puppies run"]))
        self.assertFalse(in_raw_captions("dog", ["two dogs"], exact_only=True))

    def test_removed_before_selection(self):
        keep, removed = filter_one(["DOG", "pool", "duck", "dog", "red car", "cat"],
                                   ["a dog and a red car stand together"])
        self.assertEqual(keep, ["pool", "duck", "cat"])
        self.assertEqual(removed, ["dog", "red car"])

    def test_full_filter_io(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = {"images": [{"id": 1}], "annotations": [
                {"image_id": 1, "caption": t} for t in [
                    "A dog", "It is sunny", "Grass", "Red ball", "a tree"]]}
            (td / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
            (td / "candidates.json").write_text(json.dumps({"1": ["DOG", "dog", "cat", "tree"]}), encoding="utf-8")
            run_filter(td / "candidates.json", td / "raw.json", td / "out.json", td / "audit.json")
            self.assertEqual(json.loads((td / "out.json").read_text()), {"1": ["cat"]})
            audit = json.loads((td / "audit.json").read_text())
            self.assertEqual(audit["summary"]["removed"], 2)


if __name__ == "__main__":
    unittest.main()
