import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "ecsd_core.py"
SPEC = importlib.util.spec_from_file_location("ecsd_core", MODULE_PATH)
ecsd_core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ecsd_core)


class ECSDBaselineTests(unittest.TestCase):
    def setUp(self):
        self.captions = [
            "A brown dog is chasing a blue frisbee in the grass.",
            "A dog runs across a green field near a frisbee.",
            "A brown and white dog plays with a toy outside.",
            "A person watches the dog playing on the grass.",
            "The blue and yellow frisbee is beside the dog.",
        ]

    def test_keeps_all_original_captions(self):
        candidates = ecsd_core.build_candidates(self.captions)
        originals = [x for x in candidates if x["candidate_type"] == "original"]
        self.assertEqual(len(originals), 5)

    def test_candidates_are_traceable(self):
        candidates = ecsd_core.build_candidates(self.captions)
        source_facts = set().union(*(ecsd_core.content_tokens(x) for x in self.captions))
        for candidate in candidates:
            self.assertTrue(candidate["source_caption_ids"])
            self.assertTrue(set(candidate["facts"]).issubset(source_facts))
            for span in candidate["source_spans"]:
                self.assertLessEqual(span["start"], span["end"])

    def test_phrase_candidates_exist(self):
        candidates = ecsd_core.build_candidates(self.captions)
        self.assertTrue(any(x["candidate_type"] == "noun_phrase" for x in candidates))

    def test_fact_weights_are_positive(self):
        _, weights = ecsd_core.build_fact_weights(self.captions)
        self.assertTrue(weights)
        self.assertTrue(all(value > 0 for value in weights.values()))


if __name__ == "__main__":
    unittest.main()
