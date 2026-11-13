import unittest

from experiments.role_aware_concept_composition.core import (
    build_candidate_pool, compose_one_image, contains_phrase, phrase_mentions,
    rank_normalize, _find_relations, _find_actions, _find_attributes,
)

class CoreTests(unittest.TestCase):
    def test_phrase_matching_and_plural_normalization(self):
        self.assertTrue(contains_phrase("Two people are riding bicycles.", "person"))
        found = phrase_mentions("A football player is near a bicycle.", ["football player", "player", "bicycle"])
        self.assertEqual([x["concept"] for x in found], ["football player", "bicycle"])

    def test_rank_normalization_is_source_local(self):
        ranks = rank_normalize([
            {"concept": "helmet", "score": 0.91},
            {"concept": "tree", "score": 0.42},
            {"concept": "road", "score": 0.18},
        ])
        self.assertEqual(ranks["helmet"], 1.0)
        self.assertAlmostEqual(ranks["tree"], 0.5)
        self.assertEqual(ranks["road"], 0.0)

    def test_action_extraction(self):
        rows = _find_actions(
            ["A person is riding a bicycle.", "A man rides a bicycle."],
            ["person", "man", "bicycle"],
        )
        self.assertTrue(any(x["action"] == "riding" and x["target"] == "bicycle" for x in rows))

    def test_attribute_extraction(self):
        rows = _find_attributes(
            ["A person wearing a red shirt is standing."],
            ["person", "shirt"],
        )
        self.assertTrue(any(x["attribute"] == "red" and x["object"] == "shirt" for x in rows))

    def test_relation_requires_multiple_caption_supports(self):
        captions = [
            "A cat is on a sofa.",
            "A cat sits on a sofa.",
            "A cat looks at a person.",
        ]
        rows = _find_relations(captions, ["cat", "sofa", "person"], min_support=2)
        self.assertTrue(any(x["subject"] == "cat" and x["object"] == "sofa" and x["relation"] == "on" for x in rows))

    def test_composition_exactly_five_and_auditable(self):
        raw = [
            "A person is riding a bicycle on a street.",
            "A man rides a bicycle near a tree.",
            "A cyclist wearing a red shirt is on the road.",
            "A person with a helmet rides past a building.",
            "A bicycle is parked beside a tree.",
        ]
        sources = {
            "f5": [
                {"concept": "helmet", "score": 0.9, "best_head": 2},
                {"concept": "tree", "score": 0.8, "best_head": 4},
                {"concept": "building", "score": 0.7, "best_head": 1},
                {"concept": "street", "score": 0.6, "best_head": 7},
            ],
            "cocu": [
                {"concept": "street", "score": 0.88},
                {"concept": "tree", "score": 0.66},
            ],
        }
        result = compose_one_image(
            7, raw, sources,
            gallery_terms=["person", "man", "bicycle", "tree", "street", "road", "helmet", "building", "shirt"],
            config={"min_relation_support": 1, "beam_width": 16},
        )
        self.assertEqual(len(result["captions"]), 5)
        self.assertEqual([x["role"] for x in result["roles"]], ["C1", "C2", "C3", "C4", "C5"])
        self.assertTrue(all(x["caption"].endswith(".") for x in result["roles"]))
        self.assertGreaterEqual(result["diagnostics"]["candidate_count"], 3)

if __name__ == "__main__":
    unittest.main()
