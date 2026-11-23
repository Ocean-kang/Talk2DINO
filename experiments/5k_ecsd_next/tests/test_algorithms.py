#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

NEXT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEXT_ROOT))

from candidate_quality.improve_candidate_construction import build_improved_candidates
from candidate_quality.semantic_role_classifier import (
    classify_semantic_roles, content_tokens, quality_flags, normalize_text,
)
from selection_ablation.selection_methods import METHODS, select_five

RAW = [
    "A man in a red shirt rides a bicycle along a city street.",
    "A dog runs across green grass near a tree.",
    "A woman holds a blue umbrella while walking on a sidewalk.",
    "A plate of food sits on a wooden table.",
    "A child throws a ball to another child in a park.",
]

class CandidateConstructionTests(unittest.TestCase):
    def test_candidates_are_nonempty_and_traceable(self):
        candidates = build_improved_candidates(RAW)
        self.assertGreaterEqual(len(candidates), 5)
        for item in candidates:
            self.assertTrue(item["text"])
            self.assertTrue(item["source_spans"])
            span = item["source_spans"][0]["text"]
            self.assertTrue(span)
            self.assertTrue(content_tokens(item["text"]).issubset(content_tokens(span)))

    def test_noun_phrase_wrapper_adds_no_new_content_facts(self):
        candidates = build_improved_candidates(["A man wears a red shirt."] * 5)
        for item in candidates:
            span = item["source_spans"][0]["text"]
            self.assertTrue(content_tokens(item["text"]).issubset(content_tokens(span)))

    def test_conjunction_inside_compound_subject_is_not_split(self):
        candidates = build_improved_candidates(["A man and a woman stand together."] * 5)
        clauses = [x for x in candidates if x["candidate_type"] == "clause"]
        self.assertFalse(any("A man." == x["text"] or "A woman." == x["text"] for x in clauses))

class SemanticRoleTests(unittest.TestCase):
    def test_multilabel_roles(self):
        result = classify_semantic_roles("A red dog runs across green grass.")
        self.assertIn("Entity", result["categories"])
        self.assertIn("Attribute", result["categories"])
        self.assertIn("Action", result["categories"])
        self.assertIn("Relation", result["categories"])
        self.assertIn("Context", result["categories"])

    def test_unknown_is_explicit(self):
        result = classify_semantic_roles("perhaps somehow")
        self.assertEqual(result["categories"], ["Unknown"])

    def test_invalid_phrase_flags(self):
        flags = quality_flags("and with", "clause")
        self.assertIn("leading_connector", flags)
        self.assertIn("dangling_connector_or_preposition", flags)

    def test_normalization(self):
        self.assertEqual(normalize_text("A man rides."), normalize_text("The man ride."))

class SelectionAblationTests(unittest.TestCase):
    def test_all_methods_share_anchor_and_return_five(self):
        candidates = build_improved_candidates(RAW)
        embedding = {}
        for item in candidates:
            text = item["text"]
            facts = content_tokens(text)
            embedding[text] = [
                float(len(facts) + 1),
                float(len(text.split()) + 1),
                float(sum(ord(c) for c in text[:20]) % 97 + 1),
            ]
        anchors = set()
        for method in METHODS:
            selected, trace = select_five(candidates, RAW, embedding, method)
            self.assertEqual(len(selected), 5)
            self.assertEqual(len(trace), 5)
            anchors.add(trace[0]["candidate_id"])
        self.assertEqual(len(anchors), 1)

    def test_fallback_preserves_exactly_five_contract(self):
        candidate = {
            "candidate_id": "only", "text": RAW[0], "candidate_type": "original",
            "facts": sorted(content_tokens(RAW[0])), "categories": ["Entity"],
            "quality_flags": [], "source_caption_ids": [0], "source_spans": [],
        }
        embedding = {RAW[0]: [1.0, 0.0, 0.0]}
        selected, trace = select_five([candidate], RAW, embedding, "B1")
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(trace), 5)

if __name__ == "__main__":
    unittest.main()
