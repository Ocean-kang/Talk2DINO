import sys
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from candidate_quality.semantic_role_classifier import content_tokens, classify_semantic_roles
from candidate_quality.build_candidate_pool_v3 import build_candidates
from candidate_quality.export_manual_audit_v3 import build_audit_rows
from selection_ablation.selection_methods_v3 import METHODS, choose_anchor, select_five
from selection_ablation.metrics_core_v3 import compute_caption_set_metrics
from selection_ablation.validate_generation_v3 import validate_generation, METHOD_FILES

RAW = [
    "A man rides a bicycle near a tree.",
    "A dog runs across green grass.",
    "A woman holds a blue umbrella.",
    "A plate of food sits on a wooden table.",
    "A child throws a ball in a park.",
]
FILTER = {"max_words": 36, "min_content_tokens": {
    "original": 2, "clause": 2, "noun_phrase": 1, "candidate": 2
}}
WEIGHTS = {
    "B1": {"coverage": 1.0, "diversity": 0.0, "role_novelty": 0.0},
    "B2": {"coverage": 0.0, "diversity": 1.0, "role_novelty": 0.0},
    "B3": {"coverage": 0.5, "diversity": 0.5, "role_novelty": 0.0},
    "B4": {"coverage": 0.4, "diversity": 0.4, "role_novelty": 0.2},
}


def candidate(i, text, kind="original"):
    return {
        "candidate_id": "c{}".format(i), "text": text, "candidate_type": kind,
        "facts": sorted(content_tokens(text)),
        "categories": classify_semantic_roles(text)["categories"],
        "source_caption_ids": [min(i, 4)], "source_spans": [],
    }


def make_embeddings(candidates):
    result = {}
    for x in candidates:
        v = np.array([len(x["text"]) % 11 + 1, len(x["text"].split()) + 1,
                      sum(map(ord, x["text"])) % 97 + 1], dtype=np.float32)
        result[x["text"]] = v / np.linalg.norm(v)
    return result


class FallbackSelectionTests(unittest.TestCase):
    def test_five_normal(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW)]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 5)
        self.assertTrue(all(x["fallback_type"] == "none" for x in trace))

    def test_relaxed_candidate_fills_fifth(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4] + ["and a dog"])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B1", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 5)
        self.assertEqual(trace[-1]["fallback_type"], "relaxed_quality_filter")

    def test_four_distinct_use_duplicate_slot(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B2", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["slot_id"] for x in trace}), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 4)
        self.assertEqual(trace[-1]["fallback_type"], "duplicate_slot")

    def test_one_candidate_fills_five_slots(self):
        cs = [candidate(0, RAW[0])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B4", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 1)
        self.assertEqual(len({x["slot_id"] for x in trace}), 5)
        self.assertEqual(sum(x["fallback_type"] == "duplicate_slot" for x in trace), 4)

    def test_all_raw_duplicates_do_not_crash(self):
        cs = [candidate(0, RAW[0])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, [RAW[0]] * 5, FILTER)
        out, trace = select_five(cs, [RAW[0]] * 5, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(sum(x["fallback_type"] == "duplicate_slot" for x in trace), 4)

    def test_b1_b4_shared_precomputed_anchor(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        anchors = []
        for method in METHODS:
            _, trace = select_five(cs, RAW, emb, method, WEIGHTS, FILTER, anchor["candidate_id"])
            anchors.append(trace[0]["candidate_id"])
        self.assertEqual(len(set(anchors)), 1)

    def test_selection_is_deterministic(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        self.assertEqual(
            select_five(cs, RAW, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"]),
            select_five(cs, RAW, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"]),
        )

    def test_cisd_keeps_duplicate_slots(self):
        repeated = [RAW[0]] * 5
        emb = make_embeddings([candidate(0, RAW[0])])
        metrics = compute_caption_set_metrics(
            repeated, RAW, emb, ["original"] * 5, FILTER,
            fallback_types=["none", "duplicate_slot", "duplicate_slot", "duplicate_slot", "duplicate_slot"],
        )
        self.assertAlmostEqual(metrics["cisd"], 0.0, places=6)
        self.assertEqual(metrics["fallback_duplicate_count"], 4)
        self.assertAlmostEqual(metrics["fallback_duplicate_rate"], 0.8)


class ValidatorDuplicateSlotTests(unittest.TestCase):
    def test_trace_and_json_accept_explicit_duplicate_slots(self):
        raw = [RAW[0]] * 5
        config = {
            "candidate_construction": {"min_clause_content_tokens": 2, "max_np_tokens": 8,
                                       "min_np_content_tokens": 1, "max_candidates_per_image": 100},
            "candidate_filter": FILTER,
        }
        candidates, _ = build_candidates(raw, 77, config)
        chosen = next(x for x in candidates if x["candidate_type"] == "original")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_path, pool_path = root / "source.json", root / "pool.jsonl"
            cap_dir, trace_dir = root / "captions", root / "traces"
            cap_dir.mkdir()
            trace_dir.mkdir()
            source_row = {"image_id": 77, "raw_captions": raw,
                          "rich_captions": raw, "rich_diversity_captions": raw}
            source_path.write_text(json.dumps([source_row]), encoding="utf-8")
            pool_path.write_text(json.dumps({"image_id": 77, "raw_captions": raw,
                                             "candidates": candidates}) + "\n", encoding="utf-8")
            for method, (filename, field) in METHOD_FILES.items():
                selected = []
                for slot in range(1, 6):
                    selected.append({
                        "step": slot, "slot_id": "slot_{:02d}".format(slot),
                        "candidate_id": chosen["candidate_id"], "text": chosen["text"],
                        "candidate_type": "original",
                        "fallback_type": "none" if slot == 1 else "duplicate_slot",
                        "duplicate_of_candidate_id": None if slot == 1 else chosen["candidate_id"],
                    })
                row = dict(source_row)
                row[field] = [chosen["text"]] * 5
                (cap_dir / filename).write_text(json.dumps([row]), encoding="utf-8")
                trace = {"image_id": 77, "method": method,
                         "anchor_candidate_id": chosen["candidate_id"], "selected": selected}
                (trace_dir / (method + "_selection_trace.jsonl")).write_text(
                    json.dumps(trace) + "\n", encoding="utf-8")
            alias_row = dict(source_row)
            alias_row["ecsd_v3_captions"] = [chosen["text"]] * 5
            (cap_dir / "5k_ECSD_v3.json").write_text(json.dumps([alias_row]), encoding="utf-8")
            report = validate_generation(source_path, pool_path, cap_dir, trace_dir, expected_images=1)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["duplicate_fallback_slots"], 16)


    def test_manual_audit_accepts_duplicate_slots_and_reports_multiplicity(self):
        raw = [RAW[0]] * 5
        config = {
            "candidate_construction": {"min_clause_content_tokens": 2, "max_np_tokens": 8,
                                       "min_np_content_tokens": 1, "max_candidates_per_image": 100},
            "candidate_filter": FILTER,
        }
        candidates, _ = build_candidates(raw, 88, config)
        chosen = next(x for x in candidates if x["candidate_type"] == "original")
        source_rows = [{"image_id": 88, "raw_captions": raw}]
        pool_rows = [{"image_id": 88, "raw_captions": raw, "candidates": candidates}]
        selected = []
        for slot in range(1, 6):
            selected.append({
                "step": slot, "slot_id": "slot_{:02d}".format(slot),
                "candidate_id": chosen["candidate_id"], "text": chosen["text"],
                "candidate_type": "original",
                "fallback_type": "none" if slot == 1 else "duplicate_slot",
                "duplicate_of_candidate_id": None if slot == 1 else chosen["candidate_id"],
            })
        trace_rows = [{"image_id": 88, "method": "B4",
                       "anchor_candidate_id": chosen["candidate_id"], "selected": selected}]
        rows, viewer, _ = build_audit_rows(source_rows, pool_rows, trace_rows, 1, 1)
        chosen_row = next(x for x in rows if x["candidate_id"] == chosen["candidate_id"])
        self.assertEqual(chosen_row["selection_status"], "Selected")
        self.assertEqual(chosen_row["selected_slot_multiplicity"], 5)
        self.assertEqual(viewer[0]["selected_slot_count"], 5)
        self.assertEqual(viewer[0]["duplicate_slot_count"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

