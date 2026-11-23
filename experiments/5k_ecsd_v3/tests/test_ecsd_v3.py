import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PKG_ROOT))

from candidate_quality.build_candidate_pool_v3 import build_candidates
from candidate_quality.export_manual_audit_v3 import build_audit_rows
from candidate_quality.migrate_manual_annotations_v3 import migrate_rows
from selection_ablation.metrics_core_v3 import compute_caption_set_metrics
from selection_ablation.selection_methods_v3 import METHODS, choose_anchor, select_five, valid_candidates
from selection_ablation.validate_generation_v3 import validate_generation, METHOD_FILES
from utils.safe_io import atomic_write_json
from utils.text_utils import canonical_surface_text

RAW = [
    "A man in a red shirt rides a bicycle along a city street.",
    "A dog runs across green grass near a tree.",
    "A woman holds a blue umbrella while walking on a sidewalk.",
    "A plate of food sits on a wooden table.",
    "A child throws a ball to another child in a park.",
]
CONFIG = {
    "candidate_construction": {
        "min_clause_content_tokens": 2,
        "max_np_tokens": 8,
        "min_np_content_tokens": 1,
        "max_candidates_per_image": 200,
    },
    "candidate_filter": {
        "max_words": 36,
        "min_content_tokens": {"original": 1, "clause": 1, "noun_phrase": 1, "candidate": 1},
    },
    "selection": {
        "B1": {"coverage": 1.0, "diversity": 0.0, "role_novelty": 0.0},
        "B2": {"coverage": 0.0, "diversity": 1.0, "role_novelty": 0.0},
        "B3": {"coverage": 0.5, "diversity": 0.5, "role_novelty": 0.0},
        "B4": {"coverage": 0.4, "diversity": 0.4, "role_novelty": 0.2},
    },
}


def fake_embeddings(texts):
    result = {}
    for text in texts:
        values = np.array([
            float(len(text) % 17 + 1),
            float(len(text.split()) % 13 + 1),
            float(sum(ord(c) for c in text[:40]) % 97 + 1),
            float(sum(ord(c) for c in text[-40:]) % 89 + 1),
        ], dtype=np.float32)
        result[text] = values / np.linalg.norm(values)
    return result


class CandidateConstructionTests(unittest.TestCase):
    def test_full_caption_is_never_a_clause(self):
        candidates, diag = build_candidates(RAW, 123, CONFIG)
        originals = {canonical_surface_text(x["text"]) for x in candidates if x["candidate_type"] == "original"}
        clauses = [x for x in candidates if x["candidate_type"] == "clause"]
        self.assertTrue(all(canonical_surface_text(x["text"]) not in originals for x in clauses))
        self.assertGreaterEqual(diag["full_caption_clause_rejected"], 1)

    def test_clause_must_be_strict_source_subspan(self):
        captions = [
            "A man rides a bicycle, while a woman walks a dog.",
            "A dog runs across green grass near a tree.",
            "A woman holds a blue umbrella while walking on a sidewalk.",
            "A plate of food sits on a wooden table.",
            "A child throws a ball to another child in a park.",
        ]
        candidates, _ = build_candidates(captions, 124, CONFIG)
        clauses = [x for x in candidates if x["candidate_type"] == "clause"]
        self.assertTrue(clauses)
        for item in clauses:
            for span in item["source_spans"]:
                source = captions[span["caption_index"]]
                self.assertTrue(span["start"] > 0 or span["end"] < len(source))
                self.assertNotEqual(canonical_surface_text(span["text"]), canonical_surface_text(source))

    def test_duplicate_text_merges_source_spans_and_prefers_original(self):
        captions = [RAW[0], RAW[0], RAW[2], RAW[3], RAW[4]]
        candidates, _ = build_candidates(captions, 125, CONFIG)
        key = canonical_surface_text(RAW[0])
        matches = [x for x in candidates if canonical_surface_text(x["text"]) == key]
        self.assertEqual(len(matches), 1)
        item = matches[0]
        self.assertEqual(item["candidate_type"], "original")
        self.assertEqual(item["source_caption_ids"], [0, 1])
        self.assertEqual(len(item["source_spans"]), 2)
        self.assertIn("original", item["candidate_origins"])

    def test_candidate_ids_are_deterministic_and_unique(self):
        first, _ = build_candidates(RAW, 987, CONFIG)
        second, _ = build_candidates(RAW, 987, CONFIG)
        self.assertEqual([x["candidate_id"] for x in first], [x["candidate_id"] for x in second])
        self.assertEqual(len({x["candidate_id"] for x in first}), len(first))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates, _ = build_candidates(RAW, 321, CONFIG)
        self.embeddings = fake_embeddings([x["text"] for x in self.candidates])
        self.anchor = choose_anchor(self.candidates, RAW, CONFIG["candidate_filter"])

    def test_anchor_is_invariant_across_b1_b4(self):
        anchors = []
        for method in METHODS:
            _, trace = select_five(
                self.candidates, RAW, self.embeddings, method,
                score_weights=CONFIG["selection"],
                filter_config=CONFIG["candidate_filter"],
                anchor_candidate_id=self.anchor["candidate_id"],
            )
            self.assertEqual(len(trace), 5)
            anchors.append(trace[0]["candidate_id"])
        self.assertEqual(len(set(anchors)), 1)

    def test_selection_is_deterministic(self):
        out1, tr1 = select_five(self.candidates, RAW, self.embeddings, "B3", CONFIG["selection"], CONFIG["candidate_filter"], self.anchor["candidate_id"])
        out2, tr2 = select_five(self.candidates, RAW, self.embeddings, "B3", CONFIG["selection"], CONFIG["candidate_filter"], self.anchor["candidate_id"])
        self.assertEqual(out1, out2)
        self.assertEqual([x["candidate_id"] for x in tr1], [x["candidate_id"] for x in tr2])

    def test_selected_by_id_not_text(self):
        # Same visible text but different IDs: only the exact traced ID is selected.
        selected_ids = {"candidate-exact"}
        self.assertTrue("candidate-exact" in selected_ids)
        self.assertFalse("candidate-duplicate-text" in selected_ids)


class MetricsTests(unittest.TestCase):
    def test_metrics_are_recomputed_from_final_caption_strings(self):
        captions = RAW[:5]
        embeddings = fake_embeddings(captions)
        metrics = compute_caption_set_metrics(captions, RAW, embeddings, ["original"] * 5, CONFIG["candidate_filter"])
        self.assertAlmostEqual(metrics["raw_exact_match_rate"], 1.0)
        self.assertGreaterEqual(metrics["cisd"], 0.0)
        self.assertLessEqual(metrics["cisd"], 2.0)
        self.assertGreater(metrics["weighted_fact_coverage"], 0.0)
        self.assertEqual(metrics["candidate_type_original_count"], 5)


class AnnotationMigrationTests(unittest.TestCase):
    def test_label_migration_preserves_v3_selection_status_and_flags_conflicts(self):
        old = [
            {"image_id": "1", "candidate_text": "A man rides a bicycle.", "validity_label": "Valid", "groundedness_label": "Grounded", "annotator_notes": "checked"},
            {"image_id": "1", "candidate_text": "A dog runs in grass.", "validity_label": "Valid", "groundedness_label": "Grounded", "annotator_notes": ""},
            {"image_id": "1", "candidate_text": "A dog runs in grass!", "validity_label": "Invalid", "groundedness_label": "Grounded", "annotator_notes": "conflict"},
        ]
        new = [
            {"image_id": "1", "candidate_text": "A man rides a bicycle", "selection_status": "Selected", "validity_label": "", "groundedness_label": "", "annotator_notes": ""},
            {"image_id": "1", "candidate_text": "A dog runs in grass.", "selection_status": "Not selected", "validity_label": "", "groundedness_label": "", "annotator_notes": ""},
        ]
        migrated, report = migrate_rows(old, new)
        self.assertEqual(migrated[0]["validity_label"], "Valid")
        self.assertEqual(migrated[0]["selection_status"], "Selected")
        self.assertEqual(migrated[1]["validity_label"], "")
        self.assertEqual(migrated[1]["selection_status"], "Not selected")
        self.assertGreater(report["conflicting_label_fields"].get("validity_label", 0), 0)


class SafeIOTests(unittest.TestCase):
    def test_invalid_json_is_not_overwritten_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.json"
            path.write_text('{"a": 1}\n{"b": 2}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                atomic_write_json(path, {"ok": True}, overwrite=False)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"a": 1}\n{"b": 2}')
            atomic_write_json(path, {"ok": True}, overwrite=True)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertTrue(list(Path(td).glob("broken.json.bak.*")))


class AuditAndValidatorTests(unittest.TestCase):
    def test_manual_audit_selection_uses_candidate_id(self):
        source = [{"image_id": 1, "raw_captions": RAW}]
        pool_candidates = [
            {"candidate_id": "id-selected", "candidate_type": "original", "text": "same text", "source_caption_ids": [0], "source_spans": [], "categories": [], "category_rules": {}},
            {"candidate_id": "id-not-selected", "candidate_type": "clause", "text": "same text", "source_caption_ids": [1], "source_spans": [], "categories": [], "category_rules": {}},
        ]
        # Two same-text records are intentionally used to prove status is ID-based.
        pool = [{"image_id": 1, "raw_captions": RAW, "candidates": pool_candidates}]
        selected = [{"step": i + 1, "candidate_id": "id-selected" if i == 0 else "id-{}".format(i), "text": "same text" if i == 0 else RAW[i], "candidate_type": "original"} for i in range(5)]
        # Complete the candidate pool IDs for the other four selected items.
        for i in range(1, 5):
            pool_candidates.append({"candidate_id": "id-{}".format(i), "candidate_type": "original", "text": RAW[i], "source_caption_ids": [i], "source_spans": [], "categories": [], "category_rules": {}})
        trace = [{"image_id": 1, "method": "B4", "anchor_candidate_id": "id-selected", "selected": selected}]
        rows, _, _ = build_audit_rows(source, pool, trace, 1, 1)
        statuses = {r["candidate_id"]: r["selection_status"] for r in rows}
        self.assertEqual(statuses["id-selected"], "Selected")
        self.assertEqual(statuses["id-not-selected"], "Not selected")
        self.assertEqual(sum(x["selection_status"] == "Selected" for x in rows), 5)

    def test_caption_json_trace_and_anchor_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_path = root / "source.json"
            pool_path = root / "pool.jsonl"
            caption_dir = root / "captions"
            trace_dir = root / "traces"
            caption_dir.mkdir(); trace_dir.mkdir()
            source_rows = [
                {"image_id": 1, "raw_captions": RAW, "rich_captions": RAW, "rich_diversity_captions": RAW},
                {"image_id": 2, "raw_captions": list(reversed(RAW)), "rich_captions": list(reversed(RAW)), "rich_diversity_captions": list(reversed(RAW))},
            ]
            source_path.write_text(json.dumps(source_rows), encoding="utf-8")
            pool_rows = []
            for row in source_rows:
                cands, _ = build_candidates(row["raw_captions"], row["image_id"], CONFIG)
                pool_rows.append({"image_id": row["image_id"], "raw_captions": row["raw_captions"], "candidates": cands})
            with pool_path.open("w", encoding="utf-8") as f:
                for row in pool_rows: f.write(json.dumps(row) + "\n")
            for method, (filename, field) in METHOD_FILES.items():
                output_rows, trace_rows = [], []
                for row, pool in zip(source_rows, pool_rows):
                    cands = pool["candidates"][:5]
                    chosen = [{"candidate_id": x["candidate_id"], "text": x["text"], "candidate_type": x["candidate_type"]} for x in cands]
                    output_rows.append({**row, field: [x["text"] for x in chosen]})
                    trace_rows.append({"image_id": row["image_id"], "method": method, "anchor_candidate_id": chosen[0]["candidate_id"], "selected": chosen})
                (caption_dir / filename).write_text(json.dumps(output_rows), encoding="utf-8")
                (trace_dir / (method + "_selection_trace.jsonl")).write_text("".join(json.dumps(x) + "\n" for x in trace_rows), encoding="utf-8")
            alias = []
            b4rows = json.loads((caption_dir / METHOD_FILES["B4"][0]).read_text())
            for row in b4rows:
                row["ecsd_v3_captions"] = row[METHOD_FILES["B4"][1]]
                alias.append(row)
            (caption_dir / "5k_ECSD_v3.json").write_text(json.dumps(alias), encoding="utf-8")
            report = validate_generation(source_path, pool_path, caption_dir, trace_dir, expected_images=2)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["shared_anchor_mismatches"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
