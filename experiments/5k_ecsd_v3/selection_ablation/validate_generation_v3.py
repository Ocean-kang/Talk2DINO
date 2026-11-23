#!/usr/bin/env python3
"""Strictly validate v3 candidate pool, caption JSONs, traces, and shared anchors."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
from utils.safe_io import atomic_write_json
from utils.text_utils import canonical_surface_text
from candidate_quality.build_candidate_pool_v3 import candidate_id_for

METHOD_FILES = {
    "B1": ("5k_ECSD_v3_B1_Coverage.json", "ecsd_v3_b1_captions"),
    "B2": ("5k_ECSD_v3_B2_Diversity.json", "ecsd_v3_b2_captions"),
    "B3": ("5k_ECSD_v3_B3_Coverage_Diversity.json", "ecsd_v3_b3_captions"),
    "B4": ("5k_ECSD_v3_B4_Coverage_Diversity_Role.json", "ecsd_v3_b4_captions"),
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    result = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("{} line {} invalid JSON: {}".format(path, line_no, exc))
    return result


def validate_generation(source_path, pool_path, caption_dir, trace_dir,
                        expected_images=5000, methods=None, v3_alias=True):
    methods = methods or ["B1", "B2", "B3", "B4"]
    source_rows = load_json(source_path)
    source_map = {int(x["image_id"]): x for x in source_rows}
    if len(source_map) != len(source_rows):
        raise ValueError("Source has duplicate image IDs")
    if expected_images and len(source_map) != expected_images:
        raise ValueError("Expected {} source images, got {}".format(expected_images, len(source_map)))
    for image_id, row in source_map.items():
        raw = row.get("raw_captions")
        if not isinstance(raw, list) or len(raw) != 5 or not all(isinstance(x, str) and x.strip() for x in raw):
            raise ValueError("Source image_id={} must have five non-empty Raw Captions".format(image_id))

    pool_rows = load_jsonl(pool_path)
    pool_map = {int(x["image_id"]): x for x in pool_rows}
    if len(pool_map) != len(pool_rows) or set(pool_map) != set(source_map):
        raise ValueError("Candidate pool image IDs do not exactly match source")
    candidate_count = 0
    for image_id, row in pool_map.items():
        if row.get("raw_captions") != source_map[image_id].get("raw_captions"):
            raise ValueError("Raw Caption mismatch in candidate pool for image_id={}".format(image_id))
        candidates = row.get("candidates", [])
        ids = [x.get("candidate_id") for x in candidates]
        surfaces = [canonical_surface_text(x.get("text", "")) for x in candidates]
        if len(ids) != len(set(ids)) or len(surfaces) != len(set(surfaces)):
            raise ValueError("Duplicate candidate IDs/texts for image_id={}".format(image_id))
        raw_captions = row["raw_captions"]
        for candidate in candidates:
            expected_id = candidate_id_for(image_id, canonical_surface_text(candidate.get("text", "")))
            if candidate.get("candidate_id") != expected_id:
                raise ValueError("Non-deterministic candidate_id for image_id={}".format(image_id))
            span_ids = set()
            for span in candidate.get("source_spans", []):
                caption_index = int(span["caption_index"])
                start, end = int(span["start"]), int(span["end"])
                if caption_index < 0 or caption_index >= 5 or start < 0 or end <= start or end > len(raw_captions[caption_index]):
                    raise ValueError("Invalid source span bounds for image_id={}".format(image_id))
                if raw_captions[caption_index][start:end] != span.get("text"):
                    raise ValueError("Source span text mismatch for image_id={}".format(image_id))
                span_ids.add(caption_index)
            if not candidate.get("source_spans") or sorted(span_ids) != candidate.get("source_caption_ids"):
                raise ValueError("Candidate source IDs/spans are incomplete for image_id={}".format(image_id))
        candidate_count += len(candidates)

    trace_maps = {}
    anchor_mismatch_count = 0
    for method in methods:
        trace_path = Path(trace_dir) / (method + "_selection_trace.jsonl")
        trace_rows = load_jsonl(trace_path)
        trace_map = {int(x["image_id"]): x for x in trace_rows}
        if len(trace_map) != len(trace_rows) or set(trace_map) != set(source_map):
            raise ValueError("{} trace image IDs do not match source".format(method))
        for image_id, trace_row in trace_map.items():
            if trace_row.get("method") != method:
                raise ValueError("Trace method mismatch for {} image_id={}".format(method, image_id))
            selected = trace_row.get("selected", [])
            if len(selected) != 5:
                raise ValueError("{} image_id={} has {} selected trace items".format(method, image_id, len(selected)))
            slot_ids = [x.get("slot_id", "slot_{:02d}".format(int(x.get("step", i + 1)))) for i, x in enumerate(selected)]
            if len(slot_ids) != 5 or len(set(slot_ids)) != 5:
                raise ValueError("{} image_id={} must have five unique slot_id values".format(method, image_id))
            candidate_map = {x["candidate_id"]: x for x in pool_map[image_id]["candidates"]}
            seen_ids = set()
            for item in selected:
                cid = item.get("candidate_id")
                if cid not in candidate_map:
                    raise ValueError("{} image_id={} trace references unknown candidate_id {}".format(method, image_id, cid))
                if item.get("text") != candidate_map[cid].get("text"):
                    raise ValueError("{} image_id={} trace text differs from candidate pool for {}".format(method, image_id, cid))
                fallback_type = item.get("fallback_type", "none")
                if cid in seen_ids:
                    if fallback_type != "duplicate_slot" or item.get("duplicate_of_candidate_id") != cid:
                        raise ValueError("{} image_id={} repeated candidate_id {} without explicit duplicate_slot trace".format(method, image_id, cid))
                elif fallback_type == "duplicate_slot":
                    raise ValueError("{} image_id={} marks first occurrence of {} as duplicate_slot".format(method, image_id, cid))
                seen_ids.add(cid)
            if trace_row.get("anchor_candidate_id") != selected[0].get("candidate_id"):
                raise ValueError("{} image_id={} anchor field differs from first selected item".format(method, image_id))
        trace_maps[method] = trace_map

        filename, field = METHOD_FILES[method]
        rows = load_json(Path(caption_dir) / filename)
        row_map = {int(x["image_id"]): x for x in rows}
        if len(row_map) != len(rows) or set(row_map) != set(source_map):
            raise ValueError("{} caption JSON image IDs do not match source".format(method))
        for image_id, row in row_map.items():
            if row.get("raw_captions") != source_map[image_id].get("raw_captions"):
                raise ValueError("{} raw captions mismatch for image_id={}".format(method, image_id))
            captions = row.get(field)
            if not isinstance(captions, list) or len(captions) != 5 or not all(isinstance(x, str) and x.strip() for x in captions):
                raise ValueError("{} image_id={} must have five non-empty captions".format(method, image_id))
            trace_texts = [x["text"] for x in trace_maps[method][image_id]["selected"]]
            if captions != trace_texts:
                raise ValueError("{} final JSON does not exactly match Trace for image_id={}".format(method, image_id))

    reference = trace_maps[methods[0]]
    for method in methods[1:]:
        for image_id in source_map:
            if trace_maps[method][image_id]["anchor_candidate_id"] != reference[image_id]["anchor_candidate_id"]:
                anchor_mismatch_count += 1
    if anchor_mismatch_count:
        raise ValueError("Shared anchor mismatch across methods: {} image-method comparisons".format(anchor_mismatch_count))

    alias_count = 0
    if v3_alias and "B4" in methods:
        alias_path = Path(caption_dir) / "5k_ECSD_v3.json"
        if alias_path.is_file():
            alias_rows = load_json(alias_path)
            alias_map = {int(x["image_id"]): x for x in alias_rows}
            if set(alias_map) != set(source_map):
                raise ValueError("ECSD v3 alias image IDs do not match source")
            for image_id, row in alias_map.items():
                if row.get("raw_captions") != source_map[image_id].get("raw_captions"):
                    raise ValueError("ECSD v3 alias Raw Captions mismatch for image_id={}".format(image_id))
                expected = [x["text"] for x in trace_maps["B4"][image_id]["selected"]]
                if row.get("ecsd_v3_captions") != expected:
                    raise ValueError("ECSD v3 alias does not match B4 trace for image_id={}".format(image_id))
                alias_count += 1

    return {
        "status": "PASS",
        "num_images": len(source_map),
        "num_candidates": candidate_count,
        "methods": methods,
        "trace_rows_per_method": {m: len(trace_maps[m]) for m in methods},
        "shared_anchor_mismatches": anchor_mismatch_count,
        "duplicate_fallback_slots": sum(
            1 for method in methods for row in trace_maps[method].values()
            for item in row.get("selected", []) if item.get("fallback_type") == "duplicate_slot"
        ),
        "relaxed_fallback_slots": sum(
            1 for method in methods for row in trace_maps[method].values()
            for item in row.get("selected", []) if item.get("fallback_type") == "relaxed_quality_filter"
        ),
        "v3_alias_rows_validated": alias_count,
        "checks": [
            "unique source and pool image IDs", "Raw Captions exact match",
            "unique candidate IDs and canonical texts", "stable candidate ID derivation", "source span offsets/text provenance",
            "five unique slot IDs per image; repeated candidate IDs require explicit duplicate_slot trace",
            "trace candidate IDs/texts exist in candidate pool", "final caption JSON equals trace order",
            "shared B1-B4 anchor invariance",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--caption-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--expected-images", type=int, default=5000)
    parser.add_argument("--methods", nargs="+", default=["B1", "B2", "B3", "B4"])
    parser.add_argument("--report", default="outputs/5k_ecsd_v3/integrity_audit.json")
    args = parser.parse_args()
    report = validate_generation(args.source, args.candidate_pool, args.caption_dir,
                                 args.trace_dir, args.expected_images, args.methods)
    atomic_write_json(args.report, report, overwrite=True)
    print("[PASS] v3 generation integrity:", args.report)


if __name__ == "__main__":
    main()
