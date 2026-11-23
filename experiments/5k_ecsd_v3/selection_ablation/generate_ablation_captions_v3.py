#!/usr/bin/env python3
"""Generate B1-B4 captions with shared anchors, stable IDs, and atomic outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from config_utils import load_config
from selection_ablation.selection_methods_v3 import METHODS, choose_anchor, select_five, valid_candidates
from utils.clip_embedding_cache import encode_texts_cached
from utils.safe_io import atomic_write_json, atomic_write_jsonl
from utils.text_utils import canonical_surface_text
from selection_ablation.validate_generation_v3 import validate_generation

OUTPUTS = {
    "B1": ("5k_ECSD_v3_B1_Coverage.json", "ecsd_v3_b1_captions"),
    "B2": ("5k_ECSD_v3_B2_Diversity.json", "ecsd_v3_b2_captions"),
    "B3": ("5k_ECSD_v3_B3_Coverage_Diversity.json", "ecsd_v3_b3_captions"),
    "B4": ("5k_ECSD_v3_B4_Coverage_Diversity_Role.json", "ecsd_v3_b4_captions"),
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid candidate JSONL at line {}: {}".format(line_no, exc))
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_generation(source_rows, pool_rows, embedding_by_text, config, methods):
    source_map = {int(x["image_id"]): x for x in source_rows}
    pool_map = {int(x["image_id"]): x for x in pool_rows}
    if len(source_map) != len(source_rows) or len(pool_map) != len(pool_rows):
        raise ValueError("Duplicate image IDs in source or candidate pool")
    if set(source_map) != set(pool_map):
        raise ValueError("Candidate pool IDs do not exactly match source IDs")

    filter_config = config.get("candidate_filter", {})
    score_weights = config.get("selection", {})
    result_rows = {m: [] for m in methods}
    trace_rows = {m: [] for m in methods}
    anchors = {}

    for index, image_id in enumerate(sorted(source_map)):
        base = source_map[image_id]
        pool = pool_map[image_id]
        raw = base.get("raw_captions")
        if raw != pool.get("raw_captions"):
            raise ValueError("Raw Captions mismatch for image_id={}".format(image_id))
        if len(raw or []) != 5 or not all(isinstance(x, str) and x.strip() for x in raw):
            raise ValueError("image_id={} must have five non-empty Raw Captions".format(image_id))
        candidates = pool.get("candidates", [])
        valid = valid_candidates(candidates, filter_config)
        anchor = choose_anchor(candidates, raw, filter_config)
        anchors[image_id] = anchor["candidate_id"]

        for method in methods:
            selected_texts, trace = select_five(
                candidates, raw, embedding_by_text, method,
                score_weights=score_weights,
                filter_config=filter_config,
                anchor_candidate_id=anchor["candidate_id"],
            )
            if len(selected_texts) != 5 or len(trace) != 5:
                raise RuntimeError("{} image_id={} did not select exactly five captions".format(method, image_id))
            if trace[0]["candidate_id"] != anchors[image_id]:
                raise AssertionError("Shared anchor changed for {} image_id={}".format(method, image_id))
            filename, field = OUTPUTS[method]
            out_row = dict(base)
            out_row[field] = selected_texts
            result_rows[method].append(out_row)
            trace_rows[method].append({
                "image_id": image_id,
                "method": method,
                "anchor_candidate_id": anchors[image_id],
                "selected": trace,
            })
        if (index + 1) % 500 == 0:
            print("[v3 select] {}/{}".format(index + 1, len(source_map)), flush=True)

    return result_rows, trace_rows, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--candidate-pool", default="outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl")
    parser.add_argument("--config", default="experiments/5k_ecsd_v3/configs/default.json")
    parser.add_argument("--output-dir", default="data/coco2014_subset_50k")
    parser.add_argument("--trace-dir", default="outputs/5k_ecsd_v3/selection_traces")
    parser.add_argument("--cache-dir", default="outputs/5k_ecsd_v3/clip_cache")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-cache", default="~/.cache/clip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expected-images", type=int, default=5000)
    parser.add_argument("--methods", nargs="+", choices=["B1", "B2", "B3", "B4"], default=["B1", "B2", "B3", "B4"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source_rows = load_json(args.input)
    pool_rows = load_jsonl(args.candidate_pool)
    if args.expected_images and len(source_rows) != args.expected_images:
        raise ValueError("Expected {} source images, got {}".format(args.expected_images, len(source_rows)))
    if args.expected_images and len(pool_rows) != args.expected_images:
        raise ValueError("Expected {} candidate pool images, got {}".format(args.expected_images, len(pool_rows)))

    unique_texts = sorted({
        str(candidate["text"])
        for row in pool_rows
        for candidate in row.get("candidates", [])
        if str(candidate.get("text", "")).strip()
    })
    clip_cache = Path(args.clip_cache).expanduser()
    embedding_by_text, cache_info = encode_texts_cached(
        unique_texts,
        args.clip_model,
        clip_cache,
        args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("[CLIP] unique texts={} cache_hit={} cache={}".format(
        cache_info["num_texts"], cache_info["cache_hit"], cache_info["cache_path"]), flush=True)

    result_rows, trace_rows, anchors = prepare_generation(
        source_rows, pool_rows, embedding_by_text, config, args.methods
    )
    output_dir = Path(args.output_dir)
    trace_dir = Path(args.trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        filename, _ = OUTPUTS[method]
        json_status = atomic_write_json(output_dir / filename, result_rows[method], overwrite=args.overwrite)
        trace_status = atomic_write_jsonl(
            trace_dir / (method + "_selection_trace.jsonl"), trace_rows[method], overwrite=args.overwrite
        )
        print("[{}] caption_json={} trace={}".format(method, json_status, trace_status))

    if "B4" in args.methods:
        v3_rows = []
        for row in result_rows["B4"]:
            copy = dict(row)
            copy["ecsd_v3_captions"] = list(copy[OUTPUTS["B4"][1]])
            v3_rows.append(copy)
        status = atomic_write_json(output_dir / "5k_ECSD_v3.json", v3_rows, overwrite=args.overwrite)
        print("[ECSD-v3] alias_json={}".format(status))

    fallback_summary = {
        method: {
            "duplicate_slot_count": sum(
                1 for row in trace_rows[method] for item in row["selected"]
                if item.get("fallback_type") == "duplicate_slot"
            ),
            "relaxed_quality_filter_count": sum(
                1 for row in trace_rows[method] for item in row["selected"]
                if item.get("fallback_type") == "relaxed_quality_filter"
            ),
        }
        for method in args.methods
    }
    metadata = {
        "version": "ECSD-v3.1",
        "fallback_summary": fallback_summary,
        "methods": list(args.methods),
        "method_definitions": METHODS,
        "anchor_rule": "one shared anchor per image, selected once before B1-B4; valid Original with maximum inverse-frequency weighted fact mass; candidate_id tie-break",
        "candidate_pool": str(args.candidate_pool),
        "candidate_pool_sha256": sha256_file(args.candidate_pool),
        "source_caption_sha256": sha256_file(args.input),
        "config_sha256": sha256_file(args.config),
        "clip_model": args.clip_model,
        "clip_cache_dir": str(clip_cache),
        "embedding_cache_path": cache_info.get("cache_path"),
        "evidence_source": "Raw Captions only",
        "captions_per_image": 5,
        "score_weights": config.get("selection", {}),
        "candidate_filter": config.get("candidate_filter", {}),
        "num_images": len(source_rows),
        "unique_candidate_texts": len(unique_texts),
        "shared_anchor_invariance": True,
        "anchor_count": len(anchors),
    }
    atomic_write_json(trace_dir / "generation_metadata.json", metadata, overwrite=args.overwrite)

    if set(args.methods) == {"B1", "B2", "B3", "B4"}:
        report = validate_generation(
            args.input, args.candidate_pool, output_dir, trace_dir,
            expected_images=args.expected_images, methods=["B1", "B2", "B3", "B4"],
        )
        atomic_write_json(Path(args.cache_dir).parent / "integrity_audit.json", report, overwrite=True)
        print("[PASS] generation integrity and shared anchors validated")


if __name__ == "__main__":
    main()
