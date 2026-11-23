#!/usr/bin/env python3
# CPU-only candidate shortage audit.
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_utils import load_config
from selection_ablation.selection_methods_v3 import valid_candidates
from candidate_quality.semantic_role_classifier import quality_flags
from utils.text_utils import canonical_surface_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--candidate-pool", default="outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl")
    parser.add_argument("--config", default="experiments/5k_ecsd_v3/configs/default.json")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_v3")
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    source_map = {int(x["image_id"]): x for x in source}
    pool = []
    with Path(args.candidate_pool).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    pool.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSONL line {}: {}".format(line_no, exc))
    pool_map = {int(x["image_id"]): x for x in pool}
    if len(source_map) != len(source) or set(source_map) != set(pool_map):
        raise ValueError("Source and candidate-pool image IDs are not a one-to-one match")
    config = load_config(args.config)
    filter_config = config.get("candidate_filter", {})
    rows = []
    for image_id in sorted(source_map):
        raw = source_map[image_id].get("raw_captions", [])
        candidates = pool_map[image_id].get("candidates", [])
        if raw != pool_map[image_id].get("raw_captions"):
            raise ValueError("Raw Caption mismatch for image_id={}".format(image_id))
        valid = valid_candidates(candidates, filter_config)
        valid_ids = {x["candidate_id"] for x in valid}
        surfaces = {canonical_surface_text(x.get("text", "")) for x in candidates}
        raw_surfaces = {canonical_surface_text(x) for x in raw}
        reasons = Counter()
        for item in candidates:
            if item["candidate_id"] not in valid_ids:
                flags = quality_flags(item.get("text", ""), item.get("candidate_type", "candidate"), filter_config)
                reasons.update(flags or ["other_filter"])
        relaxed = len(candidates) - len(valid_ids)
        rows.append({
            "image_id": image_id,
            "raw_caption_count": len(raw),
            "raw_unique_count": len(raw_surfaces),
            "candidate_pool_count": len(candidates),
            "candidate_unique_count": len(surfaces),
            "valid_candidate_count": len(valid_ids),
            "relaxed_candidate_count": relaxed,
            "invalid_candidate_count": relaxed,
            "invalid_reasons": json.dumps(dict(reasons), ensure_ascii=False, sort_keys=True),
            "fallback_required": int(len(surfaces) < 5),
        })
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "selection_shortage_audit_v3.json"
    csv_path = out / "selection_shortage_summary_v3.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print("[DONE] images={} valid<5={} distinct<5={}".format(
        len(rows), sum(x["valid_candidate_count"] < 5 for x in rows),
        sum(x["fallback_required"] for x in rows)))
    print("[DONE] JSON:", json_path)
    print("[DONE] CSV:", csv_path)


if __name__ == "__main__":
    main()
