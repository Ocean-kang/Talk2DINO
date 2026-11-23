#!/usr/bin/env python3
"""Recompute final B1-B4 and baseline metrics from caption JSON + Trace using cached CLIP."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import sys

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
from config_utils import load_config
from selection_ablation.metrics_core_v3 import compute_caption_set_metrics
from selection_ablation.validate_generation_v3 import validate_generation, METHOD_FILES, load_json, load_jsonl
from utils.clip_embedding_cache import encode_texts_cached
from utils.safe_io import atomic_write_csv, atomic_write_json

BASE_FIELDS = {
    "Raw": "raw_captions",
    "Rich": "rich_captions",
    "Rich-Diversity": "rich_diversity_captions",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--candidate-pool", default="outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl")
    parser.add_argument("--caption-dir", default="data/coco2014_subset_50k")
    parser.add_argument("--trace-dir", default="outputs/5k_ecsd_v3/selection_traces")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_v3/ablation_metrics")
    parser.add_argument("--config", default="experiments/5k_ecsd_v3/configs/default.json")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-cache", default="~/.cache/clip")
    parser.add_argument("--embedding-cache-dir", default="outputs/5k_ecsd_v3/clip_cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expected-images", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source_rows = load_json(args.source)
    source_map = {int(x["image_id"]): x for x in source_rows}
    if len(source_map) != len(source_rows):
        raise ValueError("Source contains duplicate image IDs")
    report = validate_generation(
        args.source, args.candidate_pool, args.caption_dir, args.trace_dir,
        expected_images=args.expected_images, methods=["B1", "B2", "B3", "B4"],
    )

    arms = {name: {iid: source_map[iid][field] for iid in source_map} for name, field in BASE_FIELDS.items()}
    trace_maps = {}
    type_maps = {}
    for method in ("B1", "B2", "B3", "B4"):
        filename, field = METHOD_FILES[method]
        rows = load_json(Path(args.caption_dir) / filename)
        row_map = {int(x["image_id"]): x for x in rows}
        arms[method] = {iid: row_map[iid][field] for iid in source_map}
        traces = load_jsonl(Path(args.trace_dir) / (method + "_selection_trace.jsonl"))
        trace_maps[method] = {int(x["image_id"]): x for x in traces}
        type_maps[method] = {
            int(x["image_id"]): [z.get("candidate_type", "fallback") for z in x["selected"]]
            for x in traces
        }

    unique_texts = sorted({
        text for per_image in arms.values() for captions in per_image.values() for text in captions
    })
    embeddings, cache_info = encode_texts_cached(
        unique_texts, args.clip_model, Path(args.clip_cache).expanduser(), args.embedding_cache_dir,
        batch_size=args.batch_size, device=args.device,
    )
    print("[metrics v3] unique_texts={} cache_hit={}".format(cache_info["num_texts"], cache_info["cache_hit"]))

    per_image = []
    values = defaultdict(lambda: defaultdict(list))
    for idx, image_id in enumerate(sorted(source_map)):
        raw = source_map[image_id]["raw_captions"]
        row = {"image_id": image_id}
        for arm, image_captions in arms.items():
            types = type_maps.get(arm, {}).get(image_id, ["original"] * 5)
            metrics = compute_caption_set_metrics(
                image_captions[image_id], raw, embeddings, types,
                filter_config=config.get("candidate_filter", {}),
            )
            for key, value in metrics.items():
                row[arm + "_" + key] = value
                values[arm][key].append(value)
        per_image.append(row)
        if (idx + 1) % 500 == 0:
            print("[metrics v3] {}/{}".format(idx + 1, len(source_map)), flush=True)

    summary_rows = []
    summary_json = {}
    for arm, metric_values in values.items():
        record = {"arm": arm, "num_images": len(source_map)}
        summary_json[arm] = {"num_images": len(source_map)}
        for metric, vals in metric_values.items():
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals)
            median = statistics.median(vals)
            record["mean_" + metric] = mean
            record["std_" + metric] = std
            record["median_" + metric] = median
            summary_json[arm][metric] = {"mean": mean, "std_population": std, "median": median}
            if metric.startswith("candidate_type_") and metric.endswith("_count"):
                total = int(sum(vals))
                record["total_" + metric] = total
                record["share_" + metric.replace("_count", "")] = total / float(5 * len(source_map))
                summary_json[arm].setdefault("candidate_type_totals", {})[metric] = total
        summary_rows.append(record)

    # Pairwise selection-set consistency is measured by stable candidate IDs, not text matching.
    pairwise_rows = []
    consistency = {}
    methods = ["B1", "B2", "B3", "B4"]
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            exact = 0
            shared_total = 0
            for image_id in source_map:
                left_ids = {x["candidate_id"] for x in trace_maps[left][image_id]["selected"]}
                right_ids = {x["candidate_id"] for x in trace_maps[right][image_id]["selected"]}
                shared_total += len(left_ids & right_ids)
                exact += int(left_ids == right_ids)
            n = len(source_map)
            record = {
                "method_a": left, "method_b": right,
                "identical_selection_images": exact,
                "identical_selection_rate": exact / n,
                "mean_shared_candidate_count": shared_total / n,
                "mean_shared_candidate_ratio": shared_total / (5.0 * n),
            }
            pairwise_rows.append(record)
            consistency[left + "_vs_" + right] = record

    anchor_reference = trace_maps[methods[0]]
    anchor_mismatches = {
        method: sum(
            trace_maps[method][iid]["anchor_candidate_id"] != anchor_reference[iid]["anchor_candidate_id"]
            for iid in source_map
        ) for method in methods[1:]
    }
    if any(anchor_mismatches.values()):
        raise ValueError("Anchor invariance failed during metric evaluation: {}".format(anchor_mismatches))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(out_dir / "per_image_metrics_v3.csv", list(per_image[0].keys()), per_image, overwrite=args.overwrite)
    atomic_write_csv(out_dir / "summary_v3.csv", list(summary_rows[0].keys()), summary_rows, overwrite=args.overwrite)
    atomic_write_csv(out_dir / "pairwise_selection_consistency_v3.csv", list(pairwise_rows[0].keys()), pairwise_rows, overwrite=args.overwrite)
    atomic_write_json(out_dir / "summary_v3.json", summary_json, overwrite=args.overwrite)
    atomic_write_json(out_dir / "selection_consistency_v3.json", {
        "pairwise": consistency,
        "anchor_mismatches_vs_B1": anchor_mismatches,
        "anchor_invariance": "PASS",
    }, overwrite=args.overwrite)
    atomic_write_json(out_dir / "data_integrity_audit_v3.json", report, overwrite=args.overwrite)
    atomic_write_json(out_dir / "metrics_metadata_v3.json", {
        "version": "ECSD-v3",
        "metric_source": "final caption JSON and exact candidate-ID traces",
        "cisd": "1 - mean cosine similarity over the ten unordered pairs of L2-normalized CLIP ViT-B/16 text embeddings",
        "weighted_fact_coverage": "inverse-frequency weighted union of selected content tokens intersected with Raw Caption content tokens, divided by total Raw Caption fact weight",
        "role_coverage": "unique rule-classified roles among Entity/Attribute/Action/Relation/Context divided by five",
        "redundancy_weighted_jaccard": "mean weighted Jaccard over ten unordered pairs of selected caption fact sets",
        "raw_exact_match_rate": "fraction of five selected captions whose lowercase alphanumeric surface token sequence equals one of the five Raw Captions",
        "candidate_type_distribution": "counts from selected Trace items, by Original/Clause/Noun Phrase",
        "candidate_validity_proxy": "fraction without configured lexical surface-form flags; not human ground truth",
        "clip_model": args.clip_model,
        "clip_embedding_cache_path": cache_info.get("cache_path"),
        "clip_weight_signature": cache_info.get("weight_signature"),
        "num_images": len(source_map),
        "arms": list(arms),
        "warning": "All lexical metrics are proxies; no image-grounded or human correctness is inferred.",
    }, overwrite=args.overwrite)
    print("[DONE] v3 metrics written to", out_dir)


if __name__ == "__main__":
    main()
