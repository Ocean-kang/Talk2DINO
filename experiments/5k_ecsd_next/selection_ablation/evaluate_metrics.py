#!/usr/bin/env python3
"""Compute CISD, complementary semantic coverage, category coverage and redundancy."""
import argparse
import csv
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEXT_ROOT = HERE.parent
REPO_ROOT = NEXT_ROOT.parents[1]
sys.path.insert(0, str(NEXT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "5k_ecsd"))

from config_utils import load_config
from candidate_quality.semantic_role_classifier import (
    classify_semantic_roles, content_tokens, quality_flags,
)
from selection_ablation.selection_methods import build_fact_weights, weighted_jaccard
from clip_utils import load_clip_model, encode_texts

BASE_FIELDS = {
    "Raw": ("base", "raw_captions"),
    "Rich": ("base", "rich_captions"),
    "Rich-Diversity": ("base", "rich_diversity_captions"),
    "ECSD-v2": ("base", "ecsd_v2_captions"),
}
ABLATION_FIELDS = {
    "B1-Coverage": ("5k_ECSD_B1_Coverage.json", "ecsd_b1_captions"),
    "B2-Diversity": ("5k_ECSD_B2_Diversity.json", "ecsd_b2_captions"),
    "B3-Coverage-Diversity": ("5k_ECSD_B3_Coverage_Diversity.json", "ecsd_b3_captions"),
    "B4-Coverage-Diversity-Role": ("5k_ECSD_B4_Coverage_Diversity_Role.json", "ecsd_b4_captions"),
}
PAIRS = list(combinations(range(5), 2))

def read_rows(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("{} must contain a JSON list.".format(path))
    return {int(x["image_id"]): x for x in rows}

def mean_std(values):
    return (statistics.mean(values), statistics.pstdev(values)) if values else (float("nan"), float("nan"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-captions", default="data/coco2014_subset_50k/5k_ECSD_v2.json")
    parser.add_argument("--config", default="experiments/5k_ecsd_next/configs/default.json")
    parser.add_argument("--ablation-dir", default="data/coco2014_subset_50k")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_next/ablation_metrics")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-cache", default="~/.cache/clip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expected-images", type=int, default=5000)
    parser.add_argument("--include-ablation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    filter_config = config.get("candidate_filter", {})
    base = read_rows(args.base_captions)
    if args.expected_images and len(base) != args.expected_images:
        raise ValueError("Expected {} base images, got {}.".format(args.expected_images, len(base)))
    ids = sorted(base)
    arms = {}
    for name, (_, field) in BASE_FIELDS.items():
        for image_id in ids:
            captions = base[image_id].get(field)
            if not isinstance(captions, list) or len(captions) != 5:
                raise ValueError("image_id={} missing five {} captions.".format(image_id, name))
        arms[name] = {image_id: base[image_id][field] for image_id in ids}

    if args.include_ablation:
        for name, (filename, field) in ABLATION_FIELDS.items():
            path = Path(args.ablation_dir) / filename
            if not path.is_file():
                print("[SKIP] Missing ablation caption file:", path)
                continue
            rows = read_rows(path)
            if set(rows) != set(ids):
                raise ValueError("{} image IDs differ from base JSON.".format(filename))
            arms[name] = {}
            for image_id in ids:
                captions = rows[image_id].get(field)
                if not isinstance(captions, list) or len(captions) != 5:
                    raise ValueError("image_id={} missing five captions in {}.".format(image_id, field))
                arms[name][image_id] = captions

    unique_texts = list(dict.fromkeys(
        text for per_image in arms.values() for image_id in ids for text in per_image[image_id]
    ))
    clip_module, model, device = load_clip_model(args.clip_model, args.device, args.clip_cache)
    embeddings = encode_texts(unique_texts, clip_module, model, device, batch_size=args.batch_size)
    emb_by_text = {text: embeddings[i] for i, text in enumerate(unique_texts)}

    per_image_rows = []
    values = {arm: {k: [] for k in (
        "cisd", "mean_pairwise_similarity", "semantic_coverage",
        "category_coverage", "redundancy", "candidate_validity_rate",
        "unique_semantic_facts",
    )} for arm in arms}

    for image_id in ids:
        raw = base[image_id]["raw_captions"]
        fact_freq, fact_weights = build_fact_weights(raw)
        source_facts = set(fact_freq)
        total_weight = sum(fact_weights.get(x, 1.0) for x in source_facts) or 1.0
        row = {"image_id": image_id}
        for arm, per_image in arms.items():
            captions = per_image[image_id]
            vectors = [emb_by_text[x] for x in captions]
            similarities = [
                float((vectors[i] * vectors[j]).sum().item())
                if hasattr(vectors[i], "sum") else sum(a*b for a,b in zip(vectors[i], vectors[j]))
                for i, j in PAIRS
            ]
            mean_similarity = sum(similarities) / len(similarities)
            cisd = 1.0 - mean_similarity
            selected_facts = [content_tokens(x) for x in captions]
            covered = set().union(*selected_facts)
            coverage = sum(fact_weights.get(x, 1.0) for x in (covered & source_facts)) / total_weight
            categories = set()
            validity = []
            for text in captions:
                categories.update(x for x in classify_semantic_roles(text)["categories"] if x != "Unknown")
                validity.append(not quality_flags(text, "original" if arm == "Raw" else "candidate", filter_config))
            category_coverage = len(categories) / 5.0
            redundancies = [
                weighted_jaccard(selected_facts[i], selected_facts[j], fact_weights)
                for i, j in PAIRS
            ]
            redundancy = sum(redundancies) / len(redundancies)
            valid_rate = sum(validity) / len(validity)
            metrics = {
                "cisd": cisd,
                "mean_pairwise_similarity": mean_similarity,
                "semantic_coverage": coverage,
                "category_coverage": category_coverage,
                "redundancy": redundancy,
                "candidate_validity_rate": valid_rate,
                "unique_semantic_facts": len(covered),
            }
            for key, value in metrics.items():
                row[arm + "_" + key] = value
                values[arm][key].append(value)
        per_image_rows.append(row)
        if len(per_image_rows) % 500 == 0:
            print("[metrics] {}/{}".format(len(per_image_rows), len(ids)), flush=True)

    summary_rows = []
    for arm, metrics in values.items():
        out = {"arm": arm, "num_images": len(ids)}
        for metric, vals in metrics.items():
            mean, std = mean_std(vals)
            out["mean_" + metric] = mean
            out["std_" + metric] = std
            out["median_" + metric] = statistics.median(vals)
        summary_rows.append(out)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = {
        "cisd_definition": "1 - mean cosine similarity over the ten unordered caption pairs",
        "coverage_definition": "weighted union of selected lexical facts divided by weighted union of Raw Caption facts",
        "category_coverage_definition": "number of distinct known lexical roles covered divided by five",
        "redundancy_definition": "mean weighted Jaccard overlap over the ten unordered caption pairs",
        "clip_model": args.clip_model,
        "device": device,
        "arms": list(arms),
        "num_images": len(ids),
        "candidate_filter": filter_config,
        "captions_per_image": 5,
        "warning": "Lexical coverage and validity are proxies, not verified semantic correctness.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[DONE] Metrics:", out_dir)

if __name__ == "__main__":
    main()
