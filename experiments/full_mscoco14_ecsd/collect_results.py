#!/usr/bin/env python3
"""Collect per-seed Raw/ECSD segmentation results and paired deltas."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def read_csv_map(path, key):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/full_mscoco14_ecsd")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--seg-results-dir", default="segmentation_results")
    p.add_argument("--seeds", default="123")
    p.add_argument("--arms", default="raw ecsd")
    p.add_argument("--benchmarks", default="voc20 context59 coco_stuff ade20k")
    args = p.parse_args()

    root = Path(args.root)
    weights_dir = Path(args.weights_dir)
    seg_dir = Path(args.seg_results_dir)
    seeds = [x for x in args.seeds.split()]
    arms = [x for x in args.arms.split()]
    benchmarks = ["context59" if x == "pc59" else x for x in args.benchmarks.split()]

    cisd = {}
    for split in ("train", "val"):
        cisd[split] = read_csv_map(root / "cisd" / split / "summary.csv", "arm")
    coverage_meta = read_json(root / "cisd" / "train" / "metadata.json")

    per_run = []
    metric_lookup = {}
    for seed in seeds:
        for arm in arms:
            weight_name = "vitb_mlp_infonce_full14_s{}_{}".format(seed, arm)
            weight_path = weights_dir / (weight_name + ".pth")
            seg_metrics = read_json(seg_dir / (weight_name + ".json"))
            train_cisd = cisd["train"].get(arm, {})
            val_cisd = cisd["val"].get(arm, {})
            row = {
                "seed": seed,
                "arm": arm,
                "weight_name": weight_name,
                "weight_exists": weight_path.is_file(),
                "train_mean_cisd": train_cisd.get("mean_cisd", ""),
                "train_mean_pairwise_similarity": train_cisd.get("mean_pairwise_similarity", ""),
                "train_mean_unique_content_facts": train_cisd.get("mean_unique_content_facts", ""),
                "val_mean_cisd": val_cisd.get("mean_cisd", ""),
                "ecsd_source_fact_coverage_train": coverage_meta.get("mean_ecsd_source_fact_coverage", "") if arm == "ecsd" else "",
            }
            for benchmark in benchmarks:
                row[benchmark + "_mIoU"] = seg_metrics.get(benchmark, "")
                metric_lookup[(seed, arm, benchmark)] = number(seg_metrics.get(benchmark))
            per_run.append(row)

    per_fields = ["seed", "arm", "weight_name", "weight_exists", "train_mean_cisd",
                  "train_mean_pairwise_similarity", "train_mean_unique_content_facts",
                  "val_mean_cisd", "ecsd_source_fact_coverage_train"] + [x + "_mIoU" for x in benchmarks]
    write_csv(root / "summary" / "per_seed_results.csv", per_run, per_fields)

    paired = []
    for seed in seeds:
        row = {"seed": seed}
        for benchmark in benchmarks:
            raw = metric_lookup.get((seed, "raw", benchmark))
            ecsd = metric_lookup.get((seed, "ecsd", benchmark))
            row[benchmark + "_raw_mIoU"] = "" if raw is None else raw
            row[benchmark + "_ecsd_mIoU"] = "" if ecsd is None else ecsd
            row[benchmark + "_delta_ecsd_minus_raw"] = "" if raw is None or ecsd is None else ecsd - raw
        paired.append(row)
    paired_fields = ["seed"]
    for benchmark in benchmarks:
        paired_fields.extend([benchmark + "_raw_mIoU", benchmark + "_ecsd_mIoU", benchmark + "_delta_ecsd_minus_raw"])
    write_csv(root / "summary" / "paired_seed_delta.csv", paired, paired_fields)

    aggregate = []
    for arm in arms:
        row = {"arm": arm}
        for benchmark in benchmarks:
            vals = [metric_lookup.get((seed, arm, benchmark)) for seed in seeds]
            vals = [x for x in vals if x is not None]
            row[benchmark + "_mean_mIoU"] = statistics.mean(vals) if vals else ""
            row[benchmark + "_std_mIoU"] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else "")
            row[benchmark + "_n"] = len(vals)
        aggregate.append(row)
    delta_row = {"arm": "ecsd_minus_raw"}
    for name in benchmarks:
        vals = [metric_lookup.get((seed, "ecsd", name)) - metric_lookup.get((seed, "raw", name))
                for seed in seeds
                if metric_lookup.get((seed, "ecsd", name)) is not None
                and metric_lookup.get((seed, "raw", name)) is not None]
        delta_row[name + "_mean_mIoU"] = statistics.mean(vals) if vals else ""
        delta_row[name + "_std_mIoU"] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else "")
        delta_row[name + "_n"] = len(vals)
    aggregate.append(delta_row)
    agg_fields = ["arm"]
    for benchmark in benchmarks:
        agg_fields.extend([benchmark + "_mean_mIoU", benchmark + "_std_mIoU", benchmark + "_n"])
    write_csv(root / "summary" / "aggregate_results.csv", aggregate, agg_fields)

    print("[DONE] {}".format(root / "summary" / "per_seed_results.csv"))
    print("[DONE] {}".format(root / "summary" / "paired_seed_delta.csv"))
    print("[DONE] {}".format(root / "summary" / "aggregate_results.csv"))


if __name__ == "__main__":
    main()
