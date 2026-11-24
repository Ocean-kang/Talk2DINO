#!/usr/bin/env python3
"""Collect scale/seed/arm segmentation results and paired ECSD-minus-Raw deltas."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="outputs/ecsd_scale_sensitivity")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--seg-results-dir", default="segmentation_results")
    p.add_argument("--scales", default="5k 10k 20k 40k full")
    p.add_argument("--seeds", default="123 456 789")
    p.add_argument("--benchmarks", default="voc20 context59 coco_stuff ade20k")
    a = p.parse_args()

    out_root = Path(a.output_root)
    weights_dir = Path(a.weights_dir)
    seg_dir = Path(a.seg_results_dir)
    scales, seeds, benchmarks = a.scales.split(), a.seeds.split(), a.benchmarks.split()
    per_seed, lookup = [], {}

    for scale in scales:
        for seed in seeds:
            for arm in ("raw", "ecsd"):
                suffix = "scale{}_s{}_{}".format(scale, seed, arm)
                weight_name = "vitb_mlp_infonce_" + suffix
                metric_path = seg_dir / (weight_name + ".json")
                metrics = read_json(metric_path)
                row = {
                    "scale": scale,
                    "num_images": 82783 if scale == "full" else int(scale[:-1]) * 1000,
                    "seed": seed,
                    "arm": arm,
                    "weight_name": weight_name,
                    "weight_exists": (weights_dir / (weight_name + ".pth")).is_file(),
                    "metrics_file_exists": metric_path.is_file(),
                }
                for benchmark in benchmarks:
                    value = to_float(metrics.get(benchmark))
                    row[benchmark + "_mIoU"] = "" if value is None else value
                    lookup[(scale, seed, arm, benchmark)] = value
                per_seed.append(row)

    per_fields = ["scale", "num_images", "seed", "arm", "weight_name", "weight_exists", "metrics_file_exists"]
    per_fields += [x + "_mIoU" for x in benchmarks]
    write_csv(out_root / "summary" / "per_seed_results.csv", per_seed, per_fields)

    paired = []
    for scale in scales:
        for seed in seeds:
            row = {
                "scale": scale,
                "num_images": 82783 if scale == "full" else int(scale[:-1]) * 1000,
                "seed": seed,
            }
            for benchmark in benchmarks:
                raw = lookup.get((scale, seed, "raw", benchmark))
                ecsd = lookup.get((scale, seed, "ecsd", benchmark))
                row[benchmark + "_raw_mIoU"] = "" if raw is None else raw
                row[benchmark + "_ecsd_mIoU"] = "" if ecsd is None else ecsd
                row[benchmark + "_delta_ecsd_minus_raw"] = "" if raw is None or ecsd is None else ecsd - raw
            paired.append(row)
    paired_fields = ["scale", "num_images", "seed"]
    for benchmark in benchmarks:
        paired_fields += [benchmark + "_raw_mIoU", benchmark + "_ecsd_mIoU", benchmark + "_delta_ecsd_minus_raw"]
    write_csv(out_root / "summary" / "paired_seed_delta.csv", paired, paired_fields)

    aggregate = []
    for scale in scales:
        for arm in ("raw", "ecsd"):
            row = {"scale": scale, "num_images": 82783 if scale == "full" else int(scale[:-1]) * 1000, "arm": arm}
            for benchmark in benchmarks:
                values = [lookup.get((scale, seed, arm, benchmark)) for seed in seeds]
                values = [x for x in values if x is not None]
                row[benchmark + "_mean_mIoU"] = statistics.mean(values) if values else ""
                row[benchmark + "_std_mIoU"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
                row[benchmark + "_n"] = len(values)
            aggregate.append(row)
        row = {"scale": scale, "num_images": 82783 if scale == "full" else int(scale[:-1]) * 1000, "arm": "ecsd_minus_raw"}
        for benchmark in benchmarks:
            values = [
                lookup[(scale, seed, "ecsd", benchmark)] - lookup[(scale, seed, "raw", benchmark)]
                for seed in seeds
                if lookup.get((scale, seed, "ecsd", benchmark)) is not None
                and lookup.get((scale, seed, "raw", benchmark)) is not None
            ]
            row[benchmark + "_mean_mIoU"] = statistics.mean(values) if values else ""
            row[benchmark + "_std_mIoU"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
            row[benchmark + "_n"] = len(values)
        aggregate.append(row)

    agg_fields = ["scale", "num_images", "arm"]
    for benchmark in benchmarks:
        agg_fields += [benchmark + "_mean_mIoU", benchmark + "_std_mIoU", benchmark + "_n"]
    write_csv(out_root / "summary" / "aggregate_results.csv", aggregate, agg_fields)
    print("[DONE] summary tables written to {}".format(out_root / "summary"))


if __name__ == "__main__":
    main()
