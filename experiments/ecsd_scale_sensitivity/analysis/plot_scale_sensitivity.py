#!/usr/bin/env python3
"""Plot Raw/ECSD scaling curves and paired deltas from aggregate_results.csv."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-dir", default="outputs/ecsd_scale_sensitivity/summary")
    p.add_argument("--benchmarks", default="voc20 context59 coco_stuff ade20k")
    a = p.parse_args()
    summary_dir = Path(a.summary_dir)
    rows = read_rows(summary_dir / "aggregate_results.csv")
    sizes = {"5k": 5000, "10k": 10000, "20k": 20000, "40k": 40000, "full": 82783}

    for benchmark in a.benchmarks.split():
        plt.figure(figsize=(8, 5))
        for arm in ("raw", "ecsd"):
            points = [x for x in rows if x["arm"] == arm]
            points.sort(key=lambda x: sizes[x["scale"]])
            points = [x for x in points if number(x[benchmark + "_mean_mIoU"]) is not None]
            xs = [sizes[x["scale"]] for x in points]
            ys = [number(x[benchmark + "_mean_mIoU"]) for x in points]
            es = [number(x[benchmark + "_std_mIoU"]) or 0.0 for x in points]
            if points:
                plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=arm.upper())
        plt.xscale("log")
        plt.xticks([5000, 10000, 20000, 40000, 82783], ["5K", "10K", "20K", "40K", "Full"])
        plt.xlabel("Number of Train2014 images")
        plt.ylabel("mIoU (%)")
        plt.title("{}: data scale vs segmentation performance".format(benchmark))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_dir / ("scale_sensitivity_" + benchmark + ".png"), dpi=180)
        plt.close()

        points = [x for x in rows if x["arm"] == "ecsd_minus_raw"]
        points.sort(key=lambda x: sizes[x["scale"]])
        points = [x for x in points if number(x[benchmark + "_mean_mIoU"]) is not None]
        xs = [sizes[x["scale"]] for x in points]
        ys = [number(x[benchmark + "_mean_mIoU"]) for x in points]
        es = [number(x[benchmark + "_std_mIoU"]) or 0.0 for x in points]
        plt.figure(figsize=(8, 5))
        if points:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3)
        plt.axhline(0.0, linestyle="--")
        plt.xscale("log")
        plt.xticks([5000, 10000, 20000, 40000, 82783], ["5K", "10K", "20K", "40K", "Full"])
        plt.xlabel("Number of Train2014 images")
        plt.ylabel("ECSD - Raw mIoU (percentage points)")
        plt.title("{}: ECSD improvement vs data scale".format(benchmark))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(summary_dir / ("scale_delta_" + benchmark + ".png"), dpi=180)
        plt.close()
    print("[DONE] plots written to {}".format(summary_dir))


if __name__ == "__main__":
    main()
