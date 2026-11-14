#!/usr/bin/env python3
"""Collect training losses, checkpoint status, CISD, and segmentation mIoU."""
import argparse
import csv
import json
import re
from pathlib import Path

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+):\s*train_loss=(?:tensor\()?\s*(" + FLOAT +
    r")\)?\s*-\s*val_loss=(?:tensor\()?\s*(" + FLOAT + r")\)?"
)


def parse_final_losses(path):
    path = Path(path)
    if not path.is_file():
        return "", "", ""
    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_PATTERN.search(line)
        if match:
            matches.append((
                int(match.group(1)) + 1,
                float(match.group(2)),
                float(match.group(3)),
            ))
    return matches[-1] if matches else ("", "", "")


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def load_cisd(path):
    path = Path(path)
    if not path.is_file():
        return {}
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["arm"]] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/5k_ecsd")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--seg-results-dir", default="segmentation_results")
    parser.add_argument("--weight-tag", default="ecsd5k")
    parser.add_argument("--arms", default="raw rich rich_weaken rich_diversity ecsd")
    parser.add_argument("--benchmarks", default="voc20 context59 coco_stuff ade20k")
    args = parser.parse_args()

    root = Path(args.root)
    weights_dir = Path(args.weights_dir)
    seg_dir = Path(args.seg_results_dir)
    cisd_rows = load_cisd(root / "cisd" / "summary.csv")
    benchmarks = ["context59" if x == "pc59" else x for x in args.benchmarks.split()]

    fields = [
        "arm", "weight_name", "weight_exists", "final_epoch",
        "final_train_loss", "final_val_loss", "mean_cisd",
        "std_cisd_population", "mean_pairwise_similarity",
    ] + ["{}_mIoU".format(x) for x in benchmarks]

    rows = []
    for arm in args.arms.split():
        weight_name = "vitb_mlp_infonce_{}_{}".format(args.weight_tag, arm)
        weight_path = weights_dir / (weight_name + ".pth")
        epoch, train_loss, val_loss = parse_final_losses(
            root / "logs" / "train_{}.log".format(arm)
        )
        metrics = load_json(seg_dir / (weight_name + ".json"))
        cisd = cisd_rows.get(arm, {})
        row = {
            "arm": arm,
            "weight_name": weight_name,
            "weight_exists": weight_path.is_file(),
            "final_epoch": epoch,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "mean_cisd": cisd.get("mean_cisd", ""),
            "std_cisd_population": cisd.get("std_cisd_population", ""),
            "mean_pairwise_similarity": cisd.get("mean_pairwise_similarity", ""),
        }
        for benchmark in benchmarks:
            row[benchmark + "_mIoU"] = metrics.get(benchmark, "")
        rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    output = root / "summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved summary: {}".format(output))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
