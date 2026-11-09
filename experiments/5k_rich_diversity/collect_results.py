#!/usr/bin/env python3
"""Collect training losses, checkpoint status, and segmentation mIoU into CSV."""
import argparse
import csv
import json
import re
from pathlib import Path

ARMS = ("raw", "rich", "rich_weaken", "rich_diversity")
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
            matches.append((int(match.group(1)) + 1, float(match.group(2)), float(match.group(3))))
    if not matches:
        return "", "", ""
    return matches[-1]


def load_metrics(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/5k_rich_diversity")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--seg-results-dir", default="segmentation_results")
    parser.add_argument("--weight-tag", default="subset5k")
    parser.add_argument("--arms", default="raw rich rich_weaken rich_diversity")
    parser.add_argument("--legacy-root", default="outputs/raw_rich_weaken_5k",
                        help="Fallback location for the previous experiment's three training logs.")
    parser.add_argument("--benchmarks", default="voc20 context59 coco_stuff ade20k")
    args = parser.parse_args()

    root = Path(args.root)
    weights_dir = Path(args.weights_dir)
    seg_dir = Path(args.seg_results_dir)
    legacy_root = Path(args.legacy_root)
    arms = args.arms.split()
    benchmarks = ["context59" if x == "pc59" else x for x in args.benchmarks.split()]
    fields = ["arm", "weight_name", "weight_exists", "final_epoch", "final_train_loss", "final_val_loss"]
    fields += ["{}_mIoU".format(x) for x in benchmarks]

    rows = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError("Unsupported arm: {}".format(arm))
        weight_name = "vitb_mlp_infonce_{}_{}".format(args.weight_tag, arm)
        weight_path = weights_dir / (weight_name + ".pth")
        log_path = root / "logs" / "train_{}.log".format(arm)
        if not log_path.is_file() and arm in ("raw", "rich", "rich_weaken"):
            log_path = legacy_root / "logs" / "train_{}.log".format(arm)
        epoch, train_loss, val_loss = parse_final_losses(log_path)
        metrics = load_metrics(seg_dir / (weight_name + ".json"))
        row = {
            "arm": arm,
            "weight_name": weight_name,
            "weight_exists": weight_path.is_file(),
            "final_epoch": epoch,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
        }
        for benchmark in benchmarks:
            row["{}_mIoU".format(benchmark)] = metrics.get(benchmark, "")
        rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved summary: {}".format(output_path))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
