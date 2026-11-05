#!/usr/bin/env python3
"""Collect final training losses and Talk2DINO segmentation mIoU into one CSV."""
import argparse
import csv
import json
import re
from pathlib import Path


ARMS = ("raw", "rich", "rich_weaken")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+):\s*train_loss=(?:tensor\()?\s*("
    + FLOAT
    + r")\)?\s*-\s*val_loss=(?:tensor\()?\s*("
    + FLOAT
    + r")\)?"
)


def parse_final_losses(path):
    path = Path(path)
    if not path.is_file():
        return "", "", ""

    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_PATTERN.search(line)
        if match:
            matches.append(
                (int(match.group(1)) + 1, float(match.group(2)), float(match.group(3)))
            )

    if not matches:
        return "", "", ""
    epoch, train_loss, val_loss = matches[-1]
    return epoch, train_loss, val_loss


def load_segmentation_metrics(path):
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
    parser.add_argument("--root", default="outputs/raw_rich_weaken_5k")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--seg-results-dir", default="segmentation_results")
    parser.add_argument(
        "--benchmarks",
        default="voc20 context59 coco_stuff ade20k",
        help="Space-separated benchmark names, e.g. 'voc20 coco_stuff'.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    weights_dir = Path(args.weights_dir)
    seg_dir = Path(args.seg_results_dir)
    benchmarks = ["context59" if item == "pc59" else item for item in args.benchmarks.split()]

    fieldnames = [
        "arm",
        "weight_name",
        "weight_exists",
        "final_epoch",
        "final_train_loss",
        "final_val_loss",
    ] + [f"{benchmark}_mIoU" for benchmark in benchmarks]

    rows = []
    for arm in ARMS:
        weight_name = f"vitb_mlp_infonce_subset5k_{arm}"
        weight_path = weights_dir / f"{weight_name}.pth"
        log_path = root / "logs" / f"train_{arm}.log"
        epoch, train_loss, val_loss = parse_final_losses(log_path)
        seg_metrics = load_segmentation_metrics(seg_dir / f"{weight_name}.json")

        row = {
            "arm": arm,
            "weight_name": weight_name,
            "weight_exists": weight_path.is_file(),
            "final_epoch": epoch,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
        }
        for benchmark in benchmarks:
            row[f"{benchmark}_mIoU"] = seg_metrics.get(benchmark, "")
        rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary: {output_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
