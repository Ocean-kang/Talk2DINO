"""Summarize one weight tag, with separate columns for each PAMR mode."""

import argparse
import csv
import json
import re
from pathlib import Path

from prepare import ARMS, NEW_ARM

BENCHMARKS = ("voc20", "context59", "coco_stuff", "ade20k")
MODES = ("no_pamr", "pamr")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH = re.compile(rf"Epoch\s+(\d+):\s*train_loss=({NUMBER})\s*-\s*val_loss=({NUMBER})")


def final_losses(path):
    if not path.is_file():
        return "", "", ""
    matches = EPOCH.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        return "", "", ""
    epoch, train, val = matches[-1]
    return int(epoch) + 1, float(train), float(val)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/5k_rich_diversity_strengthen"))
    parser.add_argument("--weight-tag", default="subset5k_strengthen")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=[NEW_ARM])
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.weight_tag):
        parser.error("Invalid weight tag")

    fields = ["arm", "weight_name", "weight_exists", "final_epoch", "final_train_loss", "final_val_loss"]
    fields.extend(f"{task}_{mode}_mIoU" for mode in MODES for task in BENCHMARKS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"summary_{args.weight_tag}.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in args.arms:
            name = f"vitb_mlp_infonce_{args.weight_tag}_{arm}"
            log = args.out_dir / "logs" / args.weight_tag / f"train_{arm}.log"
            epoch, train, val = final_losses(log)
            row = {
                "arm": arm, "weight_name": name,
                "weight_exists": (Path("weights") / f"{name}.pth").is_file(),
                "final_epoch": epoch, "final_train_loss": train, "final_val_loss": val,
            }
            for mode in MODES:
                path = Path("segmentation_results") / f"{name}_{mode}.json"
                if path.is_file():
                    metrics = json.loads(path.read_text(encoding="utf-8"))
                    row.update({f"{task}_{mode}_mIoU": metrics[task] for task in BENCHMARKS if task in metrics})
            writer.writerow(row)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
