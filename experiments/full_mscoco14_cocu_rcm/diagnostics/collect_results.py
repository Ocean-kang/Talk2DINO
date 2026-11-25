#!/usr/bin/env python3
"""Collect full-scale training and OVSS metrics without touching original collectors."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/full_mscoco14_cocu_rcm")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--seg-results-dir", default="segmentation_results")
    p.add_argument("--arms", default="raw cocu cocu_same cocu_rcm cocu_rcm_shuffled")
    p.add_argument("--seeds", default="123")
    p.add_argument("--benchmarks", default="voc20 context59 coco_stuff ade20k")
    a = p.parse_args()
    root, weights, seg = Path(a.root), Path(a.weights_dir), Path(a.seg_results_dir)
    arms, seeds, benchmarks = a.arms.split(), a.seeds.split(), a.benchmarks.split()
    fields = [
        "arm", "seed", "weight_name", "weight_exists", "epochs_completed",
        "final_train_global_loss", "final_train_local_raw",
        "final_train_local_scaled", "final_weighted_local_loss",
        "final_val_global_loss", "final_global_grad_norm",
        "final_local_grad_norm", "final_local_global_grad_ratio",
    ] + [f"{x}_mIoU" for x in benchmarks]
    rows = []
    for seed in seeds:
        for arm in arms:
            name = f"vitb_mlp_infonce_full_mscoco14_cocu_rcm_{arm}_seed{seed}"
            csv_path = root / "logs" / f"train_{arm}_seed{seed}.csv"
            train_rows = []
            if csv_path.is_file():
                with csv_path.open("r", encoding="utf-8") as f:
                    train_rows = list(csv.DictReader(f))
            last = train_rows[-1] if train_rows else {}
            metric = load_json(seg / f"{name}.json")
            row = {
                "arm": arm, "seed": seed, "weight_name": name,
                "weight_exists": (weights / f"{name}.pth").is_file(),
                "epochs_completed": len(train_rows),
                "final_train_global_loss": last.get("train_global_loss", ""),
                "final_train_local_raw": last.get("train_local_raw", ""),
                "final_train_local_scaled": last.get("train_local_scaled", ""),
                "final_weighted_local_loss": last.get("weighted_local_loss", ""),
                "final_val_global_loss": last.get("val_global_loss", ""),
                "final_global_grad_norm": last.get("global_grad_norm", ""),
                "final_local_grad_norm": last.get("local_grad_norm", ""),
                "final_local_global_grad_ratio": last.get("local_global_grad_ratio", ""),
            }
            for benchmark in benchmarks:
                value = metric.get(benchmark, "")
                if isinstance(value, dict):
                    value = value.get("mIoU", value.get("miou", ""))
                row[f"{benchmark}_mIoU"] = value
            rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUMMARY] {output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
