#!/usr/bin/env python3
"""Aggregate 12 Talk2DINO runs and four-dataset OVSS metrics."""
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+):\s*train_loss=(?:tensor\()?\s*(" + FLOAT +
    r")\)?\s*-\s*val_loss=(?:tensor\()?\s*(" + FLOAT + r")\)?"
)
ARMS = ["raw", "rich", "rich_diversity", "ecsdv2"]
SEEDS = [123, 456, 789]
BENCHMARKS = ["voc20", "context59", "coco_stuff", "ade20k"]

def parse_epoch_losses(path):
    path = Path(path)
    if not path.is_file():
        return []
    matches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_PATTERN.search(line)
        if match:
            matches.append((int(match.group(1)) + 1, float(match.group(2)), float(match.group(3))))
    return matches

def parse_losses(path):
    matches = parse_epoch_losses(path)
    return matches[-1] if matches else ("", "", "")

def load_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/5k_ecsd_next/multiseed")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--seg-results-dir", default="segmentation_results")
    parser.add_argument("--metrics-summary", default="outputs/5k_ecsd_next/ablation_metrics/summary.csv")
    parser.add_argument("--arms", default="raw rich rich_diversity ecsdv2")
    parser.add_argument("--seeds", default="123 456 789")
    parser.add_argument("--benchmarks", default="voc20 context59 coco_stuff ade20k")
    args = parser.parse_args()

    root = Path(args.root)
    weights_dir = Path(args.weights_dir)
    seg_dir = Path(args.seg_results_dir)
    arms = args.arms.split()
    seeds = [int(x) for x in args.seeds.split()]
    benchmarks = args.benchmarks.split()

    run_rows = []
    learning_curve_rows = []
    convergence_rows = []
    training_values = defaultdict(lambda: defaultdict(list))
    values = defaultdict(list)
    for arm in arms:
        for seed in seeds:
            weight_name = "vitb_mlp_infonce_{}_seed{}".format(arm, seed)
            weight_path = weights_dir / (weight_name + ".pth")
            log_path = root / "logs" / "train_{}_seed{}.log".format(arm, seed)
            epoch_history = parse_epoch_losses(log_path)
            epoch, train_loss, val_loss = epoch_history[-1] if epoch_history else ("", "", "")
            metrics = load_json(seg_dir / (weight_name + ".json"))
            for ep, tr_loss, va_loss in epoch_history:
                learning_curve_rows.append({
                    "arm": arm, "seed": seed, "epoch": ep,
                    "train_loss": tr_loss, "validation_loss": va_loss,
                })
            if epoch_history:
                best_val = min(epoch_history, key=lambda x: x[2])
                convergence_rows.append({
                    "arm": arm, "seed": seed, "epochs_completed": len(epoch_history),
                    "first_train_loss": epoch_history[0][1],
                    "final_train_loss": epoch_history[-1][1],
                    "first_validation_loss": epoch_history[0][2],
                    "final_validation_loss": epoch_history[-1][2],
                    "best_validation_loss": best_val[2],
                    "best_validation_epoch": best_val[0],
                    "train_loss_drop": epoch_history[0][1] - epoch_history[-1][1],
                    "validation_loss_drop": epoch_history[0][2] - epoch_history[-1][2],
                })
                training_values[arm]["final_train_loss"].append(epoch_history[-1][1])
                training_values[arm]["final_validation_loss"].append(epoch_history[-1][2])
                training_values[arm]["best_validation_loss"].append(best_val[2])
                training_values[arm]["epochs_completed"].append(len(epoch_history))
            row = {
                "arm": arm, "seed": seed, "weight_name": weight_name,
                "checkpoint_exists": weight_path.is_file(),
                "final_epoch": epoch, "final_train_loss": train_loss,
                "final_val_loss": val_loss,
            }
            for benchmark in benchmarks:
                value = metrics.get(benchmark, "")
                row[benchmark + "_mIoU"] = value
                if value not in ("", None):
                    values[(arm, benchmark)].append(float(value))
            run_rows.append(row)

    aggregate_rows = []
    for arm in arms:
        for benchmark in benchmarks:
            vals = values[(arm, benchmark)]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals)
                best, worst = max(vals), min(vals)
            else:
                mean = std = best = worst = ""
            baseline_values = values[("raw", benchmark)]
            rich_values = values[("rich", benchmark)]
            richdiv_values = values[("rich_diversity", benchmark)]
            raw_mean = statistics.mean(baseline_values) if baseline_values else None
            rich_mean = statistics.mean(rich_values) if rich_values else None
            richdiv_mean = statistics.mean(richdiv_values) if richdiv_values else None
            aggregate_rows.append({
                "arm": arm, "benchmark": benchmark, "n_seeds": len(vals),
                "mean_mIoU": mean, "std_mIoU_population": std,
                "best_mIoU": best, "worst_mIoU": worst,
                "relative_improvement_over_raw_pct": (
                    (mean - raw_mean) / raw_mean * 100.0
                    if vals and raw_mean not in (None, 0) else ""
                ),
                "relative_improvement_over_rich_pct": (
                    (mean - rich_mean) / rich_mean * 100.0
                    if vals and rich_mean not in (None, 0) else ""
                ),
                "difference_vs_rich_diversity": (
                    mean - richdiv_mean if vals and richdiv_mean is not None else ""
                ),
            })

    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "per_run_summary.csv", run_rows)
    write_csv(root / "aggregate_summary.csv", aggregate_rows)
    write_csv(root / "learning_curves.csv", learning_curve_rows)
    write_csv(root / "convergence_summary.csv", convergence_rows)
    training_stability_rows = []
    for arm in arms:
        row = {"arm": arm}
        for metric, vals in training_values[arm].items():
            if vals:
                row["mean_" + metric] = statistics.mean(vals)
                row["std_" + metric + "_population"] = statistics.pstdev(vals)
            else:
                row["mean_" + metric] = ""
                row["std_" + metric + "_population"] = ""
        training_stability_rows.append(row)
    write_csv(root / "training_stability.csv", training_stability_rows)

    metric_rows = read_csv(args.metrics_summary)
    metric_map = {x.get("arm"): x for x in metric_rows}
    metric_name_map = {
        "raw": "Raw", "rich": "Rich",
        "rich_diversity": "Rich-Diversity", "ecsdv2": "ECSD-v2",
    }
    join_rows = []
    for arm in arms:
        metrics = metric_map.get(metric_name_map.get(arm, ""), {})
        row = {
            "arm": arm,
            "mean_cisd": metrics.get("mean_cisd", ""),
            "mean_semantic_coverage": metrics.get("mean_semantic_coverage", ""),
            "mean_category_coverage": metrics.get("mean_category_coverage", ""),
            "mean_redundancy": metrics.get("mean_redundancy", ""),
        }
        for benchmark in benchmarks:
            match = next((x for x in aggregate_rows if x["arm"] == arm and x["benchmark"] == benchmark), None)
            row[benchmark + "_mean_mIoU"] = match["mean_mIoU"] if match else ""
            row[benchmark + "_std_mIoU"] = match["std_mIoU_population"] if match else ""
        join_rows.append(row)
    write_csv(root / "cisd_coverage_ovss_join.csv", join_rows)

    report = {
        "arms": arms, "seeds": seeds, "benchmarks": benchmarks,
        "statistics": "mean and population standard deviation across the available seeds",
        "warning": "Three seeds support a stability description, not a statistical-significance claim.",
        "num_expected_runs": len(arms) * len(seeds),
        "num_checkpoints_found": sum(bool(x["checkpoint_exists"]) for x in run_rows),
        "num_runs_with_training_logs": len(convergence_rows),
        "num_learning_curve_epochs": len(learning_curve_rows),
        "num_runs_with_all_benchmarks": sum(
            all(x.get(b + "_mIoU", "") not in ("", None) for b in benchmarks)
            for x in run_rows
        ),
    }
    (root / "aggregation_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "CORRELATION_NOTE.md").write_text(
        "# CISD / Coverage / OVSS relationship\n\n"
        "`cisd_coverage_ovss_join.csv` joins caption-set mean CISD and semantic coverage "
        "with the mean OVSS mIoU across the three seeds. Treat this as descriptive evidence only: "
        "there are four caption conditions, and the metrics are not independent observations. "
        "Do not claim statistical significance or causal correlation from this table.\n",
        encoding="utf-8",
    )
    print("[DONE] Per-run:", root / "per_run_summary.csv")
    print("[DONE] Aggregate:", root / "aggregate_summary.csv")
    print("[DONE] Learning curves:", root / "learning_curves.csv")
    print("[DONE] Convergence:", root / "convergence_summary.csv")
    print("[DONE] Training stability:", root / "training_stability.csv")
    print("[DONE] Metric join:", root / "cisd_coverage_ovss_join.csv")

if __name__ == "__main__":
    main()
