#!/usr/bin/env python3
"""Summarize and plot B1-B4 selection ablations."""
import argparse
import csv
import json
from pathlib import Path

def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/5k_ecsd_next/ablation_metrics/summary.csv")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_next/ablation_analysis")
    args = parser.parse_args()
    rows = read_csv(args.summary)
    rows = [r for r in rows if r.get("arm", "").startswith("B")]
    if not rows:
        raise ValueError("No B1-B4 ablation rows found in summary.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metric_specs = {
        "mean_cisd": ("CISD", "higher means greater text-set diversity"),
        "mean_semantic_coverage": ("Semantic Coverage", "higher means more Raw-caption facts are covered"),
        "mean_category_coverage": ("Category Coverage", "higher means more lexical roles are represented"),
        "mean_redundancy": ("Redundancy", "lower means less fact overlap"),
        "mean_candidate_validity_rate": ("Heuristic Validity Rate", "higher means fewer surface-form warnings"),
    }
    ranking = {}
    for column, (label, note) in metric_specs.items():
        valid = [r for r in rows if r.get(column, "") not in ("", None)]
        if not valid:
            continue
        reverse = column != "mean_redundancy"
        best = sorted(valid, key=lambda r: float(r[column]), reverse=reverse)[0]
        ranking[column] = {
            "label": label, "best_arm": best["arm"], "best_value": float(best[column]),
            "direction": "higher" if reverse else "lower", "interpretation": note,
        }

    (out / "ranking_summary.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    lines = [
        "# ECSD Selection Ablation Summary", "",
        "This report compares the selection objectives on the same improved candidate pool.", "",
        "| Metric | Best arm | Value | Preferred direction |",
        "|---|---|---:|---|",
    ]
    for column, item in ranking.items():
        lines.append("| {} | {} | {:.5f} | {} |".format(
            item["label"], item["best_arm"], item["best_value"], item["direction"]))
    lines += [
        "", "## Interpretation notes", "",
        "- CISD and semantic coverage are reported independently; do not collapse them into one score.",
        "- A high CISD does not imply high semantic coverage.",
        "- Lexical category coverage and heuristic validity are diagnostics, not human-verified correctness.",
        "- Use the manual annotation CSV to validate candidate quality before interpreting automatic ratios.",
        "- The best arm depends on the downstream objective; inspect all metrics together.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for column, (label, _) in metric_specs.items():
            if not any(r.get(column, "") not in ("", None) for r in rows):
                continue
            labels = [r["arm"] for r in rows]
            values = [float(r[column]) for r in rows]
            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            ax.bar(labels, values)
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fig.savefig(out / (column + ".png"), dpi=160)
            plt.close(fig)
        print("[DONE] Plots written to", out)
    except ImportError:
        print("[WARN] matplotlib is unavailable; CSV/JSON report still written.")
    print("[DONE] Ablation report:", out / "REPORT.md")

if __name__ == "__main__":
    main()
