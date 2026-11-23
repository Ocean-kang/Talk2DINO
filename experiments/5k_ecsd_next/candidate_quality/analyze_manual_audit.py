#!/usr/bin/env python3
"""Summarize human labels filled into manual_annotation.csv."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

def norm(value):
    return str(value or "").strip().lower()

def summarize(rows):
    fields = [
        "validity_label", "groundedness_label", "completeness_label",
        "category_correct_label", "redundancy_label",
    ]
    counts = {field: Counter(norm(r.get(field)) for r in rows if norm(r.get(field))) for field in fields}
    def ratio(num, den):
        return num / den if den else None
    validity = counts["validity_label"]
    grounded = counts["groundedness_label"]
    complete = counts["completeness_label"]
    category = counts["category_correct_label"]
    redundancy = counts["redundancy_label"]
    return {
        "num_rows": len(rows),
        "selected_rows": sum(norm(r.get("selection_status")) == "selected" for r in rows),
        "validity": {
            "counts": dict(validity),
            "strict_valid_rate": ratio(validity["valid"], sum(validity.values())),
            "valid_or_partial_rate": ratio(validity["valid"] + validity["partially valid"], sum(validity.values())),
        },
        "groundedness": {
            "counts": dict(grounded),
            "unsupported_rate": ratio(grounded["unsupported"], sum(grounded.values())),
        },
        "completeness": {
            "counts": dict(complete),
            "complete_rate": ratio(complete["complete"], sum(complete.values())),
        },
        "category_classification": {
            "counts": dict(category),
            "accuracy": ratio(category["correct"], category["correct"] + category["incorrect"]),
        },
        "redundancy": {
            "counts": dict(redundancy),
            "complementary_rate": ratio(redundancy["complementary"], sum(redundancy.values())),
        },
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="outputs/5k_ecsd_next/manual_audit/manual_annotation.csv")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_next/manual_audit")
    args = parser.parse_args()
    with Path(args.annotations).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No annotation rows found.")
    selected = [r for r in rows if norm(r.get("selection_status")) == "selected"]
    summary = {
        "all_candidates": summarize(rows),
        "selected_candidates": summarize(selected),
        "note": "Only manually filled labels are counted. Blank labels are excluded from denominators.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manual_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat = []
    for scope, item in summary.items():
        if scope == "note":
            continue
        for metric, values in item.items():
            if isinstance(values, dict) and "counts" in values:
                flat.append({
                    "scope": scope, "metric": metric,
                    "counts": json.dumps(values["counts"], ensure_ascii=False),
                    "primary_rate": next((v for k, v in values.items() if k != "counts"), None),
                })
    with (out / "manual_audit_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "metric", "counts", "primary_rate"])
        writer.writeheader()
        writer.writerows(flat)
    print("[DONE] Human audit summary:", out / "manual_audit_summary.json")

if __name__ == "__main__":
    main()
