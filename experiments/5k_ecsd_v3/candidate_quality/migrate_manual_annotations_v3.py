#!/usr/bin/env python3
"""Carry human labels from a v2 audit CSV to v3 by (image_id, surface-normalized text)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
from utils.safe_io import atomic_write_csv, atomic_write_json
from utils.text_utils import canonical_surface_text

LABEL_FIELDS = [
    "validity_label", "groundedness_label", "completeness_label",
    "category_correct_label", "redundancy_label", "annotator_notes",
]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def migrate_rows(old_rows, new_rows):
    index = defaultdict(list)
    for row in old_rows:
        key = (str(row.get("image_id", "")), canonical_surface_text(row.get("candidate_text", "")))
        index[key].append(row)
    migrated = 0
    conflicts = defaultdict(int)
    unmatched = 0
    output = []
    for new in new_rows:
        row = dict(new)
        key = (str(new.get("image_id", "")), canonical_surface_text(new.get("candidate_text", "")))
        matches = index.get(key, [])
        if not matches:
            unmatched += 1
            output.append(row)
            continue
        transferred_any = False
        for field in LABEL_FIELDS:
            values = sorted({str(x.get(field, "")).strip() for x in matches if str(x.get(field, "")).strip()})
            if field == "annotator_notes":
                if values:
                    row[field] = " | ".join(values)
                    transferred_any = True
            elif len(values) == 1:
                row[field] = values[0]
                transferred_any = True
            elif len(values) > 1:
                conflicts[field] += 1
        if transferred_any:
            migrated += 1
        output.append(row)
    report = {
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "new_rows_with_transferred_labels_or_notes": migrated,
        "unmatched_new_rows": unmatched,
        "conflicting_label_fields": dict(conflicts),
        "mapping_key": "(image_id, lowercase alphanumeric surface text preserving stopwords)",
        "selection_status": "not transferred; retained from v3 Candidate ID trace",
        "ambiguous_conflicts": "conflicting categorical labels are left blank rather than guessed",
    }
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-csv", required=True)
    parser.add_argument("--new-csv", required=True)
    parser.add_argument("--output-csv", default="outputs/5k_ecsd_v3/manual_audit/manual_annotation_v3_migrated.csv")
    parser.add_argument("--report", default="outputs/5k_ecsd_v3/manual_audit/annotation_migration_report.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    old_rows, new_rows = read_csv(args.old_csv), read_csv(args.new_csv)
    output, report = migrate_rows(old_rows, new_rows)
    fieldnames = list(new_rows[0].keys()) if new_rows else []
    atomic_write_csv(args.output_csv, fieldnames, output, overwrite=args.overwrite)
    atomic_write_json(args.report, report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
