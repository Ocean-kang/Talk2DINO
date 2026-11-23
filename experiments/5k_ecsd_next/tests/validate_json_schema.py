#!/usr/bin/env python3
"""Validate the Subset5K caption JSON contract without loading model dependencies."""
import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = [
    "raw_captions", "rich_captions", "rich_weaken_captions",
    "rich_diversity_captions",
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-images", type=int, default=5000)
    parser.add_argument("--require-ecsd-v2", action="store_true")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Top-level JSON must be a list.")
    if args.expected_images and len(rows) != args.expected_images:
        raise ValueError("Expected {} rows, got {}.".format(args.expected_images, len(rows)))
    ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "image_id" not in row:
            raise ValueError("Row {} missing image_id.".format(index))
        ids.append(int(row["image_id"]))
        for field in REQUIRED_FIELDS + (["ecsd_v2_captions"] if args.require_ecsd_v2 else []):
            value = row.get(field)
            if not isinstance(value, list) or len(value) != 5:
                raise ValueError("image_id={} field={} must have exactly five captions.".format(row["image_id"], field))
            if not all(isinstance(x, str) and x.strip() for x in value):
                raise ValueError("image_id={} field={} contains empty/non-string captions.".format(row["image_id"], field))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs.")
    print("[PASS] {} images; five captions per required field; unique image IDs.".format(len(ids)))

if __name__ == "__main__":
    main()
