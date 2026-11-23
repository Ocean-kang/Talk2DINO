#!/usr/bin/env python3
"""Validate that Raw/ECSD JSONs share image IDs and contain exactly five captions/image."""
import argparse
import json
from pathlib import Path

from caption_policy import CAPTIONS_PER_IMAGE


def read(path, key):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in result:
            raise ValueError("Duplicate image_id {} in {}".format(image_id, path))
        captions = row.get(key)
        if not isinstance(captions, list) or len(captions) != CAPTIONS_PER_IMAGE:
            raise ValueError(
                "{} image_id={} must contain exactly {} captions".format(
                    path, image_id, CAPTIONS_PER_IMAGE
                )
            )
        if not all(isinstance(x, str) and x.strip() for x in captions):
            raise ValueError("{} image_id={} contains an empty caption".format(path, image_id))
        result[image_id] = captions
    return result


def validate(raw_path, ecsd_path):
    raw = read(raw_path, "raw_captions")
    ecsd = read(ecsd_path, "ecsd_captions")
    if set(raw) != set(ecsd):
        raise ValueError(
            "Raw/ECSD IDs differ: raw-only={} ecsd-only={}".format(
                sorted(set(raw) - set(ecsd))[:10], sorted(set(ecsd) - set(raw))[:10]
            )
        )
    print(
        "[PASS] {}: {} images, exactly {} Raw and ECSD captions/image".format(
            raw_path, len(raw), CAPTIONS_PER_IMAGE
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--ecsd", required=True)
    args = p.parse_args()
    validate(args.raw, args.ecsd)


if __name__ == "__main__":
    main()
