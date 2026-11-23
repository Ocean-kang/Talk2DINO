#!/usr/bin/env python3
"""Convert official COCO2014 caption annotations to exactly five captions/image.

COCO annotation JSON can contain more than five captions for some images.
This script deterministically sorts each image's annotations by annotation ID
and keeps the first five. The normalized Raw set is the sole source for ECSD.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from caption_policy import (
    CAPTION_SELECTION_POLICY,
    CAPTIONS_PER_IMAGE,
    select_five_captions,
)


def load_split(annotation_path, split, return_stats=False):
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Expected COCO caption JSON with 'images' and 'annotations' lists")

    image_by_id = {}
    for image in images:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError("Duplicate image id {} in {}".format(image_id, annotation_path))
        file_name = image.get("file_name")
        if not file_name:
            raise ValueError("Image {} has no file_name".format(image_id))
        if split not in file_name.lower():
            raise ValueError(
                "Image {} filename {!r} does not look like split {!r}".format(
                    image_id, file_name, split
                )
            )
        image_by_id[image_id] = file_name

    grouped = defaultdict(list)
    seen_ann_ids = set()
    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_by_id:
            raise ValueError("Annotation references unknown image_id {}".format(image_id))
        ann_id = ann.get("id")
        if ann_id is None:
            raise ValueError(
                "COCO annotation for image_id={} has no annotation id".format(image_id)
            )
        ann_id = int(ann_id)
        if ann_id in seen_ann_ids:
            raise ValueError("Duplicate annotation id {}".format(ann_id))
        seen_ann_ids.add(ann_id)
        caption = ann.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Invalid caption for image_id {}".format(image_id))
        grouped[image_id].append({"id": ann_id, "image_id": image_id, "caption": caption.strip()})

    rows = []
    dropped_extra_annotations = 0
    source_caption_counts = {}
    bad_counts = []
    for image_id in sorted(image_by_id):
        source_count = len(grouped.get(image_id, []))
        source_caption_counts[image_id] = source_count
        if source_count < CAPTIONS_PER_IMAGE:
            bad_counts.append((image_id, source_count))
            continue
        if source_count == 0:
            bad_counts.append((image_id, source_count))
            continue

        captions, dropped = select_five_captions(
            grouped[image_id], image_id, CAPTIONS_PER_IMAGE
        )
        dropped_extra_annotations += dropped

        rows.append({"image_id": image_id, "raw_captions": captions})

    if bad_counts:
        raise ValueError(
            "Every image must have at least {} valid captions; examples (image_id,count): {}".format(
                CAPTIONS_PER_IMAGE, bad_counts[:20]
            )
        )
    if len(rows) != len(image_by_id):
        raise ValueError("Some images are missing captions")

    # Final invariant: the normalized Raw set has exactly five captions/image.
    invalid = [(x["image_id"], len(x["raw_captions"])) for x in rows
               if len(x["raw_captions"]) != CAPTIONS_PER_IMAGE]
    if invalid:
        raise AssertionError("Caption normalization failed: {}".format(invalid[:20]))

    stats = {
        "num_images": len(rows),
        "source_annotation_count": len(annotations),
        "source_caption_count_min": min(source_caption_counts.values()) if source_caption_counts else 0,
        "source_caption_count_max": max(source_caption_counts.values()) if source_caption_counts else 0,
        "images_with_more_than_five_source_captions": sum(
            1 for n in source_caption_counts.values() if n > CAPTIONS_PER_IMAGE
        ),
        "discarded_extra_annotations": dropped_extra_annotations,
        "normalized_caption_count": len(rows) * CAPTIONS_PER_IMAGE,
    }
    return (rows, stats) if return_stats else rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-annotations", required=True)
    p.add_argument("--val-annotations", required=True)
    p.add_argument("--output-dir", default="data/coco2014_full")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_stats = {}
    for split, ann_path in (("train", args.train_annotations), ("val", args.val_annotations)):
        rows, stats = load_split(ann_path, split, return_stats=True)
        path = out / "raw_{}.json".format(split)
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        split_stats[split] = stats
        print(
            "[{}] images={} normalized_captions={} source_max={} images_with_extra={} "
            "discarded_extra={} -> {}".format(
                split, stats["num_images"], stats["normalized_caption_count"],
                stats["source_caption_count_max"],
                stats["images_with_more_than_five_source_captions"],
                stats["discarded_extra_annotations"], path
            )
        )

    manifest = {
        "dataset": "MSCOCO2014",
        "caption_source": "official COCO2014 caption annotations",
        "captions_per_image": CAPTIONS_PER_IMAGE,
        "caption_selection_policy": CAPTION_SELECTION_POLICY,
        "selection_policy_description": (
            "For each image, sort source annotations by annotation id ascending "
            "and retain the first five; ECSD is generated only from this normalized Raw set."
        ),
        "splits": split_stats,
    }
    (out / "raw_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[DONE] {}".format(out / "raw_manifest.json"))


if __name__ == "__main__":
    main()
