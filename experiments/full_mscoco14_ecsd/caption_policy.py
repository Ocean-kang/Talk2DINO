#!/usr/bin/env python3
"""Shared deterministic policy for normalizing COCO image captions to exactly five."""
from typing import Dict, Iterable, List, Tuple

CAPTIONS_PER_IMAGE = 5
CAPTION_SELECTION_POLICY = "sort_by_annotation_id_ascending_then_take_first_5"


def select_five_captions(
    annotations: Iterable[Dict],
    image_id: int,
    captions_per_image: int = CAPTIONS_PER_IMAGE,
) -> Tuple[List[str], int]:
    """Return exactly K captions, deterministically ordered by COCO annotation ID.

    COCO2014 annotation JSON may contain more than five captions for an image.
    Official annotation IDs are globally unique, so sorting by annotation ID and
    taking the first K yields a stable subset independent of JSON list ordering.

    Returns:
        (selected captions, number of discarded extra annotations)

    Raises:
        ValueError if fewer than K valid annotations are available, or if an
        annotation ID is missing/invalid.
    """
    rows = []
    for ann in annotations:
        if int(ann["image_id"]) != int(image_id):
            continue
        caption = ann.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Invalid caption for image_id={}".format(image_id))
        ann_id = ann.get("id")
        if ann_id is None:
            raise ValueError(
                "COCO annotation for image_id={} has no annotation id; "
                "cannot apply deterministic selection policy".format(image_id)
            )
        rows.append((int(ann_id), caption.strip()))

    rows.sort(key=lambda item: (item[0], item[1]))
    if len(rows) < captions_per_image:
        raise ValueError(
            "image_id={} has only {} valid captions; at least {} are required".format(
                image_id, len(rows), captions_per_image
            )
        )
    selected = [caption for _, caption in rows[:captions_per_image]]
    return selected, len(rows) - captions_per_image
