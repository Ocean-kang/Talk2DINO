#!/usr/bin/env python3
"""Prepare and audit the 5K Raw / Rich / Rich-Weaken Talk2DINO experiment.

This side-car script does not modify Talk2DINO source files. It creates:
  - a staged image manifest and base.pth for DINO feature extraction;
  - identical train/validation image splits for all three caption arms;
  - six COCO-style feature files consumed by the original train.py.
"""
import argparse
import filecmp
import json
import random
from pathlib import Path

import torch


ARMS = {
    "raw": "raw_captions",
    "rich": "rich_captions",
    "rich_weaken": "rich_weaken_captions",
}


def load_captions(path, expected_images=5000):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if not isinstance(rows, list):
        raise ValueError("Caption JSON must be a list of image records.")

    caption_map = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or "image_id" not in row:
            raise ValueError(f"Record {row_index} is missing image_id.")

        image_id = int(row["image_id"])
        if image_id in caption_map:
            raise ValueError(f"Duplicate image_id: {image_id}")

        for field in ARMS.values():
            captions = row.get(field)
            if not isinstance(captions, list) or len(captions) != 5:
                raise ValueError(
                    f"image_id={image_id}: {field} must contain exactly 5 captions."
                )
            if not all(isinstance(item, str) and item.strip() for item in captions):
                raise ValueError(
                    f"image_id={image_id}: {field} contains an empty/non-string caption."
                )

        caption_map[image_id] = row

    if expected_images is not None and len(caption_map) != expected_images:
        raise ValueError(
            f"Expected {expected_images} unique images, found {len(caption_map)}."
        )

    return caption_map


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_image(image_root, image_id):
    """Resolve COCO 2014 train/val filenames in common subset layouts."""
    image_root = Path(image_root).resolve()
    filenames = (
        f"COCO_train2014_{image_id:012d}.jpg",
        f"COCO_val2014_{image_id:012d}.jpg",
    )
    directories = [
        image_root,
        image_root / "images",
        image_root / "train2014",
        image_root / "val2014",
        image_root / "images" / "train2014",
        image_root / "images" / "val2014",
    ]

    for directory in directories:
        for filename in filenames:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()

    searched = "\n".join(f"  - {directory}" for directory in directories)
    raise FileNotFoundError(
        f"Cannot locate COCO image for image_id={image_id} under {image_root}.\n"
        f"Searched directories:\n{searched}\n"
        f"Expected filename such as {filenames[0]} or {filenames[1]}."
    )


def init(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work_dir = Path(args.work).resolve()
    image_root = Path(args.images).resolve()

    stage_dir = work_dir / "images" / "train2014"
    stage_dir.mkdir(parents=True, exist_ok=True)

    images = []
    manifest = []
    for image_id in sorted(caption_map):
        source = resolve_image(image_root, image_id)
        staged_name = f"COCO_train2014_{image_id:012d}.jpg"
        target = stage_dir / staged_name

        if target.is_symlink():
            if not target.exists() or target.resolve() != source:
                raise RuntimeError(
                    f"Existing staged symlink points elsewhere: {target}\n"
                    f"Expected: {source}\n"
                    "Remove the stale link manually or use a fresh WORK_DIR."
                )
        elif target.exists():
            if not filecmp.cmp(target, source, shallow=False):
                raise RuntimeError(
                    f"Existing staged file differs from source: {target}\n"
                    "Use a fresh WORK_DIR rather than silently mixing images."
                )
        else:
            target.symlink_to(source)

        images.append({"id": image_id, "file_name": staged_name})
        manifest.append(
            {"image_id": image_id, "staged_file": str(target), "source_file": str(source)}
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "annotations": []}, work_dir / "base.pth")
    save_json(manifest, work_dir / "image_manifest.json")
    save_json(
        {
            "num_images": len(images),
            "caption_json": str(Path(args.captions).resolve()),
            "image_root": str(image_root),
            "staged_images": str(stage_dir),
            "note": "All staged filenames use COCO_train2014 prefix so the original DINO extractor reads them from work/images/train2014.",
        },
        work_dir / "prepare_metadata.json",
    )

    print(f"[init] staged {len(images)} images")
    print(f"[init] base annotation: {work_dir / 'base.pth'}")
    print(f"[init] image manifest: {work_dir / 'image_manifest.json'}")


def assemble(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work_dir = Path(args.work).resolve()
    dino_path = work_dir / "dino.pth"
    if not dino_path.is_file():
        raise FileNotFoundError(
            f"Missing {dino_path}. Run dino_extraction_v2.py before assemble."
        )

    dino_data = torch.load(dino_path, map_location="cpu", weights_only=False)
    if "images" not in dino_data:
        raise ValueError("dino.pth does not contain an 'images' list.")

    image_by_id = {}
    for image in dino_data["images"]:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError(f"Duplicate image_id in dino.pth: {image_id}")
        image_by_id[image_id] = image

    caption_ids = set(caption_map)
    dino_ids = set(image_by_id)
    if caption_ids != dino_ids:
        missing_features = sorted(caption_ids - dino_ids)[:20]
        extra_features = sorted(dino_ids - caption_ids)[:20]
        raise ValueError(
            "DINO feature IDs and caption IDs do not match.\n"
            f"Missing DINO IDs (first 20): {missing_features}\n"
            f"Extra DINO IDs (first 20): {extra_features}"
        )

    ids = sorted(caption_ids)
    if args.val_images <= 0 or args.val_images >= len(ids):
        raise ValueError("--val-images must be between 1 and image_count - 1.")

    rng = random.Random(args.seed)
    shuffled_ids = ids.copy()
    rng.shuffle(shuffled_ids)
    val_ids = set(shuffled_ids[: args.val_images])
    train_ids = caption_ids - val_ids

    save_json(sorted(val_ids), work_dir / "val_ids.json")
    save_json(
        {
            "seed": args.seed,
            "num_images": len(ids),
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "captions_per_image": 5,
            "train_annotations_per_arm": len(train_ids) * 5,
            "val_annotations_per_arm": len(val_ids) * 5,
            "arms": ARMS,
        },
        work_dir / "split_metadata.json",
    )

    for arm, caption_field in ARMS.items():
        for split, selected_ids in (("train", train_ids), ("val", val_ids)):
            selected_sorted = sorted(selected_ids)
            images = [image_by_id[image_id] for image_id in selected_sorted]
            annotations = []

            for image_id in selected_sorted:
                for caption_index, caption in enumerate(caption_map[image_id][caption_field]):
                    annotations.append(
                        {
                            "id": image_id * 10 + caption_index,
                            "image_id": image_id,
                            "caption": caption.strip(),
                        }
                    )

            output_path = work_dir / f"{arm}_{split}.pth"
            torch.save({"images": images, "annotations": annotations}, output_path)
            print(
                f"[assemble] {arm}_{split}: "
                f"{len(images)} images / {len(annotations)} captions -> {output_path}"
            )

    print(f"[assemble] fixed split saved to {work_dir / 'val_ids.json'}")


def audit(args):
    work_dir = Path(args.work).resolve()
    val_ids_path = work_dir / "val_ids.json"
    if not val_ids_path.is_file():
        raise FileNotFoundError(f"Missing {val_ids_path}. Run assemble first.")

    val_ids = set(int(value) for value in json.loads(val_ids_path.read_text(encoding="utf-8")))
    all_ids = set()
    for arm in ARMS:
        train_path = work_dir / f"{arm}_train.pth"
        val_path = work_dir / f"{arm}_val.pth"
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(f"Missing train/val files for arm={arm}.")

        split_ids = {}
        for split, path in (("train", train_path), ("val", val_path)):
            data = torch.load(path, map_location="cpu", weights_only=False)
            image_ids = [int(image["id"]) for image in data["images"]]
            if len(image_ids) != len(set(image_ids)):
                raise ValueError(f"{path}: duplicate image IDs.")
            ids = set(image_ids)
            split_ids[split] = ids

            if split == "train":
                if not all("disentangled_self_attn" in image for image in data["images"]):
                    raise ValueError(f"{path}: missing DINO disentangled_self_attn features.")
            if split == "val":
                if not all("avg_self_attn_out" in image for image in data["images"]):
                    raise ValueError(f"{path}: missing DINO avg_self_attn_out features.")

            expected_annotations = len(image_ids) * 5
            if len(data["annotations"]) != expected_annotations:
                raise ValueError(
                    f"{path}: expected {expected_annotations} annotations, "
                    f"found {len(data['annotations'])}."
                )

            if not all(
                isinstance(ann.get("caption"), str) and ann["caption"].strip()
                for ann in data["annotations"]
            ):
                raise ValueError(f"{path}: empty/missing caption text.")

            if not all(torch.is_tensor(ann.get("ann_feats")) for ann in data["annotations"]):
                raise ValueError(
                    f"{path}: missing CLIP ann_feats. Run text_features_extraction.py first."
                )

        if split_ids["train"] & split_ids["val"]:
            raise ValueError(f"{arm}: train/validation image overlap.")
        if len(split_ids["train"]) + len(split_ids["val"]) != args.expected_images:
            raise ValueError(f"{arm}: train + val image count is not {args.expected_images}.")
        if split_ids["val"] != val_ids:
            raise ValueError(f"{arm}: validation IDs differ from shared val_ids.json.")

        arm_ids = split_ids["train"] | split_ids["val"]
        if all_ids and arm_ids != all_ids:
            raise ValueError("Image ID sets differ across caption arms.")
        all_ids = arm_ids

        print(
            f"[audit] {arm}: train={len(split_ids['train'])} images, "
            f"val={len(split_ids['val'])} images, "
            f"train captions={len(split_ids['train']) * 5}, "
            f"val captions={len(split_ids['val']) * 5}"
        )

    report = {
        "status": "ok",
        "num_images": len(all_ids),
        "num_train_images": args.expected_images - args.val_images,
        "num_val_images": args.val_images,
        "captions_per_image": 5,
        "arms": list(ARMS),
        "same_image_ids_across_arms": True,
        "same_validation_ids_across_arms": True,
        "clip_features_present": True,
        "dino_features_present": True,
    }
    save_json(report, work_dir / "audit_report.json")
    print(f"[audit] PASS -> {work_dir / 'audit_report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Stage the 5K COCO images.")
    init_parser.add_argument("--captions", required=True)
    init_parser.add_argument("--images", required=True, help="Image root or subset data root.")
    init_parser.add_argument("--work", required=True)
    init_parser.add_argument("--expected-images", type=int, default=5000)
    init_parser.set_defaults(func=init)

    assemble_parser = subparsers.add_parser(
        "assemble", help="Create three arms with one shared train/validation split."
    )
    assemble_parser.add_argument("--captions", required=True)
    assemble_parser.add_argument("--work", required=True)
    assemble_parser.add_argument("--seed", type=int, default=123)
    assemble_parser.add_argument("--val-images", type=int, default=500)
    assemble_parser.add_argument("--expected-images", type=int, default=5000)
    assemble_parser.set_defaults(func=assemble)

    audit_parser = subparsers.add_parser(
        "audit", help="Verify all six feature files before training."
    )
    audit_parser.add_argument("--work", required=True)
    audit_parser.add_argument("--expected-images", type=int, default=5000)
    audit_parser.add_argument("--val-images", type=int, default=500)
    audit_parser.set_defaults(func=audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
