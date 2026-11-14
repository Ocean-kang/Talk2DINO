#!/usr/bin/env python3
"""Prepare and audit the five-arm ECSD subset5K Talk2DINO experiment."""
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
    "rich_diversity": "rich_diversity_captions",
    "ecsd": "ecsd_captions",
}


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_captions(path, expected_images=5000):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Caption JSON must be a list of image records.")
    caption_map = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or "image_id" not in row:
            raise ValueError("Record {} is missing image_id.".format(row_index))
        image_id = int(row["image_id"])
        if image_id in caption_map:
            raise ValueError("Duplicate image_id: {}".format(image_id))
        for field in ARMS.values():
            captions = row.get(field)
            if not isinstance(captions, list) or len(captions) != 5:
                raise ValueError(
                    "image_id={} field={} must contain exactly five captions.".format(
                        image_id, field
                    )
                )
            if not all(isinstance(x, str) and x.strip() for x in captions):
                raise ValueError("Invalid caption at image_id={} field={}".format(
                    image_id, field
                ))
        caption_map[image_id] = row
    if expected_images is not None and len(caption_map) != expected_images:
        raise ValueError(
            "Expected {} unique images, found {}.".format(
                expected_images, len(caption_map)
            )
        )
    return caption_map


def resolve_image(image_root, image_id):
    image_root = Path(image_root).resolve()
    filenames = (
        "COCO_train2014_{:012d}.jpg".format(image_id),
        "COCO_val2014_{:012d}.jpg".format(image_id),
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
    raise FileNotFoundError(
        "Cannot find COCO image_id={} under {}. Expected filename such as {}.".format(
            image_id, image_root, filenames[0]
        )
    )


def init(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work_dir = Path(args.work).resolve()
    stage_dir = work_dir / "images" / "train2014"
    stage_dir.mkdir(parents=True, exist_ok=True)
    images, manifest = [], []

    for image_id in sorted(caption_map):
        source = resolve_image(args.images, image_id)
        staged_name = "COCO_train2014_{:012d}.jpg".format(image_id)
        target = stage_dir / staged_name
        if target.is_symlink():
            if not target.exists() or target.resolve() != source:
                raise RuntimeError(
                    "Stale image symlink: {}. Use a clean WORK_DIR.".format(target)
                )
        elif target.exists():
            if not filecmp.cmp(str(target), str(source), shallow=False):
                raise RuntimeError(
                    "Staged image differs from source: {}. Use a clean WORK_DIR.".format(target)
                )
        else:
            target.symlink_to(source)

        images.append({"id": image_id, "file_name": staged_name})
        manifest.append({
            "image_id": image_id,
            "source_file": str(source),
            "staged_file": str(target),
        })

    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "annotations": []}, work_dir / "base.pth")
    save_json(manifest, work_dir / "image_manifest.json")
    save_json({
        "num_images": len(images),
        "caption_json": str(Path(args.captions).resolve()),
        "image_root": str(Path(args.images).resolve()),
        "caption_arms": ARMS,
        "image_staging": "symlinks",
    }, work_dir / "prepare_metadata.json")
    print("[init] staged {} images".format(len(images)))


def assemble(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work_dir = Path(args.work).resolve()
    dino_path = work_dir / "dino.pth"
    if not dino_path.is_file():
        raise FileNotFoundError("Missing {}. Run DINO extraction first.".format(dino_path))

    dino_data = torch_load(dino_path)
    if "images" not in dino_data:
        raise ValueError("dino.pth does not contain an images list.")
    image_by_id = {}
    for image in dino_data["images"]:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError("Duplicate image ID in DINO features: {}".format(image_id))
        image_by_id[image_id] = image

    caption_ids = set(caption_map)
    dino_ids = set(image_by_id)
    if caption_ids != dino_ids:
        raise ValueError(
            "Caption/DINO image IDs differ. Missing DINO IDs (first 20): {}; "
            "extra DINO IDs (first 20): {}".format(
                sorted(caption_ids - dino_ids)[:20], sorted(dino_ids - caption_ids)[:20]
            )
        )

    ids = sorted(caption_ids)
    if args.val_images <= 0 or args.val_images >= len(ids):
        raise ValueError("--val-images must be between 1 and image_count - 1.")
    shuffled = ids[:]
    random.Random(args.seed).shuffle(shuffled)
    val_ids = set(shuffled[:args.val_images])
    train_ids = caption_ids - val_ids

    save_json(sorted(val_ids), work_dir / "val_ids.json")
    save_json({
        "seed": args.seed,
        "num_images": len(ids),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "captions_per_image": 5,
        "train_annotations_per_arm": len(train_ids) * 5,
        "val_annotations_per_arm": len(val_ids) * 5,
        "arms": list(ARMS),
    }, work_dir / "split_metadata.json")

    for arm, field in ARMS.items():
        for split, selected_ids in (("train", train_ids), ("val", val_ids)):
            selected = sorted(selected_ids)
            images = [image_by_id[image_id] for image_id in selected]
            annotations = []
            for image_id in selected:
                for caption_index, caption in enumerate(caption_map[image_id][field]):
                    annotations.append({
                        "id": image_id * 10 + caption_index,
                        "image_id": image_id,
                        "caption": caption.strip(),
                    })
            output_path = work_dir / "{}_{}.pth".format(arm, split)
            torch.save({"images": images, "annotations": annotations}, output_path)
            print("[assemble] {}_{}: {} images / {} captions".format(
                arm, split, len(images), len(annotations)
            ))


def audit(args):
    work_dir = Path(args.work).resolve()
    val_path = work_dir / "val_ids.json"
    if not val_path.is_file():
        raise FileNotFoundError("Missing val_ids.json. Run assemble first.")
    val_ids = set(int(x) for x in json.loads(val_path.read_text(encoding="utf-8")))
    all_ids = None

    for arm in ARMS:
        split_ids = {}
        for split in ("train", "val"):
            path = work_dir / "{}_{}.pth".format(arm, split)
            if not path.is_file():
                raise FileNotFoundError("Missing {}".format(path))
            data = torch_load(path)
            ids = [int(image["id"]) for image in data["images"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate image IDs in {}".format(path))
            split_ids[split] = set(ids)
            required = "disentangled_self_attn" if split == "train" else "avg_self_attn_out"
            if not all(required in image for image in data["images"]):
                raise ValueError("{} missing DINO feature {}".format(path, required))
            if len(data["annotations"]) != len(ids) * 5:
                raise ValueError("{} annotation count is not image_count * 5".format(path))
            if not all(
                isinstance(ann.get("caption"), str) and ann["caption"].strip()
                for ann in data["annotations"]
            ):
                raise ValueError("{} has invalid captions".format(path))
            if not all(torch.is_tensor(ann.get("ann_feats")) for ann in data["annotations"]):
                raise ValueError(
                    "{} missing CLIP ann_feats. Run text_features_extraction.py first.".format(path)
                )
            if {int(ann["image_id"]) for ann in data["annotations"]} != split_ids[split]:
                raise ValueError("{} annotation IDs do not match image IDs".format(path))

        if split_ids["train"] & split_ids["val"]:
            raise ValueError("Train/val overlap for arm {}".format(arm))
        if split_ids["val"] != val_ids:
            raise ValueError("Validation IDs differ from shared val_ids.json for {}".format(arm))
        arm_ids = split_ids["train"] | split_ids["val"]
        if len(arm_ids) != args.expected_images:
            raise ValueError("Incorrect image count for arm {}".format(arm))
        if all_ids is not None and arm_ids != all_ids:
            raise ValueError("Image ID sets differ across caption arms")
        all_ids = arm_ids
        print("[audit] {} PASS".format(arm))

    report = {
        "status": "ok",
        "num_images": len(all_ids),
        "num_train_images": args.expected_images - args.val_images,
        "num_val_images": args.val_images,
        "captions_per_image": 5,
        "arms": list(ARMS),
        "same_image_ids_across_arms": True,
        "same_validation_ids_across_arms": True,
        "dino_features_present": True,
        "clip_features_present": True,
    }
    save_json(report, work_dir / "audit_report.json")
    print("[audit] PASS -> {}".format(work_dir / "audit_report.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("--captions", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--expected-images", type=int, default=5000)
    p.set_defaults(func=init)

    p = sub.add_parser("assemble")
    p.add_argument("--captions", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--val-images", type=int, default=500)
    p.add_argument("--expected-images", type=int, default=5000)
    p.set_defaults(func=assemble)

    p = sub.add_parser("audit")
    p.add_argument("--work", required=True)
    p.add_argument("--expected-images", type=int, default=5000)
    p.add_argument("--val-images", type=int, default=500)
    p.set_defaults(func=audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
