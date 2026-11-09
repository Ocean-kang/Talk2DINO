#!/usr/bin/env python3
"""Prepare, assemble, and audit the 5K Rich-Diversity Talk2DINO experiment.

This is an isolated side-car experiment. It reuses the repository's original
DINO feature extractor, CLIP text feature extractor, train.py and segmentation
evaluator. No core Talk2DINO source file is modified.
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
    "rich_diversity": "rich_diversity_captions",
}


def torch_load(path):
    """Support the current Talk2DINO PyTorch environment and older torch APIs."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_captions(path, expected_images=5000):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
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
                raise ValueError(
                    "image_id={} field={} contains an empty/non-string caption.".format(
                        image_id, field
                    )
                )
        caption_map[image_id] = row

    if expected_images is not None and len(caption_map) != expected_images:
        raise ValueError(
            "Expected {} unique images, found {}.".format(
                expected_images, len(caption_map)
            )
        )
    return caption_map


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_image(image_root, image_id):
    """Resolve COCO 2014 train/val images in common subset directory layouts."""
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
    searched = "\n".join("  - {}".format(x) for x in directories)
    raise FileNotFoundError(
        "Cannot locate COCO image_id={} under {}.\nSearched:\n{}\n"
        "Expected a filename such as {} or {}.".format(
            image_id, image_root, searched, filenames[0], filenames[1]
        )
    )


def init(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work_dir = Path(args.work).resolve()
    image_root = Path(args.images).resolve()
    stage_dir = work_dir / "images" / "train2014"
    stage_dir.mkdir(parents=True, exist_ok=True)

    images, manifest = [], []
    for image_id in sorted(caption_map):
        source = resolve_image(image_root, image_id)
        staged_name = "COCO_train2014_{:012d}.jpg".format(image_id)
        target = stage_dir / staged_name
        if target.is_symlink():
            if not target.exists() or target.resolve() != source:
                raise RuntimeError(
                    "Existing staged symlink points elsewhere: {}\nExpected: {}\n"
                    "Use a fresh WORK_DIR or remove the stale link manually.".format(
                        target, source
                    )
                )
        elif target.exists():
            if not filecmp.cmp(str(target), str(source), shallow=False):
                raise RuntimeError(
                    "Existing staged file differs from source: {}. Use a fresh WORK_DIR.".format(
                        target
                    )
                )
        else:
            target.symlink_to(source)

        images.append({"id": image_id, "file_name": staged_name})
        manifest.append(
            {
                "image_id": image_id,
                "staged_file": str(target),
                "source_file": str(source),
            }
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
            "caption_arms": ARMS,
            "note": (
                "Staged filenames use the COCO_train2014 prefix so the original "
                "DINO extractor reads them from WORK_DIR/images/train2014."
            ),
        },
        work_dir / "prepare_metadata.json",
    )
    print("[init] staged {} images".format(len(images)))
    print("[init] base annotation: {}".format(work_dir / "base.pth"))


def assemble(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work_dir = Path(args.work).resolve()
    dino_path = work_dir / "dino.pth"
    if not dino_path.is_file():
        raise FileNotFoundError(
            "Missing {}. Run dino_extraction_v2.py before assemble.".format(dino_path)
        )

    dino_data = torch_load(dino_path)
    if "images" not in dino_data:
        raise ValueError("dino.pth does not contain an 'images' list.")
    image_by_id = {}
    for image in dino_data["images"]:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError("Duplicate image_id in dino.pth: {}".format(image_id))
        image_by_id[image_id] = image

    caption_ids = set(caption_map)
    dino_ids = set(image_by_id)
    if caption_ids != dino_ids:
        raise ValueError(
            "DINO feature IDs and caption IDs differ. Missing features (first 20): {}; "
            "extra features (first 20): {}".format(
                sorted(caption_ids - dino_ids)[:20], sorted(dino_ids - caption_ids)[:20]
            )
        )
    ids = sorted(caption_ids)
    if args.val_images <= 0 or args.val_images >= len(ids):
        raise ValueError("--val-images must be between 1 and image_count - 1.")

    rng = random.Random(args.seed)
    shuffled_ids = ids[:]
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
            output_path = work_dir / "{}_{}.pth".format(arm, split)
            torch.save({"images": images, "annotations": annotations}, output_path)
            print(
                "[assemble] {}_{}: {} images / {} captions -> {}".format(
                    arm, split, len(images), len(annotations), output_path
                )
            )
    print("[assemble] shared validation IDs: {}".format(work_dir / "val_ids.json"))


def audit(args):
    work_dir = Path(args.work).resolve()
    val_ids_path = work_dir / "val_ids.json"
    if not val_ids_path.is_file():
        raise FileNotFoundError("Missing {}. Run assemble first.".format(val_ids_path))
    val_ids = set(
        int(value) for value in json.loads(val_ids_path.read_text(encoding="utf-8"))
    )
    all_ids = set()

    for arm in ARMS:
        train_path = work_dir / "{}_train.pth".format(arm)
        val_path = work_dir / "{}_val.pth".format(arm)
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError("Missing train/val feature files for arm={}.".format(arm))
        split_ids = {}
        for split, path in (("train", train_path), ("val", val_path)):
            data = torch_load(path)
            image_ids = [int(image["id"]) for image in data["images"]]
            if len(image_ids) != len(set(image_ids)):
                raise ValueError("{} contains duplicate image IDs.".format(path))
            ids = set(image_ids)
            split_ids[split] = ids
            required_feature = (
                "disentangled_self_attn" if split == "train" else "avg_self_attn_out"
            )
            if not all(required_feature in image for image in data["images"]):
                raise ValueError("{} is missing DINO feature {}.".format(path, required_feature))
            expected_annotations = len(image_ids) * 5
            if len(data["annotations"]) != expected_annotations:
                raise ValueError(
                    "{} expected {} annotations, found {}.".format(
                        path, expected_annotations, len(data["annotations"])
                    )
                )
            if not all(
                isinstance(ann.get("caption"), str) and ann["caption"].strip()
                for ann in data["annotations"]
            ):
                raise ValueError("{} contains an empty/missing caption.".format(path))
            if not all(torch.is_tensor(ann.get("ann_feats")) for ann in data["annotations"]):
                raise ValueError(
                    "{} is missing CLIP ann_feats. Run text_features_extraction.py first.".format(
                        path
                    )
                )
            annotation_ids = {int(ann["image_id"]) for ann in data["annotations"]}
            if annotation_ids != ids:
                raise ValueError("{} annotation image IDs do not match its image IDs.".format(path))

        if split_ids["train"] & split_ids["val"]:
            raise ValueError("{} has train/validation image overlap.".format(arm))
        if len(split_ids["train"]) + len(split_ids["val"]) != args.expected_images:
            raise ValueError("{} train + val image count is incorrect.".format(arm))
        if split_ids["val"] != val_ids:
            raise ValueError("{} validation IDs differ from shared val_ids.json.".format(arm))
        arm_ids = split_ids["train"] | split_ids["val"]
        if all_ids and arm_ids != all_ids:
            raise ValueError("Image ID sets differ across caption arms.")
        all_ids = arm_ids
        print(
            "[audit] {}: train={} images, val={} images, train_caps={}, val_caps={}".format(
                arm,
                len(split_ids["train"]),
                len(split_ids["val"]),
                len(split_ids["train"]) * 5,
                len(split_ids["val"]) * 5,
            )
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
    print("[audit] PASS -> {}".format(work_dir / "audit_report.json"))


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
        "assemble", help="Create four caption arms with one shared train/validation split."
    )
    assemble_parser.add_argument("--captions", required=True)
    assemble_parser.add_argument("--work", required=True)
    assemble_parser.add_argument("--seed", type=int, default=123)
    assemble_parser.add_argument("--val-images", type=int, default=500)
    assemble_parser.add_argument("--expected-images", type=int, default=5000)
    assemble_parser.set_defaults(func=assemble)

    audit_parser = subparsers.add_parser("audit", help="Verify all eight feature files.")
    audit_parser.add_argument("--work", required=True)
    audit_parser.add_argument("--expected-images", type=int, default=5000)
    audit_parser.add_argument("--val-images", type=int, default=500)
    audit_parser.set_defaults(func=audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
