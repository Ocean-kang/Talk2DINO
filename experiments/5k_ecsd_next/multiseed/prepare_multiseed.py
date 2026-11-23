#!/usr/bin/env python3
"""Prepare shared DINO features and one fixed 4500/500 split for four caption arms."""
import argparse
import filecmp
import json
import random
from pathlib import Path

import torch

ARMS = {
    "raw": "raw_captions",
    "rich": "rich_captions",
    "rich_diversity": "rich_diversity_captions",
    "ecsdv2": "ecsd_v2_captions",
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
        raise ValueError("Caption JSON must be a list.")
    result = {}
    for index, row in enumerate(rows):
        if "image_id" not in row:
            raise ValueError("Record {} is missing image_id.".format(index))
        image_id = int(row["image_id"])
        if image_id in result:
            raise ValueError("Duplicate image_id {}.".format(image_id))
        for field in ARMS.values():
            captions = row.get(field)
            if not isinstance(captions, list) or len(captions) != 5:
                raise ValueError("image_id={} field={} must have exactly five captions.".format(image_id, field))
            if not all(isinstance(x, str) and x.strip() for x in captions):
                raise ValueError("image_id={} field={} contains empty captions.".format(image_id, field))
        result[image_id] = row
    if expected_images and len(result) != expected_images:
        raise ValueError("Expected {} images, found {}.".format(expected_images, len(result)))
    return result

def resolve_image(image_root, image_id):
    root = Path(image_root).resolve()
    names = (
        "COCO_train2014_{:012d}.jpg".format(image_id),
        "COCO_val2014_{:012d}.jpg".format(image_id),
    )
    dirs = [
        root, root / "images", root / "train2014", root / "val2014",
        root / "images" / "train2014", root / "images" / "val2014",
    ]
    for directory in dirs:
        for name in names:
            path = directory / name
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError("Cannot find COCO image_id={} under {}.".format(image_id, root))

def init(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work = Path(args.work).resolve()
    stage_dir = work / "images" / "train2014"
    stage_dir.mkdir(parents=True, exist_ok=True)
    images, manifest = [], []
    for image_id in sorted(caption_map):
        source = resolve_image(args.images, image_id)
        staged_name = "COCO_train2014_{:012d}.jpg".format(image_id)
        target = stage_dir / staged_name
        if target.is_symlink():
            if not target.exists() or target.resolve() != source:
                raise RuntimeError("Stale image symlink: {}. Use a clean work directory.".format(target))
        elif target.exists():
            if not filecmp.cmp(str(target), str(source), shallow=False):
                raise RuntimeError("Staged image differs from source: {}.".format(target))
        else:
            target.symlink_to(source)
        images.append({"id": image_id, "file_name": staged_name})
        manifest.append({"image_id": image_id, "source_file": str(source), "staged_file": str(target)})
    torch.save({"images": images, "annotations": []}, work / "base.pth")
    save_json(manifest, work / "image_manifest.json")
    save_json({
        "num_images": len(images), "caption_json": str(Path(args.captions).resolve()),
        "image_root": str(Path(args.images).resolve()), "caption_arms": ARMS,
        "split_seed": args.split_seed, "val_images": args.val_images,
    }, work / "prepare_metadata.json")

    if args.dino_source:
        source = Path(args.dino_source).resolve()
        if not source.is_file():
            raise FileNotFoundError("Configured DINO feature source does not exist: {}".format(source))
        target = work / "dino.pth"
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError("{} already exists and points elsewhere.".format(target))
        else:
            target.symlink_to(source)
    print("[init] staged {} images under {}".format(len(images), stage_dir))
    print("[init] DINO features must be available at {}".format(work / "dino.pth"))

def assemble(args):
    caption_map = load_captions(args.captions, args.expected_images)
    work = Path(args.work).resolve()
    dino_path = work / "dino.pth"
    if not dino_path.is_file():
        raise FileNotFoundError(
            "Missing {}. Point --dino-source to an existing compatible 5K DINO feature file. "
            "This script does not download or extract models.".format(dino_path)
        )
    dino = torch_load(dino_path)
    if not isinstance(dino, dict) or not isinstance(dino.get("images"), list):
        raise ValueError("DINO feature file must contain an images list.")
    image_by_id = {}
    for image in dino["images"]:
        image_id = int(image["id"])
        if image_id in image_by_id:
            raise ValueError("Duplicate DINO image ID {}.".format(image_id))
        image_by_id[image_id] = image
    if set(image_by_id) != set(caption_map):
        raise ValueError("DINO image IDs and caption image IDs are not identical.")

    ids = sorted(caption_map)
    if args.val_images <= 0 or args.val_images >= len(ids):
        raise ValueError("--val-images must be between 1 and image_count - 1.")
    shuffled = ids[:]
    random.Random(args.split_seed).shuffle(shuffled)
    val_ids = set(shuffled[:args.val_images])
    train_ids = set(ids) - val_ids
    save_json(sorted(val_ids), work / "val_ids.json")
    save_json({
        "split_seed": args.split_seed, "num_images": len(ids),
        "train_images": len(train_ids), "val_images": len(val_ids),
        "train_val_split": "{}/{}".format(len(train_ids), len(val_ids)),
        "captions_per_image": 5, "arms": ARMS,
        "note": "This split is fixed across all training seeds 123, 456, and 789.",
    }, work / "split_metadata.json")

    for arm, field in ARMS.items():
        for split, selected_ids in (("train", train_ids), ("val", val_ids)):
            ordered = sorted(selected_ids)
            images = [image_by_id[i] for i in ordered]
            annotations = []
            for image_id in ordered:
                for caption_index, caption in enumerate(caption_map[image_id][field]):
                    annotations.append({
                        "id": image_id * 10 + caption_index,
                        "image_id": image_id,
                        "caption": caption.strip(),
                    })
            path = work / "{}_{}.pth".format(arm, split)
            if path.exists() and not args.overwrite:
                raise FileExistsError("{} exists; pass --overwrite to rebuild.".format(path))
            torch.save({"images": images, "annotations": annotations}, path)
            print("[assemble] {}_{}: {} images / {} captions".format(
                arm, split, len(images), len(annotations)))

def audit(args):
    work = Path(args.work).resolve()
    val_path = work / "val_ids.json"
    if not val_path.is_file():
        raise FileNotFoundError("Missing {}. Run assemble first.".format(val_path))
    val_ids = set(json.loads(val_path.read_text(encoding="utf-8")))
    all_arm_ids = None
    for arm in ARMS:
        split_ids = {}
        for split in ("train", "val"):
            path = work / "{}_{}.pth".format(arm, split)
            if not path.is_file():
                raise FileNotFoundError("Missing {}".format(path))
            data = torch_load(path)
            ids = [int(x["id"]) for x in data["images"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate image IDs in {}".format(path))
            split_ids[split] = set(ids)
            required = "disentangled_self_attn" if split == "train" else "avg_self_attn_out"
            if not all(required in x for x in data["images"]):
                raise ValueError("{} lacks DINO field {}.".format(path, required))
            if len(data["annotations"]) != len(ids) * 5:
                raise ValueError("{} has an incorrect annotation count.".format(path))
            if not all(isinstance(a.get("caption"), str) and a["caption"].strip()
                       for a in data["annotations"]):
                raise ValueError("{} contains empty captions.".format(path))
            if not all(torch.is_tensor(a.get("ann_feats")) for a in data["annotations"]):
                raise ValueError("{} lacks CLIP ann_feats; run text_features_extraction.py.".format(path))
            if {int(a["image_id"]) for a in data["annotations"]} != split_ids[split]:
                raise ValueError("{} annotation IDs do not match image IDs.".format(path))
        if split_ids["train"] & split_ids["val"]:
            raise ValueError("Train/validation overlap in arm {}.".format(arm))
        if split_ids["val"] != val_ids:
            raise ValueError("Validation IDs differ from shared val_ids.json in arm {}.".format(arm))
        ids = split_ids["train"] | split_ids["val"]
        if len(ids) != args.expected_images:
            raise ValueError("Incorrect total image count in arm {}.".format(arm))
        if all_arm_ids is not None and ids != all_arm_ids:
            raise ValueError("Image IDs differ across arms.")
        all_arm_ids = ids
        print("[audit] {} PASS".format(arm))
    report = {
        "status": "ok", "num_images": len(all_arm_ids),
        "num_train_images": args.expected_images - args.val_images,
        "num_val_images": args.val_images, "captions_per_image": 5,
        "arms": list(ARMS), "same_ids_across_arms": True,
        "same_validation_ids_across_arms": True,
        "dino_features_present": True, "clip_features_present": True,
    }
    save_json(report, work / "audit_report.json")
    print("[audit] PASS ->", work / "audit_report.json")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--captions", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--dino-source", default=None)
    p.add_argument("--expected-images", type=int, default=5000)
    p.add_argument("--split-seed", type=int, default=123)
    p.add_argument("--val-images", type=int, default=500)
    p.set_defaults(func=init)
    p = sub.add_parser("assemble")
    p.add_argument("--captions", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--expected-images", type=int, default=5000)
    p.add_argument("--split-seed", type=int, default=123)
    p.add_argument("--val-images", type=int, default=500)
    p.add_argument("--overwrite", action="store_true")
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
