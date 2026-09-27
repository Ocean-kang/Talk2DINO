"""Prepare paired COCO caption files for the repository's feature extractors."""
import argparse
import json
import random
import re
from pathlib import Path

import torch


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def resolve_image(root, image_id):
    names = [f"COCO_train2014_{image_id:012d}.jpg", f"COCO_val2014_{image_id:012d}.jpg"]
    for directory in (root, root / "train2014", root / "val2014"):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Missing COCO image for id {image_id} under {root}")


def checked_records(captions, images_dir):
    if len(captions) != 1000:
        raise ValueError(f"Expected exactly 1000 image ids, found {len(captions)}")
    records = []
    for key, value in captions.items():
        image_id = int(key)
        if value.get("image_id") != image_id:
            raise ValueError(f"Mismatched image_id for key {key}")
        for field in ("raw_captions", "rich_captions"):
            texts = value.get(field)
            if not isinstance(texts, list) or len(texts) < 5 or not all(isinstance(x, str) and x.strip() for x in texts[:5]):
                raise ValueError(f"{key}: {field} needs five nonempty strings")
        records.append((image_id, resolve_image(images_dir, image_id), value))
    return sorted(records)


def init(args):
    root = Path(args.work).resolve()
    records = checked_records(load(args.captions), Path(args.images).resolve())
    stage = root / "images" / "train2014"
    stage.mkdir(parents=True, exist_ok=True)
    images = []
    for image_id, source, _ in records:
        name = f"COCO_train2014_{image_id:012d}.jpg"
        target = stage / name
        if not target.exists():
            # ponytail: links avoid copying the 1000 source images; extraction reads them only.
            target.symlink_to(source)
        images.append({"id": image_id, "file_name": name})
    save({"images": images, "annotations": []}, root / "base.pth")
    print(f"Prepared {len(images)} image links in {stage}")


def assemble(args):
    root = Path(args.work).resolve()
    captions = load(args.captions)
    base = torch.load(root / "dino.pth", map_location="cpu", weights_only=False)
    ids = [int(x["id"]) for x in base["images"]]
    if len(ids) != 1000 or set(ids) != {int(x) for x in captions}:
        raise ValueError("DINO features do not match the 1000 caption image ids")
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    val_ids = set(ids[:args.val_images])
    (root / "val_ids.json").write_text(json.dumps(sorted(val_ids), indent=2), encoding="utf-8")
    for arm, field in (("raw", "raw_captions"), ("rich", "rich_captions")):
        for split, selected in (("train", set(ids) - val_ids), ("val", val_ids)):
            images = [x for x in base["images"] if x["id"] in selected]
            annotations = []
            for image in images:
                image_id = image["id"]
                for index, caption in enumerate(captions[str(image_id)][field][:5]):
                    annotations.append({"id": image_id * 10 + index, "image_id": image_id, "caption": caption.strip()})
            save({"images": images, "annotations": annotations}, root / f"{arm}_{split}.pth")
            print(f"{arm}_{split}: {len(images)} images, {len(annotations)} captions")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "assemble"):
        p = sub.add_parser(command)
        p.add_argument("--captions", required=True)
        p.add_argument("--work", required=True)
        if command == "init":
            p.add_argument("--images", required=True)
        else:
            p.add_argument("--seed", type=int, default=123)
            p.add_argument("--val-images", type=int, default=100)
    args = parser.parse_args()
    if args.command == "init":
        init(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
