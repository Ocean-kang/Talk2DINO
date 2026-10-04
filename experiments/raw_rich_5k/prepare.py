"""Prepare 5k raw/rich Talk2DINO experiment."""
import argparse
import json
import random
from pathlib import Path

import torch


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected caption JSON to be a list")

    captions = {}

    for item in data:
        image_id = int(item["image_id"])

        if image_id in captions:
            raise ValueError(f"Duplicate image_id: {image_id}")

        for field in ("raw_captions", "rich_captions"):
            texts = item.get(field)

            if not isinstance(texts, list) or len(texts) != 5:
                raise ValueError(
                    f"{image_id}: {field} must contain exactly 5 captions"
                )

            if not all(isinstance(x, str) and x.strip() for x in texts):
                raise ValueError(
                    f"{image_id}: {field} contains empty caption"
                )

        captions[image_id] = item

    if len(captions) != 5000:
        raise ValueError(
            f"Expected 5000 images, found {len(captions)}"
        )

    return captions


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def resolve_image(root, image_id):
    names = [
        f"COCO_train2014_{image_id:012d}.jpg",
        f"COCO_val2014_{image_id:012d}.jpg",
    ]

    for directory in (
        root,
        root / "train2014",
        root / "val2014",
    ):
        for name in names:
            path = directory / name

            if path.is_file():
                return path

    raise FileNotFoundError(
        f"Missing image {image_id} under {root}"
    )


def init(args):
    captions = load(args.captions)

    work = Path(args.work).resolve()
    image_root = Path(args.images).resolve()

    stage = work / "images" / "train2014"
    stage.mkdir(parents=True, exist_ok=True)

    images = []

    for image_id in sorted(captions):
        source = resolve_image(image_root, image_id)

        name = f"COCO_train2014_{image_id:012d}.jpg"
        target = stage / name

        if not target.exists():
            target.symlink_to(source)

        images.append({
            "id": image_id,
            "file_name": name,
        })

    save(
        {
            "images": images,
            "annotations": [],
        },
        work / "base.pth",
    )

    print(f"Prepared {len(images)} images")
    print(f"Saved: {work / 'base.pth'}")


def assemble(args):
    captions = load(args.captions)
    work = Path(args.work).resolve()

    base = torch.load(
        work / "dino.pth",
        map_location="cpu",
        weights_only=False,
    )

    ids = [int(x["id"]) for x in base["images"]]

    if set(ids) != set(captions):
        raise ValueError(
            "DINO features and caption image IDs do not match"
        )

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    val_ids = set(ids[:args.val_images])
    train_ids = set(ids) - val_ids

    (work / "val_ids.json").write_text(
        json.dumps(sorted(val_ids), indent=2),
        encoding="utf-8",
    )

    for arm, field in (
        ("raw", "raw_captions"),
        ("rich", "rich_captions"),
    ):
        for split, selected in (
            ("train", train_ids),
            ("val", val_ids),
        ):
            images = [
                x
                for x in base["images"]
                if int(x["id"]) in selected
            ]

            annotations = []

            for image in images:
                image_id = int(image["id"])

                for index, caption in enumerate(
                    captions[image_id][field]
                ):
                    annotations.append({
                        "id": image_id * 10 + index,
                        "image_id": image_id,
                        "caption": caption.strip(),
                    })

            output = work / f"{arm}_{split}.pth"

            save(
                {
                    "images": images,
                    "annotations": annotations,
                },
                output,
            )

            print(
                f"{arm}_{split}: "
                f"{len(images)} images, "
                f"{len(annotations)} captions"
            )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    init_parser = sub.add_parser("init")
    init_parser.add_argument("--captions", required=True)
    init_parser.add_argument("--images", required=True)
    init_parser.add_argument("--work", required=True)

    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--captions", required=True)
    assemble_parser.add_argument("--work", required=True)
    assemble_parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    assemble_parser.add_argument(
        "--val-images",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    if args.command == "init":
        init(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()