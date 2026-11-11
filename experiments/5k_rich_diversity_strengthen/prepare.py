"""Prepare and verify COCO-style feature files for the original Talk2DINO trainer."""

import argparse
import hashlib
import json
import random
from pathlib import Path

ARMS = {
    "raw": "raw_captions",
    "rich": "rich_captions",
    "rich_weaken": "rich_weaken_captions",
    "rich_diversity": "rich_diversity_captions",
    "rich_diversity_strengthen": "rich_diversity_strengthen_captions",
}
NEW_ARM = "rich_diversity_strengthen"
SEED = 123
VAL_COUNT = 500


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_captions(path, fields):
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of image records")
    captions = {}
    for row in rows:
        image_id = row["image_id"]
        if type(image_id) is not int or image_id < 1 or image_id in captions:
            raise ValueError(f"{path}: invalid or duplicate image_id {image_id!r}")
        for field in fields:
            texts = row.get(field)
            if not isinstance(texts, list) or len(texts) != 5 or not all(
                isinstance(text, str) and text.strip() for text in texts
            ):
                raise ValueError(f"{path}: image {image_id}, {field} needs five nonempty captions")
        captions[image_id] = row
    if len(captions) != 5000:
        raise ValueError(f"{path}: expected 5000 unique images, got {len(captions)}")
    return captions


def split_ids(captions):
    shuffled = sorted(captions)
    random.Random(SEED).shuffle(shuffled)
    val = sorted(shuffled[:VAL_COUNT])
    return {"train": sorted(captions.keys() - set(val)), "val": val}


def resolve_image(root, image_id):
    for directory in (root, root / "images"):
        for split in ("train2014", "val2014"):
            name = f"COCO_{split}_{image_id:012d}.jpg"
            for candidate in (directory / name, directory / split / name):
                if candidate.is_file():
                    return candidate.resolve()
    raise FileNotFoundError(f"COCO image {image_id} not found under {root}")


def image_paths(root, captions):
    return {image_id: resolve_image(root, image_id) for image_id in sorted(captions)}


def check(args, captions, ids):
    print(f"[check] {len(captions)} images, 5 captions/image in each of {len(ARMS)} fields")
    print(f"[check] split seed {SEED}: {len(ids['train'])} train / {len(ids['val'])} val")
    if args.images:
        print(f"[check] found {len(image_paths(args.images, captions))} source image files")
    if args.previous_captions:
        old_fields = [field for arm, field in ARMS.items() if arm != NEW_ARM]
        previous = load_captions(args.previous_captions, old_fields)
        if previous.keys() != captions.keys():
            raise ValueError("Previous caption JSON has different image IDs")
        for image_id in captions:
            for field in old_fields:
                if previous[image_id][field] != captions[image_id][field]:
                    raise ValueError(f"Previous captions differ at image {image_id}, field {field}")
        print("[check] previous image IDs and all four control caption fields match")
    if args.previous_val_ids:
        if sorted(read_json(args.previous_val_ids)) != ids["val"]:
            raise ValueError("Previous val_ids.json does not match this split")
        print("[check] previous validation IDs match")


def init(args, captions, torch):
    paths = image_paths(args.images, captions)
    manifest = {}
    for image_id, source in paths.items():
        stat = source.stat()
        manifest[str(image_id)] = [str(source), stat.st_size, stat.st_mtime_ns]
    manifest_path = args.work / "image_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Source images changed. Use a fresh WORK_DIR for a new DINO cache")
    if (args.work / "dino.pth").exists() and not manifest_path.exists():
        raise ValueError("Local DINO cache has no image manifest. Use DINO_SOURCE or a fresh WORK_DIR")

    stage = args.work / "images" / "train2014"
    stage.mkdir(parents=True, exist_ok=True)
    images = []
    for image_id, source in paths.items():
        name = f"COCO_train2014_{image_id:012d}.jpg"
        target = stage / name
        if target.is_symlink() or target.exists():
            if not target.is_file() or target.resolve() != source:
                raise ValueError(f"Conflicting staged file {target}. Use a fresh WORK_DIR")
        else:
            target.symlink_to(source)
        images.append({"id": image_id, "file_name": name})
    torch.save({"images": images, "annotations": []}, args.work / "base.pth")
    write_json(manifest_path, manifest)


def load_features(torch, path):
    # The feature files are produced by this experiment or supplied by the operator.
    return torch.load(path, map_location="cpu", weights_only=False)


def check_visual(torch, images, expected_ids, path):
    actual_ids = [image["id"] for image in images]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError(f"{path}: DINO image IDs do not match the expected IDs")
    for image in images:
        for name, shape in (("disentangled_self_attn", (12, 768)), ("avg_self_attn_out", (768,))):
            value = image.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"{path}: image {image['id']} needs {name} of shape {shape}")


def annotations(captions, selected, arm):
    for image_id in selected:
        for index, text in enumerate(captions[image_id][ARMS[arm]]):
            yield (image_id * 10 + index, image_id, text.strip())


def assemble(args, captions, ids, torch):
    source = load_features(torch, args.dino)
    check_visual(torch, source["images"], captions, args.dino)
    by_id = {image["id"]: image for image in source["images"]}
    args.work.mkdir(parents=True, exist_ok=True)
    write_json(args.work / "val_ids.json", ids["val"])
    for arm in args.arms:
        for split, selected in ids.items():
            path = args.work / f"{arm}_{split}.input.pth"
            torch.save({
                "images": [by_id[image_id] for image_id in selected],
                "annotations": [
                    {"id": ann_id, "image_id": image_id, "caption": text}
                    for ann_id, image_id, text in annotations(captions, selected, arm)
                ],
            }, path)
            print(f"[assemble] {path}: {len(selected)} images, {len(selected) * 5} captions")


def audit(args, captions, ids, torch):
    if read_json(args.work / "val_ids.json") != ids["val"]:
        raise ValueError("Saved validation IDs differ from the expected split")
    for arm in args.arms:
        for split, selected in ids.items():
            path = args.work / f"{arm}_{split}.pth"
            data = load_features(torch, path)
            check_visual(torch, data["images"], selected, path)
            expected = list(annotations(captions, selected, arm))
            actual = [(ann["id"], ann["image_id"], ann["caption"]) for ann in data["annotations"]]
            if actual != expected:
                raise ValueError(f"{path}: caption order, IDs or text differ from the source JSON")
            for ann in data["annotations"]:
                value = ann.get("ann_feats")
                if not torch.is_tensor(value) or tuple(value.shape) != (512,):
                    raise ValueError(f"{path}: annotation {ann['id']} needs a 512-D CLIP feature")
    write_json(args.work / "audit_report.json", {
        "status": "ok",
        "caption_json": str(args.captions.resolve()),
        "caption_sha256": hashlib.sha256(args.captions.read_bytes()).hexdigest(),
        "dino_source": str(args.dino.resolve()),
        "split_seed": SEED,
        "train_images": len(ids["train"]),
        "val_images": len(ids["val"]),
        "arms": args.arms,
    })
    print(f"[audit] PASS: {' '.join(args.arms)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "init", "assemble", "audit"):
        sub = commands.add_parser(command)
        sub.add_argument("--captions", type=Path, required=True)
        if command == "check":
            sub.add_argument("--images", type=Path)
            sub.add_argument("--previous-captions", type=Path)
            sub.add_argument("--previous-val-ids", type=Path)
        else:
            sub.add_argument("--work", type=Path, required=True)
        if command == "init":
            sub.add_argument("--images", type=Path, required=True)
        if command in ("assemble", "audit"):
            sub.add_argument("--dino", type=Path, required=True)
            sub.add_argument("--arms", nargs="+", choices=ARMS, default=[NEW_ARM])
    args = parser.parse_args()
    captions = load_captions(args.captions, ARMS.values())
    ids = split_ids(captions)
    if args.command == "check":
        check(args, captions, ids)
        return
    import torch

    if args.command == "init":
        init(args, captions, torch)
    elif args.command == "assemble":
        assemble(args, captions, ids, torch)
    else:
        audit(args, captions, ids, torch)


if __name__ == "__main__":
    main()
