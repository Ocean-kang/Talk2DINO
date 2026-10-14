"""Validate that the original 5K Oracle features are ready for bootstrap training.

Runs on existing .pth feature files; never rebuilds or edits those files.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def die(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def check_file(path: Path, visual_feature: str, expected_images: int):
    if not path.is_file():
        die(f"Missing feature file: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "images" not in obj or "annotations" not in obj:
        die(f"Invalid .pth structure: {path}")
    images = obj["images"]
    anns = obj["annotations"]
    image_ids = [int(img["id"]) for img in images]
    if len(images) != expected_images or len(set(image_ids)) != expected_images:
        die(f"{path}: expected {expected_images} distinct images; got {len(set(image_ids))}")
    for img in images:
        if visual_feature not in img:
            die(f"{path}: missing image feature '{visual_feature}' at ID {img['id']}")
        if not isinstance(img[visual_feature], torch.Tensor):
            die(f"{path}: image {img['id']} feature must be a tensor")
    if len(anns) != expected_images * 5:
        die(f"{path}: expected {expected_images * 5} annotations; got {len(anns)}")
    counts = collections.Counter()
    annotation_ids = set()
    for ann in anns:
        image_id = int(ann["image_id"])
        counts[image_id] += 1
        ann_id = int(ann["id"])
        if ann_id in annotation_ids:
            die(f"{path}: duplicate annotation ID {ann_id}")
        annotation_ids.add(ann_id)
        if not isinstance(ann.get("ann_feats"), torch.Tensor):
            die(f"{path}: missing CLIP tensor ann_feats for image ID {image_id}")
    if set(counts) != set(image_ids) or any(n != 5 for n in counts.values()):
        die(f"{path}: each image must have exactly 5 corresponding captions")
    print(f"[OK] {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: "
          f"images={len(images)}, captions={len(anns)}, visual={visual_feature}")
    return set(image_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="experiments/raw_rich_5k/work")
    args = parser.parse_args()
    work = resolve(args.work)
    cfg = resolve("configs/vitb_mlp_infonce.yaml")
    teacher = resolve("weights/vitb_mlp_infonce.pth")
    trainer = resolve("experiments/teacher_guided_bootstrap/train_bootstrap.py")
    for filename in (cfg, teacher, trainer,
                     resolve("experiments/teacher_guided_bootstrap/bootstrap_model.py")):
        if not filename.is_file():
            die(f"Required file missing: {filename}")
    config = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    tr = config["train"]
    expected = {"num_epochs": 100, "batch_size": 128, "lr": 0.0001, "ltype": "infonce", "save_best_model": False}
    for key, value in expected.items():
        if tr.get(key) != value:
            die(f"Expected original config train.{key}={value!r}, found {tr.get(key)!r}")
    print("[OK] Original Talk2DINO settings: 100 epochs, batch=128, Adam, lr=1e-4, InfoNCE; last checkpoint")
    split_ids = {}
    for split, expected_count in (("train", 4500), ("val", 500)):
        visual = "disentangled_self_attn" if split == "train" else "avg_self_attn_out"
        raw = check_file(work / f"raw_{split}.pth", visual, expected_count)
        rich = check_file(work / f"rich_{split}.pth", visual, expected_count)
        if raw != rich:
            die(f"Raw/Rich {split} image IDs differ")
        split_ids[split] = raw
    if split_ids["train"] & split_ids["val"]:
        die("Train/validation image IDs overlap")
    val_ids_file = work / "val_ids.json"
    if val_ids_file.is_file():
        val_ids = set(map(int, json.loads(val_ids_file.read_text(encoding="utf-8"))))
        if val_ids != split_ids["val"]:
            die("val_ids.json does not match validation image IDs")
        print(f"[OK] val_ids.json confirms original 5K Oracle split")
    print("[PASS] Ready for four-arm training with Full-COCO Teacher")


if __name__ == "__main__":
    main()
