#!/usr/bin/env python3
"""Build COCO feature inputs, assemble Raw/ECSD Talk2DINO files, and audit them."""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_records(path, field):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("{} must contain a JSON list".format(path))
    result = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in result:
            raise ValueError("Duplicate image_id {} in {}".format(image_id, path))
        captions = row.get(field)
        if not isinstance(captions, list) or len(captions) != 5:
            raise ValueError("image_id={} field={} must contain five captions".format(image_id, field))
        if not all(isinstance(x, str) and x.strip() for x in captions):
            raise ValueError("Invalid caption at image_id={} field={}".format(image_id, field))
        result[image_id] = [x.strip() for x in captions]
    return result


def build_base(args):
    raw = load_records(args.raw_captions, "raw_captions")
    ann_data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    image_meta = {int(x["id"]): x for x in ann_data["images"]}
    if set(raw) != set(image_meta):
        raise ValueError(
            "Raw/annotation image IDs differ. raw-only={} annotation-only={}".format(
                sorted(set(raw) - set(image_meta))[:10], sorted(set(image_meta) - set(raw))[:10]
            )
        )
    images = []
    for image_id in sorted(raw):
        file_name = image_meta[image_id].get("file_name")
        if not file_name or args.split not in file_name.lower():
            raise ValueError("Invalid filename for {} split image {}: {}".format(args.split, image_id, file_name))
        if args.image_root:
            image_path = Path(args.image_root) / (args.split + "2014") / file_name
            if not image_path.is_file():
                raise FileNotFoundError("Missing COCO image: {}".format(image_path))
        images.append({"id": image_id, "file_name": file_name})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "annotations": []}, out)
    print("[build-base] {} images -> {}".format(len(images), out))


def reuse_dino(args):
    base = torch_load(args.base)
    source = torch_load(args.source)
    base_images = base.get("images", [])
    source_images = source.get("images", [])
    base_by_id = {int(x["id"]): x for x in base_images}
    source_by_id = {int(x["id"]): x for x in source_images}
    if set(base_by_id) != set(source_by_id):
        raise ValueError(
            "Cannot reuse DINO features: ID mismatch. missing={} extra={}".format(
                sorted(set(base_by_id) - set(source_by_id))[:20],
                sorted(set(source_by_id) - set(base_by_id))[:20],
            )
        )
    merged = []
    for image_id in sorted(base_by_id):
        item = dict(source_by_id[image_id])
        item["id"] = image_id
        item["file_name"] = base_by_id[image_id]["file_name"]
        if args.required_feature not in item:
            raise ValueError("DINO source lacks required feature {!r} for image {}".format(args.required_feature, image_id))
        merged.append(item)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": merged, "annotations": []}, out)
    print("[reuse-dino] reused {} image features -> {}".format(len(merged), out))


def assemble(args):
    raw = load_records(args.raw, "raw_captions")
    ecsd = load_records(args.ecsd, "ecsd_captions")
    if set(raw) != set(ecsd):
        raise ValueError("Raw and ECSD image IDs differ")
    dino = torch_load(args.dino)
    image_by_id = {int(x["id"]): x for x in dino.get("images", [])}
    if set(raw) != set(image_by_id):
        raise ValueError(
            "Caption/DINO image IDs differ. captions-only={} DINO-only={}".format(
                sorted(set(raw) - set(image_by_id))[:20],
                sorted(set(image_by_id) - set(raw))[:20],
            )
        )
    required = "disentangled_self_attn" if args.split == "train" else "avg_self_attn_out"
    missing = [i for i, x in image_by_id.items() if required not in x]
    if missing:
        raise ValueError("DINO feature {} missing for image IDs {}".format(required, missing[:20]))

    field = "raw_captions" if args.arm == "raw" else "ecsd_captions"
    captions = raw if args.arm == "raw" else ecsd
    images = [image_by_id[i] for i in sorted(image_by_id)]
    annotations = []
    for image_id in sorted(captions):
        for index, caption in enumerate(captions[image_id]):
            annotations.append({
                "id": image_id * 10 + index,
                "image_id": image_id,
                "caption": caption,
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "annotations": annotations}, out)
    print("[assemble] {} {} images / {} captions -> {}".format(
        args.arm, len(images), len(annotations), out
    ))


def check_clip(args):
    data = torch_load(args.file)
    anns = data.get("annotations", [])
    missing = [i for i, ann in enumerate(anns) if not torch.is_tensor(ann.get("ann_feats"))]
    if missing:
        print("[check-clip] missing ann_feats: {} of {}".format(len(missing), len(anns)))
        raise SystemExit(1)
    print("[check-clip] PASS: {} ann_feats".format(len(anns)))


def audit(args):
    work = Path(args.work)
    split_info = {}
    global_ids = {}
    for split, expected in (("train", args.expected_train_images), ("val", args.expected_val_images)):
        for arm in ("raw", "ecsd"):
            path = work / "{}_{}.pth".format(arm, split)
            if not path.is_file():
                raise FileNotFoundError(path)
            data = torch_load(path)
            images = data.get("images", [])
            annotations = data.get("annotations", [])
            ids = [int(x["id"]) for x in images]
            id_set = set(ids)
            if len(ids) != len(id_set):
                raise ValueError("Duplicate image IDs in {}".format(path))
            if expected and len(ids) != expected:
                raise ValueError("{} has {} images, expected {}".format(path, len(ids), expected))
            required = "disentangled_self_attn" if split == "train" else "avg_self_attn_out"
            if any(required not in x for x in images):
                raise ValueError("{} missing DINO feature {}".format(path, required))
            counts = Counter(int(x["image_id"]) for x in annotations)
            if set(counts) != id_set or any(counts[i] != 5 for i in id_set):
                raise ValueError("{} does not have exactly five annotations per image".format(path))
            if not all(isinstance(x.get("caption"), str) and x["caption"].strip() for x in annotations):
                raise ValueError("{} contains invalid captions".format(path))
            if not all(torch.is_tensor(x.get("ann_feats")) for x in annotations):
                raise ValueError("{} missing CLIP ann_feats; run text_features_extraction.py".format(path))
            if {int(x["image_id"]) for x in annotations} != id_set:
                raise ValueError("{} annotation IDs do not match image IDs".format(path))
            split_info[(split, arm)] = {"num_images": len(ids), "num_annotations": len(annotations), "ids": id_set}
            print("[audit] {}_{} PASS".format(arm, split))
        if split_info[(split, "raw")]["ids"] != split_info[(split, "ecsd")]["ids"]:
            raise ValueError("Raw/ECSD {} IDs differ".format(split))
        global_ids[split] = split_info[(split, "raw")]["ids"]
    overlap = global_ids["train"] & global_ids["val"]
    if overlap:
        raise ValueError("Train/Val image ID overlap: {}".format(sorted(overlap)[:20]))

    report = {
        "status": "ok",
        "dataset": "MSCOCO2014",
        "arms": ["raw", "ecsd"],
        "train_images": len(global_ids["train"]),
        "val_images": len(global_ids["val"]),
        "captions_per_image": 5,
        "train_val_disjoint": True,
        "same_image_ids_raw_ecsd": True,
        "dino_features_present": True,
        "clip_ann_feats_present": True,
    }
    save_json(report, work / "audit_report.json")
    print("[audit] PASS -> {}".format(work / "audit_report.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-base")
    b.add_argument("--raw-captions", required=True)
    b.add_argument("--annotations", required=True)
    b.add_argument("--split", choices=["train", "val"], required=True)
    b.add_argument("--image-root", default=None, help="Optional COCO root used to verify image files")
    b.add_argument("--out", required=True)
    b.set_defaults(func=build_base)

    r = sub.add_parser("reuse-dino")
    r.add_argument("--base", required=True)
    r.add_argument("--source", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--required-feature", required=True)
    r.set_defaults(func=reuse_dino)

    a = sub.add_parser("assemble")
    a.add_argument("--raw", required=True)
    a.add_argument("--ecsd", required=True)
    a.add_argument("--dino", required=True)
    a.add_argument("--split", choices=["train", "val"], required=True)
    a.add_argument("--arm", choices=["raw", "ecsd"], required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=assemble)

    c = sub.add_parser("check-clip")
    c.add_argument("--file", required=True)
    c.set_defaults(func=check_clip)

    au = sub.add_parser("audit")
    au.add_argument("--work", required=True)
    au.add_argument("--expected-train-images", type=int, default=82783)
    au.add_argument("--expected-val-images", type=int, default=40504)
    au.set_defaults(func=audit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
