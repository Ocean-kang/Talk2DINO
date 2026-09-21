#!/usr/bin/env python3
import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Existing COCO .pth with annotations + ann_feats")
    p.add_argument("--images-dir", required=True, help="train2014 or val2014 image directory")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    data = torch.load(args.input, map_location="cpu")

    if not data["annotations"] or "ann_feats" not in data["annotations"][0]:
        raise KeyError(
            "ann_feats not found. Run text_features_extraction.py on the input .pth first."
        )

    transform = T.Compose([
        T.Resize(args.image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    out_images = []
    images = data["images"]
    has_cls = all("dino_features" in image for image in images)

    if has_cls:
        print("Existing dino_features found; reusing CLS features.")
        for image in images:
            out_images.append({
                "id": image["id"],
                "file_name": image["file_name"],
                "dino_features": image["dino_features"],
            })
    else:
        print("dino_features not found; extracting DINOv2 CLS features.")
        model = torch.hub.load("facebookresearch/dinov2", args.model)
        model = model.eval().to(args.device)

        with torch.no_grad():
            for start in tqdm(range(0, len(images), args.batch_size), desc="DINO CLS"):
                batch_meta = images[start:start + args.batch_size]
                batch = []

                for image in batch_meta:
                    image_path = os.path.join(args.images_dir, image["file_name"])
                    if not os.path.isfile(image_path):
                        raise FileNotFoundError(image_path)

                    with Image.open(image_path) as img:
                        batch.append(transform(img.convert("RGB")))

                batch = torch.stack(batch).to(args.device)
                cls = model(batch, is_training=True)["x_norm_clstoken"].cpu()

                for image, feature in zip(batch_meta, cls):
                    out_images.append({
                        "id": image["id"],
                        "file_name": image["file_name"],
                        "dino_features": feature,
                    })

    out_annotations = []
    for ann in data["annotations"]:
        out_annotations.append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "caption": ann["caption"],
            "ann_feats": ann["ann_feats"],
        })

    output = {
        "images": out_images,
        "annotations": out_annotations,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(output, args.output)
    print(f"Saved: {args.output}")
    print(f"Images: {len(out_images)}")
    print(f"Captions: {len(out_annotations)}")
    print(f"DINO dim: {out_images[0]['dino_features'].numel()}")
    print(f"CLIP text dim: {out_annotations[0]['ann_feats'].numel()}")


if __name__ == "__main__":
    main()
