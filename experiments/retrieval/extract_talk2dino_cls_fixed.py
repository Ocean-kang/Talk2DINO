import os
import sys
import argparse

sys.path.append("/home/master/code/oymk/Talking2dino/Talk2DINO")

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from src.model import ProjectionLayer
from coco_dataset import COCORetrievalDataset, retrieval_collate_fn


def load_projector(config_path, weight_path, device):
    projector = ProjectionLayer.from_config(config_path)

    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    # Optional DDP prefix cleanup.
    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt):
        ckpt = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in ckpt.items()
        }

    # For evaluation, fail loudly if the checkpoint does not exactly match.
    projector.load_state_dict(ckpt, strict=True)
    projector.to(device).eval()
    return projector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="data/coco2014/val2014")
    parser.add_argument(
        "--annotations",
        default="data/coco2014/annotations/captions_val2014.json",
    )
    parser.add_argument(
        "--projector-config",
        default="experiments/image_level_cls/vitb_cls_infonce.yaml",
    )
    parser.add_argument(
        "--weights",
        default="weights/vitb_mlp_infonce_cls_imagelevel.pth",
    )
    parser.add_argument(
        "--output",
        default="outputs/retrieval/outputs_talk2dino_cls_retrieval.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device

    # IMPORTANT:
    # Use the same DINO backbone as the image-level projector training.
    # This is the standard Talk2DINO ViT-B backbone.
    dino = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14_reg",
    ).to(device).eval()

    clip_model, _ = clip.load(
        "ViT-B/16",
        device=device,
        jit=False,
    )
    clip_model.eval()

    projector = load_projector(
        args.projector_config,
        args.weights,
        device,
    )

    # Match Talk2DINO ViT-B COCO feature extraction.
    transform = T.Compose([
        T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = COCORetrievalDataset(
        args.images_dir,
        args.annotations,
        transform,
    )

    # The retrieval dataset should contain one item per UNIQUE image,
    # with all captions grouped under that item.
    print("dataset size:", len(dataset))
    first = dataset[0]
    print("first image id:", first["image_id"])
    print("captions for first image:", len(first["captions"]))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=retrieval_collate_fn,
        pin_memory=True,
    )

    image_features = []
    text_features = []
    image_ids = []
    caption_to_image = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract retrieval features"):
            images = batch["image"].to(device, non_blocking=True)

            dino_out = dino.forward_features(images)
            img = dino_out["x_norm_clstoken"].float()
            img = F.normalize(img, p=2, dim=-1)

            image_features.append(img.cpu())
            image_ids.extend(int(x) for x in batch["image_id"])

            texts = []
            ids = []

            for i, captions in enumerate(batch["captions"]):
                texts.extend(captions)
                ids.extend(
                    [int(batch["image_id"][i])] * len(captions)
                )

            tokens = clip.tokenize(texts).to(device)
            txt = clip_model.encode_text(tokens)
            txt = projector.project_clip_txt(txt)
            txt = F.normalize(txt.float(), p=2, dim=-1)

            text_features.append(txt.cpu())
            caption_to_image.extend(ids)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(
            "Duplicate image_ids detected. COCORetrievalDataset must return "
            "one sample per unique image, not one sample per caption."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    torch.save(
        {
            "image_features": image_features,
            "text_features": text_features,
            "image_ids": image_ids,
            "caption_to_image": caption_to_image,
            "metadata": {
                "dino_model": "dinov2_vitb14_reg",
                "image_size": 448,
                "clip_model": "ViT-B/16",
                "projector_config": args.projector_config,
                "weights": args.weights,
                "annotations": args.annotations,
            },
        },
        args.output,
    )

    print("saved:", args.output)
    print("image_features:", tuple(image_features.shape))
    print("text_features:", tuple(text_features.shape))
    print("num images:", len(image_ids))
    print("num captions:", len(caption_to_image))


if __name__ == "__main__":
    main()
