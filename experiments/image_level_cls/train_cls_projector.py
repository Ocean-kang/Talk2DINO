#!/usr/bin/env python3
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.model import ProjectionLayer


class CocoImageLevelDataset(Dataset):
    """
    One dataset item = one unique image.
    Train: randomly choose one of that image's COCO captions.
    Val: use the first caption deterministically.
    """
    def __init__(self, pth_path, random_caption):
        data = torch.load(pth_path, map_location="cpu")

        image_features = {
            image["id"]: image["dino_features"]
            for image in data["images"]
            if "dino_features" in image
        }

        captions = defaultdict(list)
        for ann in data["annotations"]:
            if "ann_feats" in ann:
                captions[ann["image_id"]].append(ann["ann_feats"])

        self.samples = []
        for image_id, image_feature in image_features.items():
            if image_id in captions:
                self.samples.append(
                    (image_id, image_feature, captions[image_id])
                )

        if not self.samples:
            raise RuntimeError(
                f"No valid image/text pairs found in {pth_path}. "
                "Need images[*].dino_features and annotations[*].ann_feats."
            )

        self.random_caption = random_caption

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_id, image_feature, captions = self.samples[index]
        text_feature = random.choice(captions) if self.random_caption else captions[0]

        return {
            "image": image_feature,
            "text": text_feature,
            "image_id": image_id,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def info_nce(scores, temperature):
    logits = scores / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    ) / 2


def run_epoch(model, loader, optimizer, temperature, device, training):
    model.train(training)
    total_loss = 0.0
    total_items = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            texts = batch["text"].to(device, dtype=torch.float32, non_blocking=True)

            # ProjectionLayer:
            # CLIP-T [B,512] -> projector -> DINO [B,768]
            # visual input is already DINO CLS [B,768].
            scores = model(images, texts)
            loss = info_nce(scores, temperature)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / total_items


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    seed = train_cfg.get("seed", 123)
    set_seed(seed)

    train_set = CocoImageLevelDataset(args.train, random_caption=True)
    val_set = CocoImageLevelDataset(args.val, random_caption=False)

    print(f"Train images: {len(train_set)}")
    print(f"Val images:   {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=False,
    )

    model = ProjectionLayer.from_config(cfg["model"]).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    temperature = train_cfg.get("temperature", 0.07)
    epochs = train_cfg["epochs"]

    best_val = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, temperature, args.device, training=True
        )
        val_loss = run_epoch(
            model, val_loader, None, temperature, args.device, training=False
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output)
            print(f"  saved best -> {output}")

    print(f"Done. Best val loss: {best_val:.6f}")
    print(f"Weights: {output}")


if __name__ == "__main__":
    main()
