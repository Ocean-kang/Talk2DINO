"""Train one Talk2DINO projection arm with distinct images per contrastive batch."""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import ProjectionLayer


def read_data(path, feature):
    data = torch.load(path, map_location="cpu", weights_only=False)
    images = {x["id"]: x[feature].float() for x in data["images"]}
    captions = defaultdict(list)
    for ann in data["annotations"]:
        captions[ann["image_id"]].append(ann["ann_feats"].float())
    if not images or set(images) != set(captions) or any(len(x) != 5 for x in captions.values()):
        raise ValueError(f"Bad image/caption feature pairing in {path}")
    return images, captions


def batch_loss(model, image_feats, text_feats):
    logits = model(image_feats, text_feats) / 0.07
    labels = torch.arange(
        len(image_feats),
        device=image_feats.device
    )
    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) / 2


@torch.no_grad()
def evaluate(model, images, captions, device):
    model.eval()
    ids = sorted(images)
    image = F.normalize(torch.stack([images[i] for i in ids]).to(device), dim=-1)
    text = torch.stack([caption for i in ids for caption in captions[i]]).to(device)
    text = F.normalize(model.project_clip_txt(text), dim=-1)
    sims = text @ image.T
    truth = torch.arange(len(ids), device=device).repeat_interleave(5)
    ranks = sims.argsort(dim=1, descending=True)
    result = {f"t2i_R@{k}": (ranks[:, :k] == truth[:, None]).any(dim=1).float().mean().item() for k in (1, 5)}
    result["val_loss"] = sum(batch_loss(model, image[start:start + 64],
                                        torch.stack([captions[i][0] for i in ids[start:start + 64]]).to(device)).item()
                             for start in range(0, len(ids), 64)) / ((len(ids) + 63) // 64)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--arm", choices=("raw", "rich"), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training recipe")
    device = "cuda"
    root = Path(args.work).resolve()
    train_images, train_captions = read_data(root / f"{args.arm}_train.pth", "disentangled_self_attn")
    val_images, val_captions = read_data(root / f"{args.arm}_val.pth", "avg_self_attn_out")
    if set(train_images) & set(val_images):
        raise ValueError("Train/validation image overlap")
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/vitb_mlp_infonce.yaml").read_text())
    model = ProjectionLayer.from_config(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = root / "results" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    best = float("inf")
    ids = sorted(train_images)
    positions = {image_id: index for index, image_id in enumerate(ids)}
    for epoch in range(args.epochs):
        model.train()
        order = ids.copy()
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            if len(batch_ids) < 2:
                continue
            # ponytail: rotate five captions across epochs; unique image ids avoid false negatives.
            image = torch.stack([train_images[i] for i in batch_ids]).to(device)
            text = torch.stack([train_captions[i][(epoch + positions[i]) % 5] for i in batch_ids]).to(device)
            loss = batch_loss(model, image, text)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, val_images, val_captions, device)
        row = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **metrics}
        rows.append(row)
        print(args.arm, row, flush=True)
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            torch.save(model.state_dict(), output / "best.pth")
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output / 'best.pth'} and {output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
