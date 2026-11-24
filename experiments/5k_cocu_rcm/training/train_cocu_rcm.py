#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import write_json, sha256_file, torch_load
from models.rcm_loss import multi_positive_infonce
from training.rcm_dataset import RCMTrainingDataset
from src.dataset import DinoClipDataset
from src.loss import ContrastiveLoss
from src.train_util import set_seed, const_lr, cosine_lr


def build_model(config):
    class_name = config["model"].get("model_class", "ProjectionLayer")
    model_class = getattr(importlib.import_module("src.model"), class_name)
    return model_class.from_config(config["model"])


def load_or_create_init(config, config_path, seed, init_dir, device):
    init_dir = Path(init_dir)
    init_dir.mkdir(parents=True, exist_ok=True)
    init_path = init_dir / f"projector_seed{seed}.pth"
    manifest_path = init_dir / f"projector_seed{seed}.json"
    config_hash = sha256_file(config_path)
    if init_path.exists():
        if not manifest_path.exists():
            raise RuntimeError(f"Missing initialization manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_hash:
            raise RuntimeError("Config changed since initialization checkpoint was created; use a fresh init directory.")
        model = build_model(config)
        model.load_state_dict(torch_load(init_path), strict=True)
        return model.to(device), init_path
    model = build_model(config).to(device)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, init_path)
    write_json(manifest_path, {
        "seed": seed, "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_hash,
        "note": "Shared initialization for all arms with this seed."
    })
    return model, init_path


def run_validation(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            annotations = batch["annotation"].to(device, dtype=torch.float32)
            loss = criterion(images, annotations, return_similarity_mat=False)
            losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))


def main():
    p = argparse.ArgumentParser(description="Standalone Talk2DINO training with optional CoCu-RCM local loss.")
    p.add_argument("--arm", required=True, choices=["raw", "cocu", "cocu_same", "cocu_rcm", "cocu_rcm_shuffled"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--base-train", required=True)
    p.add_argument("--base-val", required=True)
    p.add_argument("--augmented-train", required=True)
    p.add_argument("--sampling-index", default="")
    p.add_argument("--region-features", default="")
    p.add_argument("--region-index", default="")
    p.add_argument("--config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument("--output-dir", default="outputs/5k_cocu_rcm")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--lambda-local", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--grad-diagnostic", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warning] CUDA unavailable; running on CPU.")
    cfg_path = Path(a.config)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    train_cfg = config["train"]
    batch_size = a.batch_size or int(train_cfg["batch_size"])
    epochs = a.epochs or int(train_cfg["num_epochs"])
    lr = float(train_cfg["lr"])
    local_arm = a.arm in ("cocu_same", "cocu_rcm", "cocu_rcm_shuffled")
    train_path = a.base_train if a.arm == "raw" else a.augmented_train
    if not Path(train_path).is_file() or not Path(a.base_val).is_file():
        raise FileNotFoundError("Train/validation PTH missing; run preprocessing first.")
    if local_arm:
        for path in (a.sampling_index, a.region_features, a.region_index):
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"Missing RCM input: {path}")

    set_seed(a.seed)
    model, init_path = load_or_create_init(
        config, cfg_path, a.seed, Path(a.output_dir) / "initializations", device
    )
    # Original do_train() resets the RNG after model construction, before loader shuffling.
    set_seed(a.seed)

    if local_arm:
        train_dataset = RCMTrainingDataset(
            train_path, a.sampling_index, a.region_features, a.region_index
        )
    else:
        train_dataset = DinoClipDataset(
            train_path, features_name="disentangled_self_attn", text_features="ann_feats"
        )
    # Preserve original train.py validation behavior for disentangled_self_attn.
    val_dataset = DinoClipDataset(
        a.base_val, features_name="avg_self_attn_out", text_features="ann_feats"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=a.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=a.num_workers, pin_memory=(device.type == "cuda")
    )

    criterion = ContrastiveLoss(
        model, margin=train_cfg.get("margin", 0.2),
        max_violation=train_cfg.get("max_violation", True),
        ltype=train_cfg.get("ltype", "infonce")
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup = int(train_cfg.get("warmup", 0))
    scheduler_name = "linear"
    if scheduler_name == "linear" and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == "cosine":
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = None

    weight_tag = a.tag or f"5k_cocu_rcm_{a.arm}_seed{a.seed}"
    weight_name = f"{Path(cfg_path).stem}_{weight_tag}"
    weight_path = Path(a.weights_dir) / f"{weight_name}.pth"
    if weight_path.exists() and not a.overwrite and not a.no_save:
        raise FileExistsError(f"Refusing to overwrite {weight_path}; pass --overwrite.")
    log_dir = Path(a.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{a.arm}_seed{a.seed}{'_smoke' if a.no_save else ''}.csv"

    columns = [
        "epoch", "train_global_loss", "train_local_raw", "train_local_scaled",
        "weighted_local_loss", "train_total_loss", "val_global_loss",
        "local_anchor_count", "positive_count", "negative_count",
        "global_grad_norm", "local_grad_norm",
    ]
    rows, global_step = [], 0
    for epoch in range(epochs):
        model.train()
        global_values, local_values, scaled_values, weighted_values, total_values = [], [], [], [], []
        local_anchor_count = positive_count = negative_count = 0
        grad_global = grad_local = ""
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"{a.arm} epoch {epoch + 1}")):
            images = batch["image"].to(device, dtype=torch.float32)
            annotations = batch["annotation"].to(device, dtype=torch.float32)
            if scheduler is not None:
                scheduler(batch_idx + epoch * len(train_loader))
            global_loss = criterion(images, annotations, return_similarity_mat=False)

            local_raw = global_loss.detach() * 0.0
            n_anchors = n_pos = n_neg = 0
            if local_arm:
                pos = batch["local_positive_features"].to(device)
                neg = batch["local_negative_features"].to(device)
                pos_mask = batch["local_positive_mask"].to(device)
                neg_mask = batch["local_negative_mask"].to(device)
                active = pos_mask.any(dim=1) & neg_mask.any(dim=1)
                if active.any():
                    projected = model.project_clip_txt(annotations[active])
                    local_raw, n_anchors, n_pos, n_neg = multi_positive_infonce(
                        projected, pos[active], neg[active],
                        pos_mask[active], neg_mask[active], temperature=a.temperature
                    )
            # The original ContrastiveLoss divides its bidirectional CE by B^2.
            local_scaled = local_raw / float(images.shape[0] ** 2)
            weighted_local = a.lambda_local * local_scaled
            total_loss = global_loss + weighted_local

            if a.grad_diagnostic and global_step == 0 and n_anchors > 0:
                params = [x for x in model.parameters() if x.requires_grad]
                gg = torch.autograd.grad(global_loss, params, retain_graph=True, allow_unused=True)
                gl = torch.autograd.grad(weighted_local, params, retain_graph=True, allow_unused=True)
                grad_global = sum(float(x.detach().pow(2).sum()) for x in gg if x is not None) ** 0.5
                grad_local = sum(float(x.detach().pow(2).sum()) for x in gl if x is not None) ** 0.5

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_values.append(float(global_loss.detach()))
            local_values.append(float(local_raw.detach()))
            scaled_values.append(float(local_scaled.detach()))
            weighted_values.append(float(weighted_local.detach()))
            total_values.append(float(total_loss.detach()))
            local_anchor_count += n_anchors
            positive_count += n_pos
            negative_count += n_neg
            global_step += 1
            if a.max_steps and global_step >= a.max_steps:
                break

        val_loss = run_validation(model, val_loader, criterion, device)
        row = {
            "epoch": epoch + 1,
            "train_global_loss": sum(global_values) / max(1, len(global_values)),
            "train_local_raw": sum(local_values) / max(1, len(local_values)),
            "train_local_scaled": sum(scaled_values) / max(1, len(scaled_values)),
            "weighted_local_loss": sum(weighted_values) / max(1, len(weighted_values)),
            "train_total_loss": sum(total_values) / max(1, len(total_values)),
            "val_global_loss": val_loss,
            "local_anchor_count": local_anchor_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "global_grad_norm": grad_global,
            "local_grad_norm": grad_local,
        }
        rows.append(row)
        print(
            f"Epoch {epoch + 1}: train_global={row['train_global_loss']:.8f} "
            f"local_raw={row['train_local_raw']:.6f} "
            f"local_scaled={row['train_local_scaled']:.8f} "
            f"weighted_local={row['weighted_local_loss']:.8f} "
            f"total={row['train_total_loss']:.8f} val={val_loss:.8f} "
            f"anchors={local_anchor_count}"
        )
        if a.max_steps and global_step >= a.max_steps:
            break

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    if not a.no_save:
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weight_path)
        write_json(log_dir / f"train_{a.arm}_seed{a.seed}.json", {
            "arm": a.arm, "seed": a.seed, "weight_name": weight_name,
            "weight_path": str(weight_path), "init_checkpoint": str(init_path),
            "train_path": str(Path(train_path).resolve()),
            "train_sha256": sha256_file(train_path),
            "val_path": str(Path(a.base_val).resolve()),
            "val_sha256": sha256_file(a.base_val),
            "sampling_index_sha256": sha256_file(a.sampling_index) if local_arm else None,
            "region_features_sha256": sha256_file(a.region_features) if local_arm else None,
            "config": str(cfg_path.resolve()), "config_sha256": sha256_file(cfg_path),
            "lambda_local": a.lambda_local, "temperature": a.temperature,
            "epochs_completed": len(rows), "final": rows[-1] if rows else {},
        })
        print(f"[saved] {weight_path}")
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
