#!/usr/bin/env python3
"""Full MSCOCO14 CoCu-RCM training adapter around original Talk2DINO components."""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import atomic_json, atomic_torch_save, sha256_file, sha256_json, torch_load



def _load_original_5k_local_loss():
    original_path = ROOT / "experiments" / "5k_cocu_rcm" / "models" / "rcm_loss.py"
    if original_path.is_file():
        spec = importlib.util.spec_from_file_location("talk2dino_5k_rcm_loss", original_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.multi_positive_infonce
    from models.rcm_loss import multi_positive_infonce as fallback
    return fallback


multi_positive_infonce = _load_original_5k_local_loss()

from training.rcm_dataset import FullRCMTrainingDataset
from src.dataset import DinoClipDataset
from src.loss import ContrastiveLoss
from src.train_util import set_seed, const_lr, cosine_lr


ARMS = ("raw", "cocu", "cocu_same", "cocu_rcm", "cocu_rcm_shuffled")


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
            raise RuntimeError(
                "Projector config changed since initialization; use a fresh output directory."
            )
        if int(manifest.get("seed", -1)) != int(seed):
            raise RuntimeError("Initialization manifest seed does not match its requested seed")
        recorded_hash = manifest.get("checkpoint_sha256")
        if not recorded_hash or recorded_hash != sha256_file(init_path):
            raise RuntimeError(f"Initialization checkpoint hash mismatch: {init_path}")
        model = build_model(config)
        model.load_state_dict(torch_load(init_path), strict=True)
        return model.to(device), init_path
    model = build_model(config).to(device)
    atomic_torch_save(
        init_path,
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
    )
    atomic_json(manifest_path, {
        "seed": seed,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_hash,
        "checkpoint_sha256": sha256_file(init_path),
        "note": "The same initialization is reused by all five arms for this seed.",
    })
    return model, init_path


def grad_norm(grads):
    return sum(float(g.detach().float().pow(2).sum()) for g in grads if g is not None) ** 0.5


def run_validation(model, loader, criterion, device, max_batches=0):
    model.eval()
    values = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            images = batch["image"].to(device, dtype=torch.float32)
            annotations = batch["annotation"].to(device, dtype=torch.float32)
            loss = criterion(images, annotations, return_similarity_mat=False)
            values.append(float(loss.item()))
    return sum(values) / max(1, len(values))


def write_history_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def capture_rng():
    return {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    torch.set_rng_state(state["torch_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--base-train", required=True)
    p.add_argument("--base-val", required=True)
    p.add_argument("--augmented-train", required=True)
    p.add_argument("--sampling-dir", default="")
    p.add_argument("--region-feature-dir", default="")
    p.add_argument("--region-index", default="")
    p.add_argument("--config", default="configs/vitb_mlp_infonce.yaml")
    p.add_argument("--arm-config", default="")
    p.add_argument("--output-dir", default="outputs/full_mscoco14_cocu_rcm")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--lambda-local", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--restart", action="store_true", help="Discard only this arm/seed epoch-resume checkpoint")
    p.add_argument("--no-save", action="store_true", help="Smoke test mode; do not save checkpoints")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warning] CUDA unavailable; this run will be very slow.")
    cfg_path = Path(a.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    train_cfg = config["train"]
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["num_epochs"])
    lr = float(train_cfg["lr"])
    warmup = int(train_cfg.get("warmup", 0))
    lambda_local = 0.1 if a.arm in ("cocu_same", "cocu_rcm", "cocu_rcm_shuffled") else 0.0
    temperature = 0.07
    if a.arm_config:
        arm_cfg = yaml.safe_load(Path(a.arm_config).read_text(encoding="utf-8"))
        lambda_local = float(arm_cfg.get("training", {}).get("lambda_local", lambda_local))
        temperature = float(arm_cfg.get("training", {}).get("temperature", temperature))
    if a.lambda_local is not None:
        lambda_local = float(a.lambda_local)
    if a.temperature is not None:
        temperature = float(a.temperature)
    a.temperature = temperature

    local_arm = a.arm in ("cocu_same", "cocu_rcm", "cocu_rcm_shuffled")
    train_path = Path(a.base_train if a.arm == "raw" else a.augmented_train)
    val_path = Path(a.base_val)
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("Train/validation PTH missing; run full preprocessing first.")
    if local_arm:
        for value in (a.sampling_dir, a.region_feature_dir, a.region_index):
            if not value or not Path(value).exists():
                raise FileNotFoundError(f"Missing local supervision input: {value}")

    output_dir = Path(a.output_dir)
    init_dir = output_dir / "initializations"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / f"{a.arm}_seed{a.seed}.resume.pt"
    weight_tag = f"full_mscoco14_cocu_rcm_{a.arm}_seed{a.seed}"
    weight_name = f"{cfg_path.stem}_{weight_tag}"
    weight_path = Path(a.weights_dir) / f"{weight_name}.pth"
    if weight_path.exists() and not a.overwrite and not a.no_save:
        raise FileExistsError(f"Refusing to overwrite {weight_path}; pass --overwrite.")
    if a.restart and resume_path.exists():
        resume_path.unlink()

    set_seed(a.seed)
    model, init_path = load_or_create_init(config, cfg_path, a.seed, init_dir, device)
    # Keep the original Talk2DINO seed-reset point before DataLoader construction.
    set_seed(a.seed)

    if local_arm:
        expected_sampling_arm = {
            "cocu_same": "same",
            "cocu_rcm": "cross",
            "cocu_rcm_shuffled": "shuffled",
        }[a.arm]
        train_dataset = FullRCMTrainingDataset(
            train_path, a.sampling_dir, a.region_feature_dir, a.region_index,
            expected_arm=expected_sampling_arm,
        )
    else:
        train_dataset = DinoClipDataset(
            str(train_path), features_name="disentangled_self_attn", text_features="ann_feats"
        )
    val_dataset = DinoClipDataset(
        str(val_path), features_name="avg_self_attn_out", text_features="ann_feats"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=bool(train_cfg.get("shuffle", True)),
        num_workers=a.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=a.num_workers, pin_memory=(device.type == "cuda"),
    )

    criterion = ContrastiveLoss(
        model, margin=train_cfg.get("margin", 0.2),
        max_violation=train_cfg.get("max_violation", True),
        ltype=train_cfg.get("ltype", "infonce"),
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler_name = "linear"  # same scheduler choice as the 5K CoCu-RCM trainer
    if scheduler_name == "linear" and warmup > 0:
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    elif scheduler_name == "cosine":
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = None

    source_hashes = {
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "config_sha256": sha256_file(cfg_path),
        "init_checkpoint_sha256": sha256_file(init_path),
        "arm_config_sha256": sha256_file(a.arm_config) if a.arm_config else None,
        "settings_sha256": sha256_json({
            "arm": a.arm, "seed": a.seed, "lambda_local": lambda_local,
            "temperature": a.temperature, "batch_size": batch_size,
            "epochs": epochs, "lr": lr, "warmup": warmup,
            "num_workers": a.num_workers, "scheduler": scheduler_name,
            "max_steps": a.max_steps,
        }),
        "sampling_sha256": sha256_file(Path(a.sampling_dir) / "manifest.json") if local_arm else None,
        "region_index_sha256": sha256_file(a.region_index) if local_arm else None,
        "region_manifest_sha256": sha256_file(Path(a.region_feature_dir).parent / "region_gallery_manifest.json") if local_arm else None,
    }
    start_epoch, global_step, history = 0, 0, []
    if resume_path.is_file() and not a.no_save:
        state = torch_load(resume_path)
        if state.get("source_hashes") != source_hashes:
            raise RuntimeError("Resume checkpoint input hashes differ; use --restart for a clean run.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        history = list(state.get("history", []))
        restore_rng(state["rng"])
        print(f"[RESUME] {a.arm} seed={a.seed} from epoch={start_epoch}")

    columns = [
        "epoch", "train_global_loss", "train_local_raw", "train_local_scaled",
        "weighted_local_loss", "train_total_loss", "val_global_loss",
        "local_anchor_count", "positive_count", "negative_count",
        "global_grad_norm", "local_grad_norm", "local_global_grad_ratio",
    ]
    csv_path = log_dir / f"train_{a.arm}_seed{a.seed}.csv"
    for epoch in range(start_epoch, epochs):
        model.train()
        global_values, local_values, scaled_values, weighted_values, total_values = [], [], [], [], []
        local_anchor_count = positive_count = negative_count = 0
        global_grad_value = local_grad_value = ""
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"{a.arm} epoch {epoch + 1}")):
            images = batch["image"].to(device, dtype=torch.float32)
            annotations = batch["annotation"].to(device, dtype=torch.float32)
            if scheduler is not None:
                scheduler(global_step)

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
                        pos_mask[active], neg_mask[active],
                        temperature=a.temperature,
                    )

            # Exact 5K convention: local mean over active anchors, then divide by B^2.
            local_scaled = local_raw / float(images.shape[0] ** 2)
            weighted_local = lambda_local * local_scaled
            total_loss = global_loss + weighted_local

            params = [x for x in model.parameters() if x.requires_grad]
            if batch_idx == 0 and global_grad_value == "":
                gg = torch.autograd.grad(global_loss, params, retain_graph=True, allow_unused=True)
                global_grad_value = grad_norm(gg)
            if local_arm and n_anchors > 0 and local_grad_value == "":
                # Compare Global and Local gradients on the SAME first active batch.
                # If that is not the first batch, replace the fallback Global norm
                # with the norm from this active batch.
                if batch_idx > 0:
                    gg = torch.autograd.grad(global_loss, params, retain_graph=True, allow_unused=True)
                    global_grad_value = grad_norm(gg)
                gl = torch.autograd.grad(weighted_local, params, retain_graph=True, allow_unused=True)
                local_grad_value = grad_norm(gl)

            optimizer.zero_grad(set_to_none=True)
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

        if local_grad_value == "":
            local_grad_value = 0.0
        val_loss = run_validation(
            model, val_loader, criterion, device,
            max_batches=2 if a.max_steps else 0,
        )
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
            "global_grad_norm": global_grad_value,
            "local_grad_norm": local_grad_value,
            "local_global_grad_ratio": (
                float(local_grad_value) / max(float(global_grad_value), 1e-12)
                if global_grad_value != "" else ""
            ),
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1}: global={row['train_global_loss']:.8f} "
            f"local_raw={row['train_local_raw']:.6f} "
            f"local_scaled={row['train_local_scaled']:.8f} "
            f"weighted_local={row['weighted_local_loss']:.8f} "
            f"total={row['train_total_loss']:.8f} val={val_loss:.8f} "
            f"anchors={local_anchor_count}"
        )

        if not a.no_save:
            state = {
                "arm": a.arm, "seed": a.seed,
                "epoch": epoch + 1, "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng(),
                "history": history,
                "source_hashes": source_hashes,
                "init_checkpoint": str(init_path.resolve()),
                "lambda_local": lambda_local,
                "temperature": a.temperature,
            }
            atomic_torch_save(resume_path, state)
        write_history_csv(csv_path, history, columns)
        if a.max_steps and global_step >= a.max_steps:
            break

    write_history_csv(csv_path, history, columns)

    if not a.no_save:
        if len(history) < epochs:
            print(f"[PARTIAL] {len(history)}/{epochs} epochs completed; resume checkpoint retained.")
        else:
            atomic_torch_save(weight_path, {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            })
            atomic_json(log_dir / f"train_{a.arm}_seed{a.seed}.json", {
                "arm": a.arm, "seed": a.seed,
                "weight_name": weight_name,
                "weight_path": str(weight_path.resolve()),
                "init_checkpoint": str(init_path.resolve()),
                "source_hashes": source_hashes,
                "lambda_local": lambda_local,
                "temperature": a.temperature,
                "epochs_completed": len(history),
                "final": history[-1] if history else {},
            })
            print(f"[SAVED] {weight_path}")
    print(f"[LOG] {csv_path}")


if __name__ == "__main__":
    main()
