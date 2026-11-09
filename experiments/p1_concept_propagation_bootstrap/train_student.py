from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml

from src.model import ProjectionLayer
from src.train_util import do_train, set_seed

from .dataset import P1PseudoFeatureDataset


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a freshly initialized Talk2DINO Student on P1 pseudo Head Features."
    )
    p.add_argument("--train-features", type=Path, default=ROOT / "outputs/coco2014_b14/train.pth")
    p.add_argument("--val-features", type=Path, default=ROOT / "outputs/coco2014_b14/val.pth")
    p.add_argument(
        "--train-cache", type=Path,
        default=ROOT / "outputs/p1_concept_propagation_bootstrap/cache/train",
    )
    p.add_argument(
        "--val-cache", type=Path,
        default=ROOT / "outputs/p1_concept_propagation_bootstrap/cache/val",
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs/vitb_mlp_infonce.yaml")
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/p1_concept_propagation_bootstrap/student_seed123",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--optimizer", choices=["Adam", "AdamW"], default="Adam")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument(
        "--allow-partial-cache", action="store_true",
        help="Smoke-test only: train/validate on image IDs already completed in partial caches.",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a smoke test.")
    if not args.config.is_file():
        raise FileNotFoundError(f"Missing model config: {args.config}")

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output}. "
            "Use a new output path; existing checkpoints are never overwritten."
        )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if "model" not in config or "train" not in config:
        raise ValueError("Expected model and train sections in the Talk2DINO YAML config.")

    train_cfg = dict(config["train"])
    if args.epochs is not None:
        train_cfg["num_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)

    train_dataset = P1PseudoFeatureDataset(
        args.train_features,
        args.train_cache,
        training=True,
        allow_partial=args.allow_partial_cache,
        max_samples=args.max_train_samples,
    )
    val_dataset = P1PseudoFeatureDataset(
        args.val_features,
        args.val_cache,
        training=False,
        allow_partial=args.allow_partial_cache,
        max_samples=args.max_val_samples,
    )

    # Match train.py: seed BEFORE constructing the randomly initialized Student.
    set_seed(args.seed)
    student = ProjectionLayer.from_config(config["model"]).to(args.device)

    print(
        f"[STUDENT] fresh ProjectionLayer; train_samples={len(train_dataset)}, "
        f"val_samples={len(val_dataset)}, train_cfg={train_cfg}",
        flush=True,
    )
    trained_model, train_losses, val_losses = do_train(
        student,
        train_dataset,
        val_dataset,
        train_cfg,
        seed=args.seed,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        warmup=args.warmup,
    )

    checkpoint_path = args.output / "student_projector.pth"
    torch.save(trained_model.state_dict(), checkpoint_path)

    result = {
        "experiment": "P1 Concept Propagation Bootstrap",
        "teacher_used_for_training": False,
        "student_initialization": "fresh ProjectionLayer.from_config; no Teacher weights copied",
        "source_config": str(args.config.resolve()),
        "source_config_sha256": sha256_file(args.config),
        "train_features": str(args.train_features.resolve()),
        "val_features": str(args.val_features.resolve()),
        "train_cache": str(args.train_cache.resolve()),
        "val_cache": str(args.val_cache.resolve()),
        "train_cache_fingerprint": json.loads(
            (args.train_cache / "metadata.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "val_cache_fingerprint": json.loads(
            (args.val_cache / "metadata.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "seed": int(args.seed),
        "optimizer": args.optimizer,
        "weight_decay": float(args.weight_decay),
        "scheduler": args.scheduler,
        "warmup": int(args.warmup),
        "train_config": train_cfg,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "partial_cache_allowed": bool(args.allow_partial_cache),
        "train_losses": [float(x) for x in train_losses.tolist()],
        "val_losses": [float(x) for x in val_losses.tolist()],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (args.output / "training_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(args.config, args.output / "model_config.yaml")

    print(f"[DONE] Student checkpoint: {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
