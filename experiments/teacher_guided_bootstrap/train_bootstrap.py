"""Train a Talk2DINO Student with Teacher-guided caption-head selection."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import torch
import yaml

# Run from repository root while keeping this experiment self-contained.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DinoClipDataset
from src.model import ProjectionLayer
from src.train_util import do_train, set_seed

from bootstrap_model import TeacherGuidedProjectionLayer


def _load_state_dict(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teacher-guided Talk2DINO head-selection bootstrap"
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        required=True,
        help="Already-trained Talk2DINO ProjectionLayer checkpoint",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Teacher selection weight in [0, 1]",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs/vitb_mlp_infonce.yaml",
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=REPO_ROOT / "outputs/coco2014_b14/train.pth",
    )
    parser.add_argument(
        "--val-dataset",
        type=Path,
        default=REPO_ROOT / "outputs/coco2014_b14/val.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Student checkpoint path. Defaults to weights/<config>_teacher_bootstrap_aX.pth",
    )
    parser.add_argument("--optimizer", default="Adam", choices=("Adam", "AdamW"))
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--scheduler", default="linear", choices=("linear", "cosine"))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--save-head-activations",
        type=Path,
        default=None,
        help="Optional JSON path using the repository's existing activation logger",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Talk2DINO training recipe")

    for path in (args.teacher_weights, args.model_config, args.train_dataset, args.val_dataset):
        if not path.exists():
            raise FileNotFoundError(path)

    config = yaml.safe_load(args.model_config.read_text())
    model_cfg = copy.deepcopy(config["model"])

    model_class = model_cfg.get("model_class", "ProjectionLayer")
    if model_class != "ProjectionLayer":
        raise ValueError(
            f"This experiment targets ProjectionLayer, but config requests {model_class!r}"
        )
    if model_cfg.get("alignment_strategy", "max_score") != "max_score":
        raise ValueError("This experiment requires alignment_strategy='max_score'")

    # Stage 2 requirement: Student must be randomly initialized exactly like the
    # original run. Never inherit a starting checkpoint from Teacher/config.
    model_cfg.pop("starting_checkpoint", None)

    device = torch.device("cuda")

    # Important for a clean alpha=0 ablation: initialize Student immediately
    # after the same seed call as original train.py. Creating Teacher first would
    # consume RNG and change Student initialization.
    set_seed(args.seed)
    student = TeacherGuidedProjectionLayer.from_config(model_cfg).to(device)

    teacher = ProjectionLayer.from_config(model_cfg).to(device)
    teacher.load_state_dict(_load_state_dict(args.teacher_weights), strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student.attach_teacher(teacher, args.alpha)

    # Keep original Talk2DINO data semantics unchanged:
    #   train -> disentangled attention-head features
    #   val   -> avg_self_attn_out
    #   text  -> ann_feats
    train_dataset = DinoClipDataset(
        str(args.train_dataset),
        features_name="disentangled_self_attn",
        text_features="ann_feats",
        load_attn_maps=False,
        is_wds=".tar" in str(args.train_dataset),
    )
    val_dataset = DinoClipDataset(
        str(args.val_dataset),
        features_name="avg_self_attn_out",
        text_features="ann_feats",
        load_attn_maps=False,
        is_wds=".tar" in str(args.val_dataset),
    )

    student, train_losses, val_losses = do_train(
        student,
        train_dataset,
        val_dataset,
        config["train"],
        seed=args.seed,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        warmup=args.warmup,
        save_head_attivations=(
            None if args.save_head_activations is None else str(args.save_head_activations)
        ),
    )

    if args.output is None:
        model_name = args.model_config.stem
        output = REPO_ROOT / "weights" / (
            f"{model_name}_teacher_bootstrap_a{_alpha_tag(args.alpha)}.pth"
        )
    else:
        output = args.output
        if not output.is_absolute():
            output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    # Only Student parameters are serialized. The state dict is load-compatible
    # with the repository's original ProjectionLayer and normal segmentation eval.
    torch.save(student.state_dict(), output)

    metrics_path = output.with_suffix(".metrics.csv")
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_loss", "val_loss"))
        for epoch, (train_loss, val_loss) in enumerate(
            zip(train_losses.tolist(), val_losses.tolist()), start=1
        ):
            writer.writerow((epoch, train_loss, val_loss))

    manifest = {
        "method": "teacher-guided caption-head selection bootstrap",
        "teacher_weights": str(args.teacher_weights),
        "student_init": "random/original Talk2DINO initialization",
        "alpha": args.alpha,
        "selection": "(1-alpha)*student_head_softmax + alpha*teacher_head_softmax -> hard argmax",
        "teacher_in_loss": False,
        "teacher_at_eval": False,
        "train_dataset": str(args.train_dataset),
        "val_dataset": str(args.val_dataset),
        "model_config": str(args.model_config),
        "student_checkpoint": str(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Saved Student checkpoint: {output}")
    print(f"Saved losses: {metrics_path}")
    print(f"Saved manifest: {output.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
