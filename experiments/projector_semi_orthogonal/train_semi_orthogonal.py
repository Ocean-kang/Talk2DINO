#!/usr/bin/env python3
"""
Minimal adapter around the repository's original train.py.

Default: enforce W^T W = I on ProjectionLayer.linear_layer.weight after every
optimizer step. Set T2DINO_SO_ENABLE=0 to run the same original training loop
without the orthogonal intervention (useful for seed-matched baselines).

No source files in Talk2DINO are edited. The adapter monkey-patches only the
runtime hooks needed for the experiment and then executes the original train.py.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))
os.chdir(ROOT)

from orthogonal import project_column_orthonormal_, orthogonality_diagnostics
from src import train_util
from src.model import ProjectionLayer


def _parse_adapter_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_config", default="configs/dinov2_vitl14_reg")
    # Keep the original repository's spelling for CLI compatibility.
    parser.add_argument("--name_pedix", default="")
    args, _ = parser.parse_known_args()
    if not args.name_pedix.strip():
        raise SystemExit(
            "ERROR: pass a non-empty --name_pedix to avoid overwriting the official checkpoint, "
            "e.g. --name_pedix so_seed123"
        )
    return args


adapter_args = _parse_adapter_args()
seed = int(os.environ.get("T2DINO_SO_SEED", "123"))
so_enabled = os.environ.get("T2DINO_SO_ENABLE", "1").strip().lower() in {
    "1", "true", "yes", "on"
}

# Seed control is a runtime override only. With seed=123 this reproduces the
# original train.py default; with other seeds it enables paired multi-seed runs.
_original_set_seed = train_util.set_seed


def _set_requested_seed(_seed=123):
    return _original_set_seed(seed)


train_util.set_seed = _set_requested_seed

if so_enabled:
    # ProjectionLayer.from_config is called by the original train.py. Project
    # the initial weight once, after any configured starting checkpoint loads.
    _original_from_config = ProjectionLayer.from_config.__func__

    def _from_config_with_so(cls, config):
        model = _original_from_config(cls, config)
        if not hasattr(model, "linear_layer"):
            raise AttributeError("ProjectionLayer has no linear_layer.weight to constrain")
        project_column_orthonormal_(model.linear_layer.weight)
        return model

    ProjectionLayer.from_config = classmethod(_from_config_with_so)

    # do_train() resolves train() from the src.train_util module at runtime.
    # Wrap that function so each original optimizer.step() is followed by the
    # projection. All loss, batching, optimizer, scheduler and epoch code stays
    # in the repository's original train_util.py.
    _original_train = train_util.train

    def _train_with_so(
        model,
        train_dataloader,
        contrastive_loss,
        optimizer,
        scheduler=None,
        wandb=False,
        save_head_attivations=None,
        n_epochs=0,
    ):
        original_step = optimizer.step

        def _step_then_project(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            project_column_orthonormal_(model.linear_layer.weight)
            return result

        optimizer.step = _step_then_project
        try:
            return _original_train(
                model,
                train_dataloader,
                contrastive_loss,
                optimizer,
                scheduler=scheduler,
                wandb=wandb,
                save_head_attivations=save_head_attivations,
                n_epochs=n_epochs,
            )
        finally:
            optimizer.step = original_step

    train_util.train = _train_with_so

print(
    "[Projector-SO] "
    f"mode={'semi-orthogonal' if so_enabled else 'baseline'} | seed={seed} | "
    f"checkpoint suffix={adapter_args.name_pedix}"
)

# Execute the repository's original entry point. The original CLI arguments
# are preserved; T2DINO_SO_* are environment variables and are not passed to it.
sys.argv[0] = str(ROOT / "train.py")
runpy.run_path(str(ROOT / "train.py"), run_name="__main__")

# Audit the checkpoint produced by the original train.py naming convention.
config_stem = Path(adapter_args.model_config).stem
checkpoint = ROOT / "weights" / f"{config_stem}_{adapter_args.name_pedix}.pth"
if not checkpoint.exists():
    raise FileNotFoundError(
        f"Expected checkpoint was not created: {checkpoint}. "
        "Check --model_config and --name_pedix."
    )

state = __import__("torch").load(str(checkpoint), map_location="cpu")
weight_key = "linear_layer.weight"
if weight_key not in state:
    raise KeyError(f"{weight_key} not found in checkpoint: {checkpoint}")

report = {
    "experiment": "Talk2DINO semi-orthogonal projector" if so_enabled else "Talk2DINO baseline",
    "semi_orthogonal_enabled": so_enabled,
    "projected_at_initialization": so_enabled,
    "projected_after_every_optimizer_step": so_enabled,
    "seed": seed,
    "checkpoint": str(checkpoint.relative_to(ROOT)),
}
report.update(orthogonality_diagnostics(state[weight_key]))

report_dir = ROOT / "outputs" / "projector_semi_orthogonal"
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / f"{adapter_args.name_pedix}_orthogonality.json"
with report_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"[Projector-SO] Weight diagnostics saved to: {report_path}")
print(json.dumps(report, indent=2, ensure_ascii=False))
