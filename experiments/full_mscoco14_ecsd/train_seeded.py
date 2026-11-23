#!/usr/bin/env python3
"""Run the original Talk2DINO train.py with an explicit seed, without editing it."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as base_train  # noqa: E402
from src.dataset import DinoClipDataset  # noqa: E402
from src.train_util import set_seed as original_set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_config", type=str, default="configs/vitb_mlp_infonce.yaml")
    p.add_argument("--feature_name", type=str, default="disentangled_self_attn")
    p.add_argument("--text_features", type=str, default="ann_feats")
    p.add_argument("--train_dataset", required=True)
    p.add_argument("--val_dataset", required=True)
    p.add_argument("--optimizer", type=str, default="Adam")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--scheduler", type=str, default="linear")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--name_pedix", type=str, required=True)
    p.add_argument("--save_head_activations", type=str, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--use_wandb", action="store_true")
    args = p.parse_args()

    # train.py hardcodes set_seed(123), and do_train defaults to seed=123.
    # Patch these two function references locally so the original model,
    # optimizer, loss, data loader, and training loop remain untouched.
    base_train.set_seed = lambda _ignored: original_set_seed(args.seed)
    original_do_train = base_train.do_train

    def seeded_do_train(model, train_dataset, val_dataset, train_cfg, *a, **kw):
        kw["seed"] = args.seed
        return original_do_train(model, train_dataset, val_dataset, train_cfg, *a, **kw)

    base_train.do_train = seeded_do_train
    if args.use_wandb:
        import wandb
        wandb.init(project="dino-clip")

    val_feature = "avg_self_attn_out" if args.feature_name == "disentangled_self_attn" else args.feature_name
    val_dataset = DinoClipDataset(
        args.val_dataset,
        features_name=val_feature,
        text_features=args.text_features,
        load_attn_maps=args.feature_name == "patch_tokens",
        is_wds=".tar" in args.val_dataset,
    )
    train_dataset = DinoClipDataset(
        args.train_dataset,
        features_name=args.feature_name,
        text_features=args.text_features,
        load_attn_maps=args.feature_name == "patch_tokens",
        is_wds=".tar" in args.train_dataset,
    )
    base_train.train_and_eval(
        args.model_config,
        train_dataset,
        val_dataset,
        texts=None,
        images=None,
        test_set=None,
        model_type="",
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup=args.warmup,
        name_pedix=args.name_pedix,
        save_head_activations=args.save_head_activations,
    )


if __name__ == "__main__":
    main()
