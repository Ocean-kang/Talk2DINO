#!/usr/bin/env python3
"""Train Talk2DINO with CoCu caption augmentation."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.dataset import DinoClipDataset
from src.train_util import do_train
from cocu_dataset import CocuDinoClipDataset


def main():

    p = argparse.ArgumentParser()


    p.add_argument(
        "--model-config",
        default="configs/vitb_mlp_infonce.yaml"
    )

    p.add_argument(
        "--train-dataset",
        required=True
    )

    p.add_argument(
        "--val-dataset",
        required=True
    )

    p.add_argument(
        "--curated-dataset",
        default=None
    )


    # =========================
    # 新参数
    # =========================

    p.add_argument(
        "--cocu-ratio",
        type=float,
        default=0.0,
        help="""
        Add CoCu captions while keeping COCO captions.

        0:
            only original COCO captions

        1:
            add same number of CoCu captions

        2:
            add twice number of CoCu captions
        """
    )


    p.add_argument(
        "--feature-name",
        default="disentangled_self_attn"
    )


    p.add_argument(
        "--text-features",
        default="ann_feats"
    )


    p.add_argument(
        "--optimizer",
        default="Adam",
        choices=["Adam", "AdamW"]
    )


    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.05
    )


    p.add_argument(
        "--scheduler",
        default="linear",
        choices=["linear", "cosine"]
    )


    p.add_argument(
        "--warmup",
        type=int,
        default=0
    )


    p.add_argument(
        "--seed",
        type=int,
        default=123
    )


    p.add_argument(
        "--output",
        required=True
    )


    args = p.parse_args()


    if not torch.cuda.is_available():
        raise RuntimeError(
            "Talk2DINO training requires CUDA."
        )


    with open(args.model_config,"r",encoding="utf-8") as f:
        cfg=yaml.safe_load(f)



    model_class_name = cfg["model"].get(
        "model_class",
        "ProjectionLayer"
    )


    ModelClass=getattr(
        importlib.import_module("src.model"),
        model_class_name
    )


    model=ModelClass.from_config(
        cfg["model"]
    ).cuda()



    # ==========================
    # 修改后的CoCu Dataset
    # ==========================

    train_set = CocuDinoClipDataset(

        args.train_dataset,

        curated_file=args.curated_dataset,

        cocu_ratio=args.cocu_ratio,

        features_name=args.feature_name,

        text_features=args.text_features,

        load_attn_maps=args.feature_name=="patch_tokens",

    )



    val_feature = (
        "avg_self_attn_out"
        if args.feature_name=="disentangled_self_attn"
        else args.feature_name
    )


    val_set=DinoClipDataset(

        args.val_dataset,

        features_name=val_feature,

        text_features=args.text_features,

        load_attn_maps=args.feature_name=="patch_tokens",

    )


    print(
        f"""
        train samples={len(train_set):,}
        val samples={len(val_set):,}
        cocu_ratio={args.cocu_ratio}
        """
    )



    model,_,_=do_train(

        model,

        train_set,

        val_set,

        cfg["train"],

        seed=args.seed,

        optimizer_name=args.optimizer,

        weight_decay=args.weight_decay,

        scheduler_name=args.scheduler,

        warmup=args.warmup,

    )


    out=Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        model.state_dict(),
        out
    )


    print(
        f"Saved model -> {out}"
    )



if __name__=="__main__":
    main()