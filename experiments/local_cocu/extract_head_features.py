import os
import sys
import json
import math
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------
# Talk2DINO root
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------
# Reuse official Talk2DINO attention implementation
# ---------------------------------------------------------

from src.hooks import (
    get_self_attention,
    process_self_attention,
    feats,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# COCO
# ---------------------------------------------------------

def load_coco_images(annotation_path):

    with open(annotation_path, "r") as f:
        data = json.load(f)

    return data["images"]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def load_model(model_name):

    print("Loading DINO model:")
    print(model_name)

    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name
    )

    model = model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------
# Transform
# ---------------------------------------------------------

def build_transform(
        resize_dim,
        crop_dim
):

    return T.Compose([
        T.Resize(
            resize_dim,
            interpolation=T.InterpolationMode.BICUBIC
        ),

        T.CenterCrop(
            crop_dim
        ),

        T.ToTensor(),

        T.Normalize(
            mean=(
                0.485,
                0.456,
                0.406
            ),
            std=(
                0.229,
                0.224,
                0.225
            )
        ),
    ])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    # -----------------------------------------------------
    # Official Talk2DINO ViT-B configuration
    # -----------------------------------------------------

    model = load_model(
        args.model
    )

    transform = build_transform(
        args.resize_dim,
        args.crop_dim
    )

    images = load_coco_images(
        args.annotation
    )

    print(
        "Number of images:",
        len(images)
    )

    # -----------------------------------------------------
    # DINOv2 ViT-B/14-reg
    #
    # 448 / 14 = 32
    #
    # 32 * 32 = 1024 patch tokens
    #
    # 1 CLS + 4 register = 5 global tokens
    # -----------------------------------------------------

    num_global_tokens = (
        5
        if "reg" in args.model
        else 1
    )

    num_patch_tokens = (
        args.crop_dim // 14
    ) ** 2

    num_tokens = (
        num_global_tokens
        +
        num_patch_tokens
    )

    embed_dim = 768

    num_attn_heads = model.num_heads

    head_dim = (
        embed_dim
        //
        num_attn_heads
    )

    scale = (
        head_dim ** -0.5
    )

    print()
    print("DINO configuration")
    print("------------------------------")
    print("model:", args.model)
    print("crop:", args.crop_dim)
    print("patch tokens:", num_patch_tokens)
    print("global tokens:", num_global_tokens)
    print("total tokens:", num_tokens)
    print("heads:", num_attn_heads)
    print("embed dim:", embed_dim)
    print("head dim:", head_dim)
    print("attention scale:", scale)
    print("------------------------------")
    print()

    # -----------------------------------------------------
    # Official Talk2DINO hook:
    #
    # last transformer block
    #        ↓
    # attention.qkv
    #        ↓
    # feats["self_attn"]
    # -----------------------------------------------------

    hook = (
        model.blocks[-1]
        .attn
        .qkv
        .register_forward_hook(
            get_self_attention
        )
    )

    results = {}

    failed = []

    total_batches = math.ceil(
        len(images)
        /
        args.batch_size
    )

    # -----------------------------------------------------
    # Extraction
    # -----------------------------------------------------

    try:

        for batch_idx in tqdm(
            range(total_batches)
        ):

            start = (
                batch_idx
                *
                args.batch_size
            )

            end = min(
                start
                +
                args.batch_size,
                len(images)
            )

            batch_meta = images[
                start:end
            ]

            batch_tensors = []

            valid_meta = []

            # -------------------------------------------------
            # Load images
            # -------------------------------------------------

            for item in batch_meta:

                image_id = item["id"]

                filename = item[
                    "file_name"
                ]

                image_path = os.path.join(
                    args.image_dir,
                    filename
                )

                try:

                    image = Image.open(
                        image_path
                    ).convert(
                        "RGB"
                    )

                except Exception as e:

                    print(
                        f"\nFailed image "
                        f"{image_id}: "
                        f"{image_path}"
                    )

                    print(e)

                    failed.append(
                        image_id
                    )

                    continue

                image = transform(
                    image
                )

                batch_tensors.append(
                    image
                )

                valid_meta.append(
                    item
                )

            if not batch_tensors:
                continue

            batch = torch.stack(
                batch_tensors
            ).to(
                device,
                non_blocking=True
            )

            current_batch_size = (
                batch.shape[0]
            )

            # -------------------------------------------------
            # DINO forward
            #
            # is_training=True is important:
            # same interface as official Talk2DINO extractor
            # -------------------------------------------------

            with torch.no_grad():

                outs = model(
                    batch,
                    is_training=True
                )

                # ---------------------------------------------
                # Normalized patch tokens
                #
                # shape:
                # [B, 1024, 768]
                # ---------------------------------------------

                patch_tokens = outs[
                    "x_norm_patchtokens"
                ]

                # ---------------------------------------------
                # process_self_attention()
                #
                # This is Talk2DINO official implementation.
                #
                # self_attn_maps:
                #
                # [B, H, Npatch]
                # ---------------------------------------------

                (
                    _,
                    self_attn_maps
                ) = process_self_attention(
                    feats[
                        "self_attn"
                    ],
                    current_batch_size,
                    num_tokens,
                    num_attn_heads,
                    embed_dim,
                    scale,
                    num_global_tokens,
                    ret_self_attn_maps=True
                )

                # ---------------------------------------------
                # Official Talk2DINO disentangled attention
                #
                # Normalize each head spatially
                # ---------------------------------------------

                self_attn_maps = (
                    self_attn_maps
                    .softmax(
                        dim=-1
                    )
                )

                # ---------------------------------------------
                # patch_tokens:
                #
                # [B, N, 768]
                #
                # ->
                #
                # [B, 1, N, 768]
                #
                #
                # attention:
                #
                # [B, H, N]
                #
                # ->
                #
                # [B, H, N, 1]
                #
                #
                # result:
                #
                # [B, H, 768]
                # ---------------------------------------------

                disentangled = (
                    patch_tokens.unsqueeze(1)
                    *
                    self_attn_maps.unsqueeze(-1)
                ).mean(
                    dim=2
                )

            # -------------------------------------------------
            # sanity check
            # -------------------------------------------------

            expected_shape = (
                current_batch_size,
                num_attn_heads,
                embed_dim
            )

            if (
                tuple(
                    disentangled.shape
                )
                !=
                expected_shape
            ):

                raise RuntimeError(
                    "\nUnexpected "
                    "disentangled feature shape\n"
                    f"got: "
                    f"{disentangled.shape}\n"
                    f"expected: "
                    f"{expected_shape}\n"
                )

            # -------------------------------------------------
            # Save each image
            #
            # float16 saves about 50% disk space.
            #
            # match_head_concepts.py later converts
            # it back to float().
            # -------------------------------------------------

            disentangled = (
                disentangled
                .detach()
                .cpu()
            )

            if args.fp16:

                disentangled = (
                    disentangled
                    .half()
                )

            for local_idx, item in enumerate(
                valid_meta
            ):

                image_id = item["id"]

                results[
                    image_id
                ] = disentangled[
                    local_idx
                ]

            # -------------------------------------------------
            # periodic checkpoint
            #
            # Avoid losing all work if extraction stops.
            # -------------------------------------------------

            if (
                args.save_every > 0
                and
                (
                    batch_idx + 1
                )
                %
                args.save_every
                == 0
            ):

                temp_path = (
                    args.output
                    +
                    ".tmp"
                )

                torch.save(
                    results,
                    temp_path
                )

                print()
                print(
                    "Checkpoint saved:"
                )
                print(
                    temp_path
                )
                print(
                    "images:",
                    len(results)
                )

    finally:

        hook.remove()

    # -----------------------------------------------------
    # Final save
    # -----------------------------------------------------

    torch.save(
        results,
        args.output
    )

    # -----------------------------------------------------
    # Failed image IDs
    # -----------------------------------------------------

    failed_path = (
        os.path.splitext(
            args.output
        )[0]
        +
        "_failed.json"
    )

    with open(
        failed_path,
        "w"
    ) as f:

        json.dump(
            failed,
            f,
            indent=2
        )

    print()
    print(
        "================================"
    )

    print(
        "Real Talk2DINO head extraction done"
    )

    print(
        "================================"
    )

    print(
        "saved images:",
        len(results)
    )

    print(
        "failed images:",
        len(failed)
    )

    print(
        "output:",
        args.output
    )

    print(
        "failed list:",
        failed_path
    )

    # -----------------------------------------------------
    # final sanity check
    # -----------------------------------------------------

    if results:

        first_id = next(
            iter(results)
        )

        first_feat = results[
            first_id
        ]

        print()
        print(
            "Example image:",
            first_id
        )

        print(
            "Feature shape:",
            first_feat.shape
        )

        print(
            "dtype:",
            first_feat.dtype
        )


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation",
        default=(
            "data/coco2014/"
            "annotations/"
            "captions_train2014.json"
        )
    )

    parser.add_argument(
        "--image_dir",
        default=(
            "data/coco2014/"
            "train2014"
        )
    )

    parser.add_argument(
        "--model",
        default=(
            "dinov2_vitb14_reg"
        )
    )

    parser.add_argument(
        "--resize_dim",
        type=int,
        default=448
    )

    parser.add_argument(
        "--crop_dim",
        type=int,
        default=448
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/local_cocu/"
            "head_features/"
            "head_features.pt"
        )
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=500
    )

    parser.add_argument(
        "--fp16",
        action="store_true"
    )

    args = parser.parse_args()

    main(args)
