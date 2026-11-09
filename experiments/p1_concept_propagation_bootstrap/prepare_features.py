from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.Caption_Head_Visualization.data import (
    caption_rows,
    resolve_image,
    select_images,
    torch_load_legacy,
)
from experiments.Caption_Head_Visualization.run import load_projector
from experiments.Concept_Guided_Head_Patch_Visualization.core import (
    concept_patch_cosine,
    propagate_concept_scores,
)
from experiments.Concept_Guided_Head_Patch_Visualization.vision import DinoRelationExtractor
from experiments.minimal_noun_supervision.common.cache import offline

from .core import pool_pseudo_head_features, summarize_caption_propagation


ROOT = Path(__file__).resolve().parents[2]
NUM_HEADS = 12
EMBED_DIM = 768
PATCH_COUNT = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_ids(value: str):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser(
        description="Build caption-conditioned P1 pseudo Head Features for Talk2DINO."
    )
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument(
        "--features", type=Path, required=True,
        help="Original Talk2DINO COCO feature PTH containing images/annotations/ann_feats.",
    )
    p.add_argument(
        "--image-root", type=Path, required=True,
        help="COCO root containing train2014/ and val2014/.",
    )
    p.add_argument(
        "--teacher-weights", type=Path,
        default=ROOT / "weights/vitb_mlp_infonce.pth",
    )
    p.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/vitb_mlp_infonce.yaml",
    )
    p.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/p1_concept_propagation_bootstrap/cache",
        help="Split-specific cache directory, e.g. .../cache/train.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 means all remaining images.")
    p.add_argument("--image-ids", default="", help="Optional comma-separated COCO image IDs.")
    p.add_argument("--dinov2-repo", default="facebookresearch/dinov2")
    p.add_argument("--hub-source", choices=["github", "local"], default="github")
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p


def _source_stat(path: Path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_or_create_cache(args, images, annotations, image_ids):
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output / "metadata.json"
    feature_path = args.output / "pseudo_features.npy"
    ann_ids_path = args.output / "annotation_ids.npy"
    image_ids_path = args.output / "image_ids.npy"
    done_path = args.output / "done_images.npy"

    payload = {
        "experiment": "P1 Concept Propagation Bootstrap",
        "split": args.split,
        "source_pth": _source_stat(args.features),
        "teacher_weights": str(args.teacher_weights.resolve()),
        "teacher_weights_sha256": sha256_file(args.teacher_weights),
        "projector_config": str(args.config.resolve()),
        "projector_config_sha256": sha256_file(args.config),
        "image_root": str(args.image_root.resolve()),
        "dinov2_model": "dinov2_vitb14_reg",
        "dinov2_repo": args.dinov2_repo,
        "hub_source": args.hub_source,
        "input_size": int(args.size),
        "patch_size": 14,
        "patch_grid": [32, 32],
        "num_global_tokens": 5,
        "num_heads": NUM_HEADS,
        "head_dim": 64,
        "embed_dim": EMBED_DIM,
        "num_patches": PATCH_COUNT,
        "p1_definition": "softmax((Q_patch / sqrt(head_dim)) @ K_patch.T, dim=key)",
        "p1_orientation": "A[h,query_patch,key_patch]",
        "propagation": "propagated[h,p,c] = sum_q A[h,p,q] * seed[q,c]",
        "seed_response": "cosine(DINO x_norm_patchtokens, Teacher.project_clip_txt(raw CLIP ann_feats))",
        "caption_unit": "one complete original COCO caption per annotation; no noun parsing or gallery lookup",
        "response_temperature": float(args.temperature),
        "weighting": "softmax(propagated_response / temperature, over patches)",
        "pooling": "mean_p(weight[h,p,c] * x_norm_patchtokens[p])",
        "feature_dtype": "float16",
        "annotation_count": int(len(annotations)),
        "image_count": int(len(images)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    existing_files = [
        feature_path.exists(), ann_ids_path.exists(), image_ids_path.exists(), done_path.exists()
    ]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing cache metadata does not match current inputs/config: {args.output}. "
                "Use a new output directory rather than mixing caches."
            )
        if not all(existing_files):
            raise RuntimeError("Cache metadata exists but one or more cache arrays are missing.")
    else:
        if any(existing_files):
            raise RuntimeError(
                f"Cache arrays exist without metadata in {args.output}; "
                "choose a fresh output directory."
            )
        np.save(ann_ids_path, np.asarray([int(a["id"]) for a in annotations], dtype=np.int64))
        np.save(image_ids_path, np.asarray(image_ids, dtype=np.int64))
        np.save(done_path, np.zeros(len(images), dtype=np.bool_))
        feature_array = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(annotations), NUM_HEADS, EMBED_DIM),
        )
        feature_array.flush()
        del feature_array
        meta = {
            **payload,
            "fingerprint": fingerprint,
            "complete": False,
            "completed_image_count": 0,
            "failed_image_count": 0,
        }
        write_json(meta_path, meta)

    feature_array = np.load(feature_path, mmap_mode="r+")
    done_array = np.load(done_path, mmap_mode="r+")
    if tuple(feature_array.shape) != (len(annotations), NUM_HEADS, EMBED_DIM):
        raise ValueError(f"Unexpected pseudo feature array shape: {feature_array.shape}")
    if tuple(done_array.shape) != (len(images),):
        raise ValueError("done_images array has an unexpected shape")
    return meta, feature_array, done_array, meta_path, feature_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.size != 448:
        raise ValueError("This experiment fixes DINOv2 ViT-B/14-reg input size to 448.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.offset < 0 or args.limit < 0:
        raise ValueError("--offset and --limit must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a slow smoke test.")
    for path, label in (
        (args.features, "source feature PTH"),
        (args.teacher_weights, "Teacher projector checkpoint"),
        (args.config, "projector config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not args.image_root.is_dir():
        raise NotADirectoryError(f"Missing COCO image root: {args.image_root}")

    # Prevent accidental torch.hub/network downloads. DINOv2 must already be cached.
    offline()

    source = torch_load_legacy(args.features)
    if not isinstance(source, dict) or not {"images", "annotations"} <= set(source):
        raise ValueError("Expected original Talk2DINO PTH with images and annotations.")
    images = source["images"]
    annotations = source["annotations"]
    if not images or not annotations:
        raise ValueError("Source PTH has no images or annotations.")

    image_ids = [int(im["id"]) for im in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image IDs in source PTH.")
    annotation_ids = [int(a["id"]) for a in annotations]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("Duplicate annotation IDs in source PTH.")

    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[int(ann["image_id"])].append(ann)
    image_index = {iid: idx for idx, iid in enumerate(image_ids)}
    annotation_row = {aid: idx for idx, aid in enumerate(annotation_ids)}

    selected = select_images(
        images,
        ids=parse_ids(args.image_ids),
        offset=args.offset,
        limit=args.limit,
    )
    if not selected:
        raise ValueError("No images selected.")

    meta, pseudo_array, done_array, meta_path, feature_path = _load_or_create_cache(
        args, images, annotations, image_ids
    )
    already_done = int(done_array.sum())
    print(
        f"[CACHE] {already_done}/{len(images)} images already complete; "
        f"{len(selected)} selected in this invocation.",
        flush=True,
    )

    teacher = load_projector(args.config, args.teacher_weights, args.device)
    teacher.eval().requires_grad_(False)

    extractor = DinoRelationExtractor(
        device=args.device,
        size=args.size,
        repo=args.dinov2_repo,
        hub_source=args.hub_source,
    )
    if extractor.num_heads != NUM_HEADS:
        raise RuntimeError(f"Expected {NUM_HEADS} heads, got {extractor.num_heads}")

    diagnostics_path = args.output / "caption_head_diagnostics.jsonl"
    errors_path = args.output / "errors.jsonl"
    diagnostics_file = None if args.no_diagnostics else diagnostics_path.open("a", encoding="utf-8")
    errors_file = errors_path.open("a", encoding="utf-8")

    processed_now = 0
    skipped_now = 0
    errors_now = 0
    try:
        for step, image_record in enumerate(selected, start=1):
            iid = int(image_record["id"])
            image_row = image_index[iid]
            if bool(done_array[image_row]):
                skipped_now += 1
                continue

            try:
                rows = caption_rows(annotations_by_image, iid)
                if any("ann_feats" not in row for row in rows):
                    raise KeyError(
                        f"Image {iid}: one or more annotations lack ann_feats. "
                        "Run the original CLIP text feature extraction first."
                    )
                captions = [str(row["caption"]) for row in rows]
                caption_features = torch.stack([
                    torch.as_tensor(row["ann_feats"]).float().reshape(-1) for row in rows
                ])
                if caption_features.ndim != 2 or caption_features.shape[1] != 512:
                    raise ValueError(
                        f"Image {iid}: expected raw CLIP caption features [C,512], "
                        f"got {tuple(caption_features.shape)}"
                    )

                image_path = resolve_image(args.image_root, image_record)
                with Image.open(image_path) as image:
                    original = image.convert("RGB").copy()

                extracted = extractor.extract(original)
                patches = extracted["patch_tokens"].float()
                p1_affinity = extracted["p1_affinity"].float()
                if tuple(patches.shape) != (PATCH_COUNT, EMBED_DIM):
                    raise ValueError(f"Unexpected patch shape: {tuple(patches.shape)}")
                if tuple(p1_affinity.shape) != (NUM_HEADS, PATCH_COUNT, PATCH_COUNT):
                    raise ValueError(f"Unexpected P1 shape: {tuple(p1_affinity.shape)}")

                with torch.no_grad():
                    projected_captions = teacher.project_clip_txt(
                        caption_features.to(args.device)
                    ).float()
                    if tuple(projected_captions.shape) != (len(rows), EMBED_DIM):
                        raise ValueError(
                            f"Teacher projected caption shape mismatch: "
                            f"{tuple(projected_captions.shape)}"
                        )

                    # Reuse the exact seed-response and P1 propagation functions
                    # from Concept_Guided_Head_Patch_Visualization/core.py.
                    seed_scores = concept_patch_cosine(patches, projected_captions)
                    propagated = propagate_concept_scores(p1_affinity, seed_scores)
                    pseudo_by_caption, patch_weights = pool_pseudo_head_features(
                        patches, propagated, temperature=args.temperature
                    )

                if tuple(pseudo_by_caption.shape) != (len(rows), NUM_HEADS, EMBED_DIM):
                    raise RuntimeError(
                        f"Unexpected pseudo feature shape: {tuple(pseudo_by_caption.shape)}"
                    )

                for caption_index, ann in enumerate(rows):
                    aid = int(ann["id"])
                    row_index = annotation_row[aid]
                    pseudo_array[row_index] = (
                        pseudo_by_caption[caption_index]
                        .detach().float().cpu().numpy().astype(np.float16)
                    )

                    if diagnostics_file is not None:
                        stats = summarize_caption_propagation(
                            seed_scores, propagated, patch_weights, caption_index
                        )
                        record = {
                            "annotation_id": aid,
                            "image_id": iid,
                            "caption": str(ann["caption"]),
                            "teacher_checkpoint": str(args.teacher_weights.resolve()),
                            "response_temperature": float(args.temperature),
                            **stats,
                        }
                        diagnostics_file.write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                        )

                # Mark an image done only after all of its caption features were written.
                pseudo_array.flush()
                done_array[image_row] = True
                done_array.flush()
                processed_now += 1

                if processed_now % 10 == 0 or step == len(selected):
                    print(
                        f"[{step}/{len(selected)}] image={iid} "
                        f"processed_now={processed_now} cache_done={int(done_array.sum())}/{len(images)}",
                        flush=True,
                    )

            except Exception as exc:
                errors_now += 1
                error_record = {
                    "image_id": iid,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors_file.write(
                    json.dumps(error_record, ensure_ascii=False) + "\n"
                )
                errors_file.flush()
                print(f"[ERROR] image {iid}: {exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
    finally:
        if diagnostics_file is not None:
            diagnostics_file.close()
        errors_file.close()
        extractor.close()

    complete = bool(np.asarray(done_array).all())
    meta.update({
        "complete": complete,
        "completed_image_count": int(np.asarray(done_array).sum()),
        "failed_image_count": int(errors_now),
        "last_invocation": {
            "offset": int(args.offset),
            "limit": int(args.limit),
            "selected_image_ids": [int(x["id"]) for x in selected],
            "processed_now": int(processed_now),
            "skipped_now": int(skipped_now),
            "errors_now": int(errors_now),
        },
    })
    write_json(meta_path, meta)
    write_json(args.output / "summary.json", {
        "split": args.split,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "completed_image_count": int(np.asarray(done_array).sum()),
        "complete": complete,
        "processed_now": processed_now,
        "skipped_now": skipped_now,
        "errors_now": errors_now,
        "feature_array": str(feature_path.resolve()),
        "feature_shape": list(pseudo_array.shape),
        "feature_dtype": str(pseudo_array.dtype),
    })

    print(
        f"[DONE] {args.split}: complete={complete}, "
        f"processed={processed_now}, skipped={skipped_now}, errors={errors_now}",
        flush=True,
    )
    if errors_now:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
