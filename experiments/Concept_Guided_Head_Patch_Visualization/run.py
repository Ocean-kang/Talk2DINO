"""Run frozen Concept-Guided Head-Patch Visualization from the Talk2DINO root."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch
import torch.nn.functional as F
from PIL import Image

from experiments.Caption_Head_Visualization.data import (
    caption_rows,
    load_coco_feature_pth,
    resolve_image,
    select_images,
)
from experiments.Caption_Head_Visualization.run import load_projector
from experiments.minimal_noun_supervision.common.cache import offline
from .core import (
    concept_patch_cosine,
    propagate_concept_scores,
    retrieve_head_concepts,
)
from .gallery import load_gallery_and_matcher, encode_gallery_clip_features
from .vision import DinoRelationExtractor
from .visualize import save_image_visualizations, write_html_gallery

ROOT = Path(__file__).resolve().parents[2]


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _parse_ids(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _slug(text):
    import re
    value = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return value[:60] or "concept"


def _concept_index(gallery):
    return {name: i for i, name in enumerate(gallery)}


def _caption_concept_rows(caption_match, gallery, head_scores):
    rows = []
    lookup = _concept_index(gallery)
    for concept in caption_match["concepts"]:
        gid = lookup[concept]
        values, best = head_scores[:, gid].max(dim=0)
        rows.append({
            "concept": concept,
            "caption_frequency": int(caption_match["caption_frequency"].get(concept, 0)),
            "gallery_index": int(gid),
            "best_head_index": int(best),
            "best_head_number": int(best) + 1,
            "best_head_cosine": float(values),
            "all_head_cosines": [float(x) for x in head_scores[:, gid].tolist()],
            "map_file": None,
        })
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Frozen P0/P1/P2 visualization with Talk2DINO W_projector and the existing MSCOCO14 gallery"
    )
    p.add_argument("--features", type=Path, default=ROOT / "outputs/coco2014_b14/train.pth")
    p.add_argument("--image-root", type=Path, required=True,
                   help="MSCOCO14 root containing train2014/ and val2014/")
    p.add_argument("--gallery", type=Path,
                   default=ROOT / "experiments/local_cocu/MSCOCO14_train_gallery.json")
    p.add_argument("--text-bank", type=Path, default=None,
                   help="Existing verified minimal_noun_supervision text_bank.pth")
    p.add_argument("--clip-checkpoint", type=Path, default=None,
                   help="Existing local CLIP ViT-B/16 checkpoint file; never a model name")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/vitb_mlp_infonce.pth")
    p.add_argument("--config", type=Path, default=ROOT / "configs/vitb_mlp_infonce.yaml")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/Concept_Guided_Head_Patch_Visualization")
    p.add_argument("--limit", type=int, default=10, help="Images after offset; 0 means all")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--image-ids", default="", help="Comma-separated COCO image IDs; overrides offset/limit")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-caption-concept-maps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--query-x", type=int, default=None,
                   help="Patch-grid x coordinate for query-conditioned maps; default center")
    p.add_argument("--query-y", type=int, default=None,
                   help="Patch-grid y coordinate for query-conditioned maps; default center")
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--min-cache-cosine", type=float, default=0.99)
    p.add_argument("--strict-cache", action="store_true",
                   help="Fail if recomputed QKV heads disagree with cached disentangled_self_attn")
    p.add_argument("--dinov2-repo", default="facebookresearch/dinov2")
    p.add_argument("--hub-source", choices=["github", "local"], default="github")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args(argv)
    if args.limit < 0 or args.offset < 0 or args.topk < 1:
        p.error("limit/offset must be nonnegative and topk positive")
    if args.size != 448:
        p.error("size is fixed at 448 to match the original Talk2DINO ViT-B/14-reg protocol")
    if not (0 <= args.alpha <= 1):
        p.error("alpha must be in [0,1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is unavailable; use --device cpu for a slower diagnostic")
    if not args.features.is_file():
        p.error(f"Missing feature PTH: {args.features}")
    if not args.gallery.is_file():
        p.error(f"Missing existing gallery JSON: {args.gallery}")
    if not args.weights.is_file() or not args.config.is_file():
        p.error("Projector weights/config not found; check --weights and --config")
    if not args.image_root.is_dir():
        p.error(f"Missing image root: {args.image_root}")
    if args.text_bank is None and args.clip_checkpoint is None:
        p.error("Supply --text-bank or --clip-checkpoint for local CLIP gallery features")
    if args.output.exists() and any(args.output.iterdir()):
        p.error(
            f"Output directory is not empty: {args.output}. "
            "Choose a fresh --output path to avoid overwriting prior results."
        )
    if args.query_x is not None and not 0 <= args.query_x < args.size // 14:
        p.error("--query-x is outside the 32x32 patch grid")
    if args.query_y is not None and not 0 <= args.query_y < args.size // 14:
        p.error("--query-y is outside the 32x32 patch grid")
    return args


def main(argv=None):
    args = parse_args(argv)

    # Reuse the repository's offline guard. No model/data/package downloads.
    offline()

    args.output.mkdir(parents=True, exist_ok=True)
    gallery, aliases, matcher = load_gallery_and_matcher(args.gallery)
    print(f"[GALLERY] {len(gallery)} concepts loaded from {args.gallery}", flush=True)

    projector = load_projector(args.config, args.weights, args.device)
    clip_gallery = encode_gallery_clip_features(
        gallery,
        device=args.device,
        text_bank=args.text_bank,
        clip_checkpoint=args.clip_checkpoint,
        batch_size=args.batch_size,
    )
    with torch.no_grad():
        projected_gallery = projector.project_clip_txt(clip_gallery.to(args.device)).float()
    if projected_gallery.shape != (len(gallery), 768):
        raise ValueError(
            f"Expected projected gallery [{len(gallery)},768], got {tuple(projected_gallery.shape)}"
        )
    projected_gallery = F.normalize(projected_gallery, dim=-1)

    images, annotations_by_image = load_coco_feature_pth(args.features)
    selected = select_images(
        images,
        ids=_parse_ids(args.image_ids),
        offset=args.offset,
        limit=args.limit,
    )
    if not selected:
        raise ValueError("No images selected from feature PTH")
    print(f"[DATA] {len(selected)} images selected from {len(images)} cached images", flush=True)

    grid = args.size // 14
    qx = grid // 2 if args.query_x is None else args.query_x
    qy = grid // 2 if args.query_y is None else args.query_y
    query_index = qy * grid + qx

    run_config = {
        "experiment": "Concept-Guided Head-Patch Visualization",
        "features": str(args.features.resolve()),
        "features_sha256": _sha256(args.features),
        "gallery": str(args.gallery.resolve()),
        "gallery_sha256": _sha256(args.gallery),
        "gallery_size": len(gallery),
        "gallery_aliases": aliases,
        "text_bank": str(args.text_bank.resolve()) if args.text_bank else None,
        "clip_checkpoint": str(args.clip_checkpoint.resolve()) if args.clip_checkpoint else None,
        "weights": str(args.weights.resolve()),
        "weights_sha256": _sha256(args.weights),
        "config": str(args.config.resolve()),
        "image_root": str(args.image_root.resolve()),
        "image_model": "dinov2_vitb14_reg",
        "text_model": "CLIP ViT-B/16",
        "projector": "src.model.ProjectionLayer.project_clip_txt",
        "image_size": 448,
        "patch_grid": [grid, grid],
        "num_global_tokens": 5,
        "num_heads": 12,
        "head_dim": 64,
        "topk_gallery_per_head": args.topk,
        "query_patch_xy": [qx, qy],
        "query_patch_index": query_index,
        "p1_definition": "softmax(Q_patch K_patch^T) over patch keys only",
        "p1_concept_propagation": "A_h @ cosine(DINO_patch, W_projector(CLIP_concept))",
        "p2_definition": "Z_h=A_h @ V_h_patch; pairwise cosine affinity over Z_h",
        "training": False,
        "weights_modified": False,
        "offline_guard": True,
        "selected_image_ids": [int(x["id"]) for x in selected],
    }
    (args.output / "run_config.json").write_text(_json(run_config), encoding="utf-8")
    (args.output / "gallery.json").write_text(
        json.dumps(gallery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    extractor = DinoRelationExtractor(
        device=args.device,
        size=args.size,
        repo=args.dinov2_repo,
        hub_source=args.hub_source,
    )
    successes, errors = [], []
    concept_counts = Counter()
    head_concept_counts = [Counter() for _ in range(12)]
    try:
        for step, image_record in enumerate(selected, start=1):
            image_id = int(image_record["id"])
            image_out = args.output / "images" / str(image_id)
            print(f"[{step}/{len(selected)}] image {image_id}", flush=True)
            try:
                rows = caption_rows(annotations_by_image, image_id)
                captions = [str(x["caption"]) for x in rows]
                annotation_ids = [x.get("id") for x in rows]
                image_path = resolve_image(args.image_root, image_record)
                with Image.open(image_path) as image:
                    original = image.convert("RGB").copy()

                extracted = extractor.extract(original, query_index=query_index)
                heads = extracted["heads"].float()
                patches = extracted["patch_tokens"].float()
                p1_affinity = extracted["p1_affinity"].float()

                cached_cosines = None
                cache_warning = None
                cached = image_record.get("disentangled_self_attn")
                if cached is not None:
                    cached = torch.as_tensor(cached).float().to(heads.device)
                    if tuple(cached.shape) == tuple(heads.shape):
                        cached_cosines = F.cosine_similarity(cached, heads, dim=-1).tolist()
                        if min(cached_cosines) < args.min_cache_cosine:
                            cache_warning = (
                                f"Cached/recomputed head cosine minimum {min(cached_cosines):.5f} "
                                f"is below {args.min_cache_cosine:.5f}"
                            )
                    else:
                        cache_warning = (
                            f"Cached head shape {tuple(cached.shape)} differs from "
                            f"recomputed {tuple(heads.shape)}"
                        )
                    if cache_warning and args.strict_cache:
                        raise ValueError(cache_warning)

                head_scores, head_topk = retrieve_head_concepts(
                    heads, projected_gallery, gallery, topk=args.topk
                )
                caption_match = matcher.extract(captions)
                caption_concepts = _caption_concept_rows(caption_match, gallery, head_scores)

                # Build patch-concept seed scores only for concepts actually used
                # in this image's per-head Top-1 or original-caption matches.
                head_top1_ids = [row[0]["gallery_index"] for row in head_topk]
                selected_ids = sorted(set(head_top1_ids + caption_match["gallery_ids"]))
                selected_projected = projected_gallery[selected_ids]
                patch_seed = concept_patch_cosine(patches, selected_projected)
                propagated = propagate_concept_scores(p1_affinity, patch_seed)

                selected_column = {gid: col for col, gid in enumerate(selected_ids)}
                p1_top1 = torch.stack([
                    propagated[h, :, selected_column[head_top1_ids[h]]]
                    for h in range(len(head_top1_ids))
                ], dim=0)

                relations = {
                    "p0_cls_to_patch": extracted["p0_cls_to_patch"].reshape(12, grid, grid).cpu(),
                    "p1_mean_incoming": extracted["p1_mean_incoming"].reshape(12, grid, grid).cpu(),
                    "p1_query_conditioned": extracted["p1_query_conditioned"].reshape(12, grid, grid).cpu(),
                    "p1_concept_propagation": p1_top1.reshape(12, grid, grid).cpu(),
                    "p2_mean_feature_affinity": extracted["p2_mean_feature_affinity"].reshape(12, grid, grid).cpu(),
                    "p2_query_feature_affinity": extracted["p2_query_feature_affinity"].reshape(12, grid, grid).cpu(),
                    "p1_patch_mass_in_full_attention": extracted["p1_patch_mass_in_full_attention"].cpu(),
                    "grid_size": grid,
                    "query_index": query_index,
                }

                image_out.mkdir(parents=True, exist_ok=True)
                heads_meta = save_image_visualizations(
                    extracted["crop"],
                    relations,
                    head_topk,
                    caption_concepts,
                    image_out,
                    alpha=args.alpha,
                )

                # Save one caption-concept propagation map on its best-matching head.
                cap_map_count = 0
                cap_dir = image_out / "caption_concepts"
                cap_dir.mkdir(parents=True, exist_ok=True)
                for concept_row in caption_concepts:
                    if cap_map_count >= args.max_caption_concept_maps:
                        break
                    gid = concept_row["gallery_index"]
                    h = concept_row["best_head_index"]
                    col = selected_column[gid]
                    scalar = (
                        propagated[h, :, col]
                        .detach()
                        .float()
                        .cpu()
                        .reshape(grid, grid)
                    )
                    filename = f"{_slug(concept_row['concept'])}_best_head_{h + 1:02d}.png"
                    from experiments.Caption_Head_Visualization.visualize import overlay_attention
                    overlay_attention(
                        extracted["crop"],
                        scalar,
                        alpha=args.alpha,
                    ).save(cap_dir / filename)
                    concept_row["map_file"] = filename
                    concept_row["map_path"] = f"caption_concepts/{filename}"
                    cap_map_count += 1

                npz_path = image_out / "relations.npz"
                import numpy as np
                np.savez_compressed(
                    npz_path,
                    p0_cls_to_patch=relations["p0_cls_to_patch"].numpy().astype("float32"),
                    p1_mean_incoming=relations["p1_mean_incoming"].numpy().astype("float32"),
                    p1_query_conditioned=relations["p1_query_conditioned"].numpy().astype("float32"),
                    p1_concept_propagation=relations["p1_concept_propagation"].numpy().astype("float32"),
                    p2_mean_feature_affinity=relations["p2_mean_feature_affinity"].numpy().astype("float32"),
                    p2_query_feature_affinity=relations["p2_query_feature_affinity"].numpy().astype("float32"),
                    p1_patch_mass_in_full_attention=relations["p1_patch_mass_in_full_attention"].numpy().astype("float32"),
                    query_patch_index=query_index,
                )

                for h, items in enumerate(head_topk):
                    if items:
                        head_concept_counts[h][items[0]["concept"]] += 1
                        concept_counts[items[0]["concept"]] += 1

                diagnostics = {
                    "image_id": image_id,
                    "file_name": image_record.get("file_name"),
                    "resolved_image_path": str(image_path.resolve()),
                    "captions": captions,
                    "annotation_ids": [int(x) if x is not None else None for x in annotation_ids],
                    "caption_concepts": caption_concepts,
                    "caption_concept_match_audit": caption_match,
                    "heads": heads_meta,
                    "head_topk": head_topk,
                    "head_gallery_cosine_matrix_shape": list(head_scores.shape),
                    "p1_patch_mass_in_full_attention": [
                        float(x) for x in relations["p1_patch_mass_in_full_attention"].tolist()
                    ],
                    "query_patch_xy": [qx, qy],
                    "query_patch_index": query_index,
                    "cached_recomputed_head_cosine": (
                        [round(float(x), 7) for x in cached_cosines]
                        if cached_cosines is not None else None
                    ),
                    "cache_warning": cache_warning,
                    "interpretation_warning": (
                        "Gallery Top-1 is a model retrieval candidate, not proof that the concept "
                        "is present in the image. Heatmaps are diagnostic relations, not segmentation masks."
                    ),
                }
                (image_out / "diagnostics.json").write_text(_json(diagnostics), encoding="utf-8")
                successes.append(diagnostics)
                print(
                    f"  caption concepts={len(caption_concepts)}, "
                    f"head0={head_topk[0][0]['concept']} ({head_topk[0][0]['cosine']:.3f})",
                    flush=True,
                )
                if cache_warning:
                    print("  [WARNING] " + cache_warning, flush=True)

            except Exception as exc:
                error = {
                    "image_id": image_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                print(f"  [ERROR] {image_id}: {exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
    finally:
        extractor.close()

    (args.output / "matches.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in successes),
        encoding="utf-8",
    )
    (args.output / "errors.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in errors),
        encoding="utf-8",
    )
    summary = {
        "num_selected": len(selected),
        "num_success": len(successes),
        "num_errors": len(errors),
        "gallery_size": len(gallery),
        "caption_concepts_total": sum(len(x["caption_concepts"]) for x in successes),
        "head_top1_concept_frequency": dict(concept_counts.most_common()),
        "head_top1_concept_frequency_by_head": [
            dict(counter.most_common()) for counter in head_concept_counts
        ],
        "num_cache_warnings": sum(bool(x["cache_warning"]) for x in successes),
        "output": str(args.output.resolve()),
    }
    (args.output / "summary.json").write_text(_json(summary), encoding="utf-8")
    write_html_gallery(args.output, successes)
    print(f"[DONE] success={len(successes)} errors={len(errors)} output={args.output}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
