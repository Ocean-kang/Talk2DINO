#!/usr/bin/env python3
"""
Strict 5+3 Talk2DINO dataset builder.

Inputs in experiments/bootstrap_5plus3/:
  clip_vitb16.json
  talk2dino_image_cls_imagelevel.json
  local_cocu_compare.json

Output:
  outputs/bootstrap_5plus3/train_*_5plus3.pth

Important:
- Base train.pth may contain 5, 6, ... captions for an image.
- This experiment ALWAYS keeps exactly the first 5 original annotations
  in base-PTH order, then appends exactly 3 curated captions.
- Original ann_feats / DINO features are reused.
- Only the 3 newly added captions are encoded by CLIP text encoder.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
import clip


HERE = Path(__file__).resolve().parent

JSON_DEFAULTS = {
    "clip": HERE / "clip_vitb16.json",
    "image": HERE / "talk2dino_image_cls_imagelevel.json",
    "local": HERE / "local_cocu_compare.json",
}

OUT_NAMES = {
    "clip": "train_clip_vitb16_5plus3.pth",
    "image": "train_talk2dino_image_cls_imagelevel_5plus3.pth",
    "local": "train_local_cocu_compare_5plus3.pth",
}

METHOD_NAMES = {
    "clip": "clip_vitb16",
    "image": "talk2dino_image_cls_imagelevel",
    "local": "local_cocu_compare",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        choices=["clip", "image", "local", "all"],
        default="all",
    )
    p.add_argument(
        "--base-train",
        default="outputs/coco2014_b14/train.pth",
    )
    p.add_argument(
        "--clip-json",
        default=str(JSON_DEFAULTS["clip"]),
    )
    p.add_argument(
        "--image-json",
        default=str(JSON_DEFAULTS["image"]),
    )
    p.add_argument(
        "--local-json",
        default=str(JSON_DEFAULTS["local"]),
    )
    p.add_argument(
        "--output-dir",
        default="outputs/bootstrap_5plus3",
    )
    p.add_argument("--num-curated", type=int, default=3)
    p.add_argument("--template", default="a photo of a {}")
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve_json(path):
    path = Path(path)
    if path.exists():
        return path

    # Also support files named without the .json extension.
    if path.suffix == ".json":
        no_suffix = path.with_suffix("")
        if no_suffix.exists():
            return no_suffix

    raise FileNotFoundError(f"JSON file not found: {path}")


def load_json(path):
    path = resolve_json(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_concept(item):
    if isinstance(item, str):
        concept = item.strip()
        if not concept:
            return None
        return {
            "concept": concept,
            "score": None,
            "best_head": None,
        }

    if not isinstance(item, dict):
        return None

    concept = item.get("concept")
    if concept is None:
        return None

    concept = str(concept).strip()
    if not concept:
        return None

    return {
        "concept": concept,
        "score": item.get("score"),
        "best_head": item.get("best_head"),
    }


def rank_dedup(items):
    items = [normalize_concept(x) for x in items]
    items = [x for x in items if x is not None]

    # Source JSONs are already ranked, but sort by score when available.
    if any(x["score"] is not None for x in items):
        items = sorted(
            enumerate(items),
            key=lambda pair: (
                float("-inf")
                if pair[1]["score"] is None
                else float(pair[1]["score"]),
                -pair[0],
            ),
            reverse=True,
        )
        items = [x for _, x in items]

    out = []
    seen = set()

    for item in items:
        key = item["concept"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def parse_global_json(payload, num_curated):
    """
    clip_vitb16 / talk2dino_image_cls_imagelevel

    Priority:
      new_curated_concepts
      -> if <3, supplement from curated_concepts
    """
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError(
            "Expected format: {'meta': ..., 'images': [...]}"
        )

    result = {}
    fallback_count = 0

    for record in payload["images"]:
        image_id = int(record["image_id"])

        new_items = rank_dedup(
            record.get("new_curated_concepts", [])
        )
        curated_items = rank_dedup(
            record.get("curated_concepts", [])
        )

        chosen = []
        seen = set()

        for item in new_items:
            key = item["concept"].lower()
            if key not in seen:
                chosen.append(item)
                seen.add(key)
            if len(chosen) == num_curated:
                break

        if len(chosen) < num_curated:
            fallback_count += 1

            for item in curated_items:
                key = item["concept"].lower()
                if key not in seen:
                    chosen.append(item)
                    seen.add(key)
                if len(chosen) == num_curated:
                    break

        result[image_id] = chosen

    return result, fallback_count


def parse_local_json(payload, num_curated):
    """
    local_cocu_compare

    Use:
      concept_head_scores
      -> score descending
      -> unique concepts
      -> Top-3
    """
    if not isinstance(payload, dict):
        raise ValueError(
            "Expected local_cocu_compare top-level object keyed by image id."
        )

    result = {}

    for record in payload.values():
        if not isinstance(record, dict):
            continue
        if "image_id" not in record:
            continue

        image_id = int(record["image_id"])

        pool = record.get("concept_head_scores")
        if not pool:
            pool = record.get("curated_concepts", [])

        result[image_id] = rank_dedup(pool)[:num_curated]

    return result, 0


def prepare_base(base, keep_original=5):
    """
    Base PTH can contain >5 captions/image.

    For strict 5+3:
      - require >=5 original annotations/image
      - preserve base PTH annotation order
      - keep exactly first 5/image
    """
    if "images" not in base or "annotations" not in base:
        raise KeyError(
            "base train.pth must contain images and annotations"
        )

    image_ids = [int(x["id"]) for x in base["images"]]
    image_id_set = set(image_ids)

    grouped = defaultdict(list)

    for ann in base["annotations"]:
        if "ann_feats" not in ann:
            raise RuntimeError(
                f"annotation id={ann.get('id')} has no ann_feats"
            )
        grouped[int(ann["image_id"])].append(ann)

    missing_dino = [
        int(image["id"])
        for image in base["images"]
        if "disentangled_self_attn" not in image
    ]
    if missing_dino:
        raise RuntimeError(
            f"{len(missing_dino)} images have no "
            "disentangled_self_attn. "
            f"First: {missing_dino[:20]}"
        )

    too_few = [
        (image_id, len(grouped[image_id]))
        for image_id in image_ids
        if len(grouped[image_id]) < keep_original
    ]
    if too_few:
        raise RuntimeError(
            f"{len(too_few)} images contain fewer than "
            f"{keep_original} original captions. "
            f"First: {too_few[:20]}"
        )

    extra = [
        (image_id, len(grouped[image_id]))
        for image_id in image_ids
        if len(grouped[image_id]) > keep_original
    ]

    if extra:
        extra_annotations = sum(
            n - keep_original for _, n in extra
        )
        print(
            f"[info] {len(extra)} images contain >{keep_original} "
            f"captions ({extra_annotations} extra annotations)."
        )
        print(
            f"[info] Strict 5+3: keep only the first "
            f"{keep_original} original captions/image."
        )
        print(
            f"[info] First extra-caption images: {extra[:20]}"
        )

    selected_by_image = {}
    selected_annotations = []

    for image_id in image_ids:
        selected = grouped[image_id][:keep_original]
        selected_by_image[image_id] = selected
        selected_annotations.extend(selected)

    expected = len(image_ids) * keep_original

    if len(selected_annotations) != expected:
        raise AssertionError(
            f"Selected-original count mismatch: "
            f"{len(selected_annotations)} != {expected}"
        )

    print(
        f"[PASS] base raw annotations      : "
        f"{len(base['annotations']):,}"
    )
    print(
        f"[PASS] selected original captions: "
        f"{len(selected_annotations):,} "
        f"(exactly {keep_original}/image)"
    )

    return (
        image_ids,
        image_id_set,
        selected_by_image,
        selected_annotations,
    )


def make_curated_rows(
    method,
    curated_map,
    image_ids,
    num_curated,
    template,
):
    rows = []
    bad = []

    for image_id in image_ids:
        items = rank_dedup(
            curated_map.get(image_id, [])
        )

        if len(items) < num_curated:
            bad.append((image_id, len(items)))
            continue

        for rank, item in enumerate(
            items[:num_curated],
            start=1,
        ):
            rows.append({
                "image_id": image_id,
                "concept": item["concept"],
                "caption": template.format(
                    item["concept"]
                ),
                "score": item["score"],
                "best_head": item["best_head"],
                "rank": rank,
                "source": METHOD_NAMES[method],
            })

    if bad:
        raise RuntimeError(
            f"Strict 5+3 failed: {len(bad)} images have "
            f"<{num_curated} curated concepts. "
            f"First: {bad[:20]}"
        )

    expected = len(image_ids) * num_curated

    if len(rows) != expected:
        raise AssertionError(
            f"Curated-row count mismatch: "
            f"{len(rows)} != {expected}"
        )

    return rows


@torch.inference_mode()
def encode_rows(
    rows,
    model,
    batch_size,
    device,
):
    for start in tqdm(
        range(0, len(rows), batch_size),
        desc="CLIP text encoding",
    ):
        batch = rows[start:start + batch_size]

        tokens = clip.tokenize(
            [x["caption"] for x in batch],
            truncate=True,
        ).to(device)

        feats = model.encode_text(tokens).cpu()

        for row, feat in zip(batch, feats):
            row["ann_feats"] = feat


def build_method(
    method,
    json_path,
    base,
    image_ids,
    selected_by_image,
    selected_original_annotations,
    clip_model,
    args,
):
    print()
    print("=" * 80)
    print(f"METHOD: {METHOD_NAMES[method]}")
    print(f"JSON  : {json_path}")
    print("=" * 80)

    payload = load_json(json_path)

    if method in {"clip", "image"}:
        curated_map, fallback_count = (
            parse_global_json(
                payload,
                args.num_curated,
            )
        )
    else:
        curated_map, fallback_count = (
            parse_local_json(
                payload,
                args.num_curated,
            )
        )

    rows = make_curated_rows(
        method=method,
        curated_map=curated_map,
        image_ids=image_ids,
        num_curated=args.num_curated,
        template=args.template,
    )

    encode_rows(
        rows=rows,
        model=clip_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    # IMPORTANT:
    # start from exactly 5 selected originals/image,
    # NOT from all base annotations.
    annotations = list(selected_original_annotations)

    max_ann_id = max(
        int(x["id"])
        for x in base["annotations"]
    )
    next_ann_id = max_ann_id + 1

    rows_by_image = defaultdict(list)

    for row in rows:
        rows_by_image[row["image_id"]].append(row)

        ann = {
            "id": next_ann_id,
            "image_id": row["image_id"],
            "caption": row["caption"],
            "ann_feats": row["ann_feats"],
            "bootstrap_source": row["source"],
            "bootstrap_concept": row["concept"],
            "bootstrap_rank": row["rank"],
        }

        if row["score"] is not None:
            ann["bootstrap_score"] = float(
                row["score"]
            )

        if row["best_head"] is not None:
            ann["bootstrap_best_head"] = int(
                row["best_head"]
            )

        annotations.append(ann)
        next_ann_id += 1

    expected_per_image = (
        5 + args.num_curated
    )
    expected_total = (
        len(image_ids) * expected_per_image
    )

    if len(annotations) != expected_total:
        raise AssertionError(
            f"Total annotation mismatch: "
            f"{len(annotations)} != {expected_total}"
        )

    # Hard final check: every image must be 8 samples.
    counts = defaultdict(int)

    for ann in annotations:
        counts[int(ann["image_id"])] += 1

    bad_counts = [
        (image_id, counts[image_id])
        for image_id in image_ids
        if counts[image_id] != expected_per_image
    ]

    if bad_counts:
        raise AssertionError(
            f"Output is not strict 5+3. "
            f"First bad counts: {bad_counts[:20]}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_pth = (
        out_dir / OUT_NAMES[method]
    )
    manifest_path = (
        out_dir
        / f"{METHOD_NAMES[method]}_5plus3_manifest.json"
    )

    out = dict(base)
    out["annotations"] = annotations
    out["bootstrap_5plus3_meta"] = {
        "method": METHOD_NAMES[method],
        "source_json": str(
            resolve_json(json_path)
        ),
        "base_train": args.base_train,
        "original_per_image": 5,
        "curated_per_image": args.num_curated,
        "samples_per_image": expected_per_image,
        "caption_template": args.template,
        "clip_text_model": args.clip_model,
        "num_images": len(image_ids),
        "num_annotations": len(annotations),
        "fallback_to_curated_concepts_images":
            fallback_count,
    }

    torch.save(out, out_pth)

    manifest = {
        "meta": out["bootstrap_5plus3_meta"],
        "images": [],
    }

    for image_id in image_ids:
        manifest["images"].append({
            "image_id": image_id,
            "original_captions": [
                x["caption"]
                for x in selected_by_image[image_id]
            ],
            "curated": [
                {
                    k: v
                    for k, v in {
                        "concept": x["concept"],
                        "caption": x["caption"],
                        "score": x["score"],
                        "best_head": x["best_head"],
                        "rank": x["rank"],
                    }.items()
                    if v is not None
                }
                for x in rows_by_image[image_id]
            ],
        })

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[PASS] {len(image_ids):,} images × "
        f"{expected_per_image} samples/image "
        f"= {len(annotations):,}"
    )
    print(
        f"[PASS] min/max samples per image: "
        f"{min(counts.values())}/"
        f"{max(counts.values())}"
    )
    print(f"[SAVE] {out_pth}")
    print(f"[SAVE] {manifest_path}")

    if method in {"clip", "image"}:
        print(
            f"[info] fallback "
            f"new_curated_concepts -> curated_concepts: "
            f"{fallback_count} images"
        )


def main():
    args = parse_args()

    if args.num_curated != 3:
        print(
            f"[warning] requested experiment is 5+3, "
            f"but --num-curated={args.num_curated}"
        )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but CUDA is unavailable."
        )

    print(
        f"Loading ORIGINAL Talk2DINO train file: "
        f"{args.base_train}"
    )

    base = torch.load(
        args.base_train,
        map_location="cpu",
    )

    (
        image_ids,
        _,
        selected_by_image,
        selected_original_annotations,
    ) = prepare_base(
        base,
        keep_original=5,
    )

    print(
        f"Loading CLIP text encoder "
        f"{args.clip_model} ..."
    )

    clip_model, _ = clip.load(
        args.clip_model,
        device=args.device,
        jit=False,
    )
    clip_model.eval()

    paths = {
        "clip": args.clip_json,
        "image": args.image_json,
        "local": args.local_json,
    }

    methods = (
        ["clip", "image", "local"]
        if args.method == "all"
        else [args.method]
    )

    for method in methods:
        build_method(
            method=method,
            json_path=paths[method],
            base=base,
            image_ids=image_ids,
            selected_by_image=selected_by_image,
            selected_original_annotations=(
                selected_original_annotations
            ),
            clip_model=clip_model,
            args=args,
        )

    print()
    print(
        "All requested bootstrap_5plus3 "
        "datasets built successfully."
    )


if __name__ == "__main__":
    main()
