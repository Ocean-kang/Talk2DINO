#!/usr/bin/env python3
"""Create 5 raw + up to K novel Local-CoCu captions, compatible with train.py.

Original image/head features and the first five raw CLIP text features are kept
byte-for-byte as tensor objects. Only supplemental captions need CLIP encoding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
from tqdm import tqdm

from filter_novel import in_raw_captions, norm_tokens


def select_novel(items, original, maximum, exact_only=False):
    """Recheck against EXACT five captions used in the training PTH (guardrail)."""
    chosen, seen = [], set()
    ranked = sorted(
        enumerate(items),
        key=lambda pair: (float(pair[1].get("score", float("-inf"))), -pair[0]),
        reverse=True,
    )
    for _, item in ranked:
        if not isinstance(item, dict):
            raise ValueError(f"Expected detailed match dict, got {type(item)!r}")
        concept = str(item.get("concept", "")).strip()
        key = norm_tokens(concept, exact_only)
        if not key or key in seen:
            continue
        seen.add(key)
        if in_raw_captions(concept, original, exact_only=exact_only):
            continue
        chosen.append(item)
        if len(chosen) == maximum:
            break
    return chosen


def plan(base: dict, matches: dict, maximum: int, exact_only=False, strict=False):
    if maximum < 1:
        raise ValueError("--max-novel must be at least 1")
    if "images" not in base or "annotations" not in base:
        raise ValueError("Base train.pth must contain images and annotations")
    images = base["images"]
    ids = [int(img["id"]) for img in images]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image ids in base PTH")
    grouped = defaultdict(list)
    for ann in base["annotations"]:
        grouped[int(ann["image_id"])].append(ann)
    raw_annotations, supplemental = [], []
    histogram = Counter()
    example_deficits = []
    for im in images:
        image_id = int(im["id"])
        if "disentangled_self_attn" not in im:
            raise KeyError(f"Missing disentangled_self_attn for image {image_id}")
        original = grouped[image_id][:5]
        if len(original) < 5:
            raise ValueError(f"Image {image_id} has {len(original)} raw annotations, expected >=5")
        for ann in original:
            if "ann_feats" not in ann or "caption" not in ann:
                raise KeyError(f"Image {image_id} lacks caption or ann_feats")
        if str(image_id) not in matches:
            raise KeyError(f"Image {image_id} missing from detailed matches; check extraction ID coverage")
        original_texts = [str(ann["caption"]) for ann in original]
        chosen = select_novel(matches[str(image_id)], original_texts, maximum, exact_only)
        histogram[len(chosen)] += 1
        if len(chosen) < maximum and len(example_deficits) < 30:
            example_deficits.append([image_id, len(chosen)])
        raw_annotations.extend(original)
        for rank, entry in enumerate(chosen, 1):
            concept = str(entry["concept"]).strip()
            supplemental.append({
                "image_id": image_id, "concept": concept, "rank": rank,
                "score": float(entry["score"]),
                "best_head": int(entry["best_head"]),
                "caption": f"a photo of a {concept}",
            })
    if strict and histogram.get(maximum, 0) != len(images):
        raise RuntimeError(f"Strict 5+{maximum} violated. Histogram: {dict(histogram)}; examples: {example_deficits}")
    if not supplemental:
        raise RuntimeError("No novel concepts survived for any image; refusing a no-op augmented run")
    return raw_annotations, supplemental, histogram, example_deficits


@torch.no_grad()
def encode_supplemental(supplemental, device, model_name, batch_size):
    try:
        import clip
    except ImportError as e:
        raise ImportError("Install the original Talk2DINO OpenAI-CLIP package before building features") from e
    model, _ = clip.load(model_name, device=device, jit=False)
    model.eval()
    for start in tqdm(range(0, len(supplemental), batch_size), desc="CLIP text embeddings"):
        chunk = supplemental[start:start + batch_size]
        tokens = clip.tokenize([r["caption"] for r in chunk], truncate=True).to(device)
        features = model.encode_text(tokens).detach().float().cpu()
        for rec, feat in zip(chunk, features):
            rec["ann_feats"] = feat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-train", default="outputs/coco2014_b14/train.pth")
    p.add_argument("--matched", required=True, help="matched_concepts_detailed.json")
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--max-novel", type=int, default=3)
    p.add_argument("--strict", action="store_true", help="Require K truly novel concepts for EVERY image")
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch-size must be >=1")

    print("[LOAD]", a.base_train, flush=True)
    base = torch.load(a.base_train, map_location="cpu")
    matches = json.loads(Path(a.matched).read_text(encoding="utf-8"))
    original, supplemental, hist, deficits = plan(
        base, matches, a.max_novel, a.exact_only, a.strict)
    print("[PLANNED] raw:", len(original), "novel:", len(supplemental), "histogram:", dict(hist), flush=True)
    encode_supplemental(supplemental, a.device, a.clip_model, a.batch_size)

    annotations = list(original)
    next_id = 1 + max(int(ann["id"]) for ann in base["annotations"])
    counts = Counter(int(ann["image_id"]) for ann in original)
    by_image = defaultdict(list)
    for rec in supplemental:
        row = {
            "id": next_id,
            "image_id": rec["image_id"],
            "caption": rec["caption"],
            "ann_feats": rec["ann_feats"],
            "novel_cocu_concept": rec["concept"],
            "novel_cocu_score": rec["score"],
            "novel_cocu_best_head": rec["best_head"],
            "novel_cocu_rank": rec["rank"],
        }
        annotations.append(row)
        by_image[str(rec["image_id"])].append({
            "concept": rec["concept"], "score": rec["score"],
            "best_head": rec["best_head"], "caption": rec["caption"],
        })
        counts[rec["image_id"]] += 1
        next_id += 1

    if min(counts.values()) < 5 or max(counts.values()) > 5 + a.max_novel:
        raise AssertionError("Corrupt per-image annotation count")
    out = dict(base)
    out["annotations"] = annotations
    out["local_cocu_novel_meta"] = {
        "policy": "first 5 original captions + up to K novel concepts; head matching before train",
        "max_novel": a.max_novel, "exact_only": a.exact_only,
        "num_images": len(base["images"]), "num_original": len(original),
        "num_novel": len(supplemental), "histogram_added": dict(sorted(hist.items())),
        "teacher_used_during_training": False,
        "clip_model": a.clip_model, "source_matches": a.matched,
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    manifest = Path(a.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "meta": out["local_cocu_novel_meta"],
        "examples_with_fewer_than_K": deficits,
        "curated_by_image": dict(by_image),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {output} images={len(base['images'])} raw={len(original)} novel={len(supplemental)}")
    print(f"[SAVED] {manifest}")


if __name__ == "__main__":
    main()
