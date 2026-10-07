#!/usr/bin/env python3
"""Scheme D: Concept-Set Supplementary Caption."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common.completion_utils import (  # noqa: E402
    CLIPTextEncoder,
    concept_items,
    copy_raw_annotation,
    load_json,
    manifest_index,
    match_concepts_to_raw,
    normalized,
    prepare_base,
    relative_visual_scores,
    save_json,
    supplementary_caption,
    validate_manifest_against_base,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["clip", "local_talk2dino"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--base-train", default="outputs/coco2014_b14/train.pth")
    p.add_argument("--output", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--image-batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-visual-ratio", type=float, default=0.85)
    p.add_argument("--max-concepts", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    payload = load_json(args.manifest)
    mani = manifest_index(payload)
    base = torch.load(args.base_train, map_location="cpu")
    image_ids, grouped = prepare_base(base, keep_original=5)
    validate_manifest_against_base(image_ids, grouped, mani, keep_original=5)

    encoder = CLIPTextEncoder(args.clip_model, args.device)
    original_annotations = []
    supplemental_rows = []
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    selected_hist = Counter()

    with audit_path.open("w", encoding="utf-8") as af:
        for start in tqdm(range(0, len(image_ids), args.image_batch_size), desc=f"Scheme D / {args.source}"):
            chunk_ids = image_ids[start:start + args.image_batch_size]
            prompt_texts = []
            chunk_meta = []
            for image_id in chunk_ids:
                raw_anns = grouped[image_id][:5]
                raw_caps = [str(x["caption"]) for x in raw_anns]
                for ann in raw_anns:
                    original_annotations.append(copy_raw_annotation(ann))
                items = concept_items(mani[image_id])
                if not items:
                    raise RuntimeError(f"No curated concepts for image {image_id}")
                visual = relative_visual_scores(items)
                selected = [x for x, vr in zip(items, visual) if vr >= args.min_visual_ratio][:args.max_concepts]
                fallback = False
                if not selected:
                    selected = [items[0]]
                    fallback = True
                selected_hist[len(selected)] += 1
                offset = len(prompt_texts)
                prompt_texts.extend([f"a photo of a {x['concept']}" for x in selected])
                chunk_meta.append((image_id, raw_anns, raw_caps, selected, visual, fallback, offset, len(selected)))

            all_cfeats = encoder.encode(prompt_texts, batch_size=args.batch_size)
            pending = []
            supplement_texts = []
            for image_id, raw_anns, raw_caps, selected, visual, fallback, offset, n_sel in chunk_meta:
                cfeats = all_cfeats[offset:offset + n_sel]
                centroid = normalized(cfeats).mean(dim=0, keepdim=True)
                best_idx, best_sim = match_concepts_to_raw(raw_anns, centroid)
                anchor_idx = int(best_idx[0])
                anchor_sim = float(best_sim[0])
                concepts = [x["concept"] for x in selected]
                text = supplementary_caption(raw_caps[anchor_idx], concepts)
                text_offset = len(supplement_texts)
                supplement_texts.append(text)
                pending.append((image_id, raw_caps, selected, visual, fallback, anchor_idx, anchor_sim, concepts, text, text_offset))

            all_sfeats = encoder.encode(supplement_texts, batch_size=args.batch_size)
            for image_id, raw_caps, selected, visual, fallback, anchor_idx, anchor_sim, concepts, text, text_offset in pending:
                feat = all_sfeats[text_offset]
                supplemental_rows.append({
                    "image_id": image_id, "caption": text, "ann_feats": feat,
                    "concepts": concepts, "anchor_index": anchor_idx,
                    "anchor_similarity": anchor_sim, "fallback": fallback,
                    "source_items": selected,
                })
                af.write(json.dumps({
                    "image_id": image_id,
                    "raw_captions": raw_caps,
                    "selected_concepts": selected,
                    "visual_ratios": visual,
                    "anchor_index": anchor_idx,
                    "anchor_similarity": anchor_sim,
                    "supplementary_caption": text,
                    "fallback": fallback,
                }, ensure_ascii=False) + "\n")

    max_ann_id = max(int(x["id"]) for x in base["annotations"])
    next_id = max_ann_id + 1
    out_annotations = list(original_annotations)
    for row in supplemental_rows:
        out_annotations.append({
            "id": next_id,
            "image_id": row["image_id"],
            "caption": row["caption"],
            "ann_feats": row["ann_feats"],
            "completion_scheme": "D_concept_set_supplementary",
            "completion_source": args.source,
            "completion_concepts": row["concepts"],
            "completion_anchor_index": row["anchor_index"],
            "completion_anchor_similarity": row["anchor_similarity"],
        })
        next_id += 1

    expected = len(image_ids) * 6
    if len(out_annotations) != expected:
        raise AssertionError(f"Scheme D must be strict 5+1: {len(out_annotations)} != {expected}")

    out = dict(base)
    out["annotations"] = out_annotations
    out["captions_concept_completion_meta"] = {
        "scheme": "D_concept_set_supplementary",
        "source": args.source,
        "manifest": args.manifest,
        "base_train": args.base_train,
        "raw_captions_per_image": 5,
        "supplementary_captions_per_image": 1,
        "samples_per_image": 6,
        "clip_text_model": args.clip_model,
        "min_visual_ratio": args.min_visual_ratio,
        "max_concepts": args.max_concepts,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    summary = {
        "scheme": "D_concept_set_supplementary",
        "source": args.source,
        "num_images": len(image_ids),
        "num_annotations": len(out_annotations),
        "samples_per_image": 6,
        "selected_concepts_per_image": dict(sorted(selected_hist.items())),
        "output": args.output,
        "audit": args.audit,
    }
    save_json(summary, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
