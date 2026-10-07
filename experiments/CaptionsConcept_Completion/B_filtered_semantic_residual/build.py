#!/usr/bin/env python3
"""Scheme B: Filtered Semantic Residual Completion."""
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
    enrich_caption,
    load_json,
    manifest_index,
    match_concepts_to_raw,
    normalized,
    prepare_base,
    residual_filter,
    save_json,
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
    p.add_argument("--semantic-floor", type=float, default=0.18)
    p.add_argument("--consistency-margin", type=float, default=0.20)
    p.add_argument("--redundancy-ceiling", type=float, default=0.90)
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
    out_annotations = []
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    keep_hist = Counter()
    reason_hist = Counter()
    changed_caption_count = 0

    with audit_path.open("w", encoding="utf-8") as af:
        for start in tqdm(range(0, len(image_ids), args.image_batch_size), desc=f"Scheme B / {args.source}"):
            chunk_ids = image_ids[start:start + args.image_batch_size]
            prompt_texts = []
            chunk_meta = []
            for image_id in chunk_ids:
                raw_anns = grouped[image_id][:5]
                raw_caps = [str(x["caption"]) for x in raw_anns]
                items = concept_items(mani[image_id])
                if not items:
                    raise RuntimeError(f"No curated concepts for image {image_id}")
                offset = len(prompt_texts)
                prompt_texts.extend([f"a photo of a {x['concept']}" for x in items])
                chunk_meta.append((image_id, raw_anns, raw_caps, items, offset, len(items)))

            all_cfeats = encoder.encode(prompt_texts, batch_size=args.batch_size)
            pending = []
            enriched_texts = []
            for image_id, raw_anns, raw_caps, items, offset, n_items in chunk_meta:
                cfeats = all_cfeats[offset:offset + n_items]
                raw_feats = torch.stack([x["ann_feats"].detach().cpu() for x in raw_anns])
                sim = normalized(cfeats) @ normalized(raw_feats).T
                kept, filter_audit = residual_filter(
                    raw_caps, items, sim,
                    min_visual_ratio=args.min_visual_ratio,
                    semantic_floor=args.semantic_floor,
                    consistency_margin=args.consistency_margin,
                    redundancy_ceiling=args.redundancy_ceiling,
                    max_concepts=args.max_concepts,
                )
                keep_hist[len(kept)] += 1
                for row in filter_audit:
                    for reason in row["filter_reasons"]:
                        reason_hist[reason] += 1

                kept_names = {x["concept"].lower() for x in kept}
                kept_positions = [i for i, x in enumerate(items) if x["concept"].lower() in kept_names]
                if kept_positions:
                    kept_feats = cfeats[kept_positions]
                    best_idx, best_sim = match_concepts_to_raw(raw_anns, kept_feats)
                else:
                    best_idx = torch.empty((0,), dtype=torch.long)
                    best_sim = torch.empty((0,), dtype=torch.float32)

                assigned = [[] for _ in range(5)]
                matching = []
                for pos, idx, m_sim in zip(kept_positions, best_idx.tolist(), best_sim.tolist()):
                    item = items[pos]
                    assigned[idx].append(item["concept"])
                    matching.append({"concept": item["concept"], "raw_index": int(idx), "similarity": float(m_sim)})
                enriched = [enrich_caption(raw_caps[i], assigned[i]) for i in range(5)]
                feat_offset = len(enriched_texts)
                enriched_texts.extend(enriched)
                pending.append((image_id, raw_anns, raw_caps, assigned, filter_audit, matching, enriched, feat_offset))

            all_efeats = encoder.encode(enriched_texts, batch_size=args.batch_size)
            for image_id, raw_anns, raw_caps, assigned, filter_audit, matching, enriched, feat_offset in pending:
                efeats = all_efeats[feat_offset:feat_offset + 5]
                for i, (ann, cap, feat) in enumerate(zip(raw_anns, enriched, efeats)):
                    new_ann = copy_raw_annotation(ann, caption=cap, feat=feat)
                    new_ann["completion_scheme"] = "B_filtered_semantic_residual"
                    new_ann["completion_source"] = args.source
                    new_ann["completion_concepts"] = assigned[i]
                    out_annotations.append(new_ann)
                    if assigned[i]:
                        changed_caption_count += 1
                af.write(json.dumps({
                    "image_id": image_id,
                    "raw_captions": raw_caps,
                    "filter": filter_audit,
                    "matching": matching,
                    "enriched_captions": enriched,
                }, ensure_ascii=False) + "\n")

    out = dict(base)
    out["annotations"] = out_annotations
    out["captions_concept_completion_meta"] = {
        "scheme": "B_filtered_semantic_residual",
        "source": args.source,
        "manifest": args.manifest,
        "base_train": args.base_train,
        "captions_per_image": 5,
        "clip_text_model": args.clip_model,
        "thresholds": {
            "min_visual_ratio": args.min_visual_ratio,
            "semantic_floor": args.semantic_floor,
            "consistency_margin": args.consistency_margin,
            "redundancy_ceiling": args.redundancy_ceiling,
            "max_concepts": args.max_concepts,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    summary = {
        "scheme": "B_filtered_semantic_residual",
        "source": args.source,
        "num_images": len(image_ids),
        "num_annotations": len(out_annotations),
        "captions_per_image": 5,
        "changed_caption_count": changed_caption_count,
        "kept_concepts_per_image": dict(sorted(keep_hist.items())),
        "filter_reason_counts": dict(reason_hist),
        "output": args.output,
        "audit": args.audit,
    }
    save_json(summary, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
