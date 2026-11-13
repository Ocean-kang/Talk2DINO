#!/usr/bin/env python3
"""Build exactly-five-caption RA-C2C training data for Talk2DINO."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
from .core import compose_one_image
from .io_utils import load_base_train, load_gallery, load_source, write_json, write_jsonl

def parse_source(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use SOURCE_NAME=/path/to/manifest.json or .jsonl")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Source name and path must be non-empty")
    return name.strip().lower(), path.strip()

def parse_args():
    p = argparse.ArgumentParser(description="RA-C2C deterministic caption composition")
    p.add_argument("--base-train", default="outputs/coco2014_b14/train.pth")
    p.add_argument("--concept-source", action="append", type=parse_source, required=True)
    p.add_argument("--gallery", default="experiments/local_cocu/MSCOCO14_train_gallery.json")
    p.add_argument("--output-dir", default="outputs/role_aware_concept_composition/run1")
    p.add_argument("--clip-checkpoint", default=None,
                   help="Existing local OpenAI CLIP ViT-B/16 checkpoint .pt. Required except --plan-only.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="First N images; 0 means all.")
    p.add_argument("--missing-policy", choices=("error", "keep_raw"), default="error")
    p.add_argument("--plan-only", action="store_true",
                   help="Generate captions and audit without loading CLIP or writing train.pth.")
    p.add_argument("--beam-width", type=int, default=32)
    p.add_argument("--max-options", type=int, default=5)
    p.add_argument("--min-action-support", type=int, default=1)
    p.add_argument("--min-relation-support", type=int, default=2)
    p.add_argument("--coverage-weight", type=float, default=0.12)
    p.add_argument("--redundancy-weight", type=float, default=0.18)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Use a fresh run directory; "
            "the builder will not overwrite previous experiment outputs."
        )
    if not args.plan_only:
        if not args.clip_checkpoint:
            raise ValueError("Full build requires --clip-checkpoint with an existing local ViT-B/16 .pt. No download is attempted.")
        if not Path(args.clip_checkpoint).is_file():
            raise FileNotFoundError(f"Local CLIP checkpoint does not exist: {args.clip_checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base, image_ids, grouped = load_base_train(args.base_train)
    if args.limit > 0:
        image_ids = image_ids[:args.limit]
    sources, source_paths = {}, {}
    for name, path in args.concept_source:
        if name in sources:
            raise ValueError(f"Duplicate source name: {name}")
        sources[name] = load_source(path, name)
        source_paths[name] = path
    gallery = load_gallery(args.gallery) if args.gallery and Path(args.gallery).is_file() else []
    missing_by_source = {name: sum(iid not in rows for iid in image_ids) for name, rows in sources.items()}
    if args.missing_policy == "error":
        for name, count in missing_by_source.items():
            if count:
                raise ValueError(
                    f"Source '{name}' misses {count}/{len(image_ids)} image IDs. "
                    "Use same-split manifests or --missing-policy keep_raw."
                )

    config = {
        "beam_width": args.beam_width, "max_options": args.max_options,
        "min_action_support": args.min_action_support,
        "min_relation_support": args.min_relation_support,
        "coverage_weight": args.coverage_weight,
        "redundancy_weight": args.redundancy_weight,
    }
    generated, audit_rows = [], []
    counts = Counter()
    output_annotations, selected_images = [], []
    image_lookup = {int(im["id"]): im for im in base["images"]}
    for pos, iid in enumerate(image_ids):
        raw_anns = grouped[iid][:5]
        raw_caps = [str(a["caption"]) for a in raw_anns]
        image_sources = {name: rows.get(iid, []) for name, rows in sources.items()}
        if all(not rows for rows in image_sources.values()) and args.missing_policy == "keep_raw":
            result = {
                "image_id": iid, "raw_captions": raw_caps, "captions": raw_caps,
                "roles": [
                    {"role": f"C{i+1}", "assigned_role": "raw_caption_fallback",
                     "caption": raw_caps[i], "concepts": [], "score": 0.0,
                     "fallback": True, "fallback_reason": "missing_candidate_source",
                     "main_concept": None}
                    for i in range(5)
                ],
                "composition_score": 0.0, "candidate_pool": [], "diagnostics": {},
            }
        else:
            result = compose_one_image(iid, raw_caps, image_sources, gallery, config)
        generated.append({
            "image_id": iid, "raw_captions": raw_caps,
            "captions": result["captions"], "roles": result["roles"],
            "composition_score": result["composition_score"],
        })
        audit_rows.append(result)
        for role in result["roles"]:
            counts[role["assigned_role"]] += 1
            if role["fallback"]:
                counts["fallback_count"] += 1
        if not args.plan_only:
            selected_images.append(image_lookup[iid])
            for ann, caption, role in zip(raw_anns, result["captions"], result["roles"]):
                output_annotations.append(dict(
                    ann, caption=caption,
                    ra_c2c_role=role["role"],
                    ra_c2c_assigned_role=role["assigned_role"],
                    ra_c2c_concepts=role["concepts"],
                    ra_c2c_fallback=role["fallback"],
                    ra_c2c_fallback_reason=role["fallback_reason"],
                ))
        if (pos + 1) % 1000 == 0:
            print(f"[PLAN] {pos+1}/{len(image_ids)} images", flush=True)

    write_json(out_dir / "generated_captions.json", {
        "meta": {
            "method": "RA-C2C", "version": "0.1.0",
            "image_count": len(generated), "captions_per_image": 5,
            "sources": source_paths, "missing_by_source": missing_by_source,
            "config": config, "llm_used": False,
        },
        "images": generated,
    })
    write_jsonl(out_dir / "composition_audit.jsonl", audit_rows)
    summary = {
        "method": "RA-C2C", "image_count": len(generated),
        "annotation_count": len(output_annotations) if not args.plan_only else None,
        "captions_per_image": 5, "plan_only": args.plan_only,
        "missing_by_source": missing_by_source,
        "assignment_counts": dict(counts),
        "fallback_rate": counts["fallback_count"] / max(1, len(generated)*5),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)

    if not args.plan_only:
        ckpt = Path(args.clip_checkpoint)
        import torch
        import clip
        print(f"[CLIP] Loading local checkpoint: {ckpt}", flush=True)
        model, _ = clip.load(str(ckpt.resolve()), device=args.device, jit=False)
        model.eval().requires_grad_(False)
        texts = [a["caption"] for a in output_annotations]
        features = []
        with torch.inference_mode():
            for start in range(0, len(texts), args.batch_size):
                batch = texts[start:start+args.batch_size]
                tokens = clip.tokenize(batch, truncate=True).to(args.device)
                features.append(model.encode_text(tokens).detach().float().cpu())
                if (start // args.batch_size) % 100 == 0:
                    print(f"[CLIP] {min(start+args.batch_size, len(texts))}/{len(texts)}", flush=True)
        features = torch.cat(features, dim=0)
        if tuple(features.shape) != (len(output_annotations), 512):
            raise RuntimeError(f"Expected CLIP features [{len(output_annotations)},512], got {tuple(features.shape)}")
        for ann, feat in zip(output_annotations, features):
            ann["ann_feats"] = feat
        out = dict(base)
        out["images"] = selected_images
        out["annotations"] = output_annotations
        out["ra_c2c_meta"] = {
            "method": "RA-C2C", "version": "0.1.0",
            "source_manifests": source_paths, "captions_per_image": 5,
            "student_reinitialized": True, "teacher_used_during_training": False,
            "clip_checkpoint": str(ckpt.resolve()), "config": config,
        }
        train_path = out_dir / "train_ra_c2c.pth"
        torch.save(out, train_path)
        summary["train_pth"] = str(train_path)
        summary["annotation_count"] = len(output_annotations)
        write_json(out_dir / "summary.json", summary)
        del model
        print(f"[SAVED] {train_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
