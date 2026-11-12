#!/usr/bin/env python3
"""Compare subset-5K caption-to-DINO-attention-head correspondence across four arms.

Run from the Talk2DINO repository root:
python -m experiments.Subset5K_Caption_Head_Comparison.run --help
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from experiments.Caption_Head_Visualization.data import resolve_image, torch_load_legacy
from experiments.Caption_Head_Visualization.vision import DinoHeadExtractor
from experiments.Caption_Head_Visualization.visualize import save_visualizations


ARMS = ("raw", "rich", "rich_weaken", "rich_diversity")
WEIGHT_FILES = {
    "raw": "vitb_mlp_infonce_subset5k_raw.pth",
    "rich": "vitb_mlp_infonce_subset5k_rich.pth",
    "rich_weaken": "vitb_mlp_infonce_subset5k_rich_weaken.pth",
    "rich_diversity": "vitb_mlp_infonce_subset5k_rich_diversity.pth",
}


def json_dump(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_projector(config_path: Path, weight_path: Path, device: str):
    """Load the original Talk2DINO ProjectionLayer and one trained checkpoint."""
    from src.model import ProjectionLayer

    projector = ProjectionLayer.from_config(str(config_path))
    checkpoint = torch_load_legacy(weight_path)

    if isinstance(checkpoint, dict):
        for wrapper_key in ("state_dict", "model_state_dict", "model", "projector"):
            candidate = checkpoint.get(wrapper_key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or a dict containing state_dict/model.")

    state = checkpoint
    # Strip only prefixes that are present on every key, avoiding accidental key edits.
    for prefix in ("module.", "projector.", "model."):
        if state and all(str(k).startswith(prefix) for k in state):
            state = {str(k)[len(prefix):]: v for k, v in state.items()}

    projector.load_state_dict(state, strict=True)
    projector = projector.to(device).eval().requires_grad_(False)
    if not hasattr(projector, "project_clip_txt"):
        raise AttributeError("Loaded ProjectionLayer has no project_clip_txt() method.")
    return projector


def work_dir_for(args, arm: str) -> Path:
    override = getattr(args, arm + "_work_dir")
    return Path(override) if override else Path(args.work_dir)


def load_arm_data(work_dir: Path, arm: str, split: str):
    """Merge arm-specific train/val COCO PTH files and validate five captions/image."""
    splits = ("train", "val") if split == "all" else (split,)
    image_records = {}
    annotation_groups = defaultdict(list)
    split_by_id = {}

    for split_name in splits:
        path = work_dir / "{}_{}.pth".format(arm, split_name)
        if not path.is_file():
            raise FileNotFoundError(
                "Missing {} feature file: {}\n"
                "Expected the 5K preprocessing output containing images[].disentangled_self_attn "
                "and annotations[].ann_feats.".format(arm, path)
            )
        data = torch_load_legacy(path)
        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            raise ValueError("Invalid COCO feature PTH: {}".format(path))

        for image in data["images"]:
            image_id = int(image["id"])
            if image_id in image_records:
                raise ValueError("Duplicate image_id={} across split files for arm={}".format(image_id, arm))
            image_records[image_id] = image
            split_by_id[image_id] = split_name

        for ann in data["annotations"]:
            annotation_groups[int(ann["image_id"])].append(ann)

    image_ids = sorted(image_records)
    if not image_ids:
        raise ValueError("No images found for arm={} under {}".format(arm, work_dir))

    captions = {}
    annotation_ids = {}
    text_features = {}
    for image_id in image_ids:
        rows = annotation_groups.get(image_id, [])
        rows = sorted(rows, key=lambda x: int(x.get("id", 0)))
        if len(rows) != 5:
            raise ValueError(
                "Expected exactly 5 captions for image_id={} arm={}, found {}.".format(
                    image_id, arm, len(rows)
                )
            )
        if any(not isinstance(row.get("caption"), str) or not row["caption"].strip() for row in rows):
            raise ValueError("Empty or invalid caption for image_id={} arm={}".format(image_id, arm))
        if any("ann_feats" not in row for row in rows):
            raise ValueError(
                "Missing ann_feats for image_id={} arm={}. "
                "Run the existing CLIP feature extraction/preprocessing first.".format(image_id, arm)
            )

        feats = torch.stack([
            torch.as_tensor(row["ann_feats"]).float().reshape(-1) for row in rows
        ])
        if feats.shape != (5, 512):
            raise ValueError(
                "Expected 5x512 CLIP ViT-B/16 features for image_id={} arm={}, got {}.".format(
                    image_id, arm, tuple(feats.shape)
                )
            )

        image = image_records[image_id]
        if "disentangled_self_attn" not in image:
            raise ValueError(
                "Missing images[].disentangled_self_attn for image_id={} arm={}.".format(image_id, arm)
            )
        head_features = torch.as_tensor(image["disentangled_self_attn"]).float()
        if head_features.shape != (12, 768):
            raise ValueError(
                "Expected [12,768] DINOv2 ViT-B/14-reg head features for image_id={}, got {}.".format(
                    image_id, tuple(head_features.shape)
                )
            )

        captions[image_id] = [row["caption"].strip() for row in rows]
        annotation_ids[image_id] = [row.get("id") for row in rows]
        text_features[image_id] = feats

    if set(annotation_groups) != set(image_records):
        missing = sorted(set(image_records) - set(annotation_groups))[:10]
        extra = sorted(set(annotation_groups) - set(image_records))[:10]
        raise ValueError(
            "Image/annotation ID mismatch for arm={}. Missing annotations={}, extra annotations={}.".format(
                arm, missing, extra
            )
        )

    return {
        "work_dir": str(work_dir.resolve()),
        "images": image_records,
        "image_ids": image_ids,
        "split_by_id": split_by_id,
        "captions": captions,
        "annotation_ids": annotation_ids,
        "text_features": text_features,
    }


def validate_four_arms(arm_data, allow_split_mismatch=False, min_head_cosine=0.999):
    reference = arm_data["raw"]
    reference_ids = set(reference["image_ids"])

    for arm in ARMS[1:]:
        current = arm_data[arm]
        current_ids = set(current["image_ids"])
        if current_ids != reference_ids:
            raise ValueError(
                "Image ID mismatch between raw and {}. raw-only={}, {}-only={}".format(
                    arm, sorted(reference_ids - current_ids)[:10], arm,
                    sorted(current_ids - reference_ids)[:10]
                )
            )

        if not allow_split_mismatch:
            mismatched = [
                image_id for image_id in reference["image_ids"]
                if reference["split_by_id"][image_id] != current["split_by_id"][image_id]
            ]
            if mismatched:
                raise ValueError(
                    "Train/val split differs between raw and {} (first IDs: {}). "
                    "Use --allow-split-mismatch only if you intentionally accept this.".format(
                        arm, mismatched[:10]
                    )
                )

    # Verify that every condition uses the same frozen DINO head features.
    min_cosines = {}
    for arm in ARMS:
        values = []
        for image_id in reference["image_ids"]:
            ref = torch.as_tensor(reference["images"][image_id]["disentangled_self_attn"]).float()
            cur = torch.as_tensor(arm_data[arm]["images"][image_id]["disentangled_self_attn"]).float()
            if cur.shape != ref.shape:
                raise ValueError("DINO head feature shape mismatch: arm={}, image_id={}".format(arm, image_id))
            values.append(F.cosine_similarity(ref, cur, dim=-1).min().item())
        min_cosines[arm] = float(min(values)) if values else 1.0
        if min_cosines[arm] < min_head_cosine:
            raise ValueError(
                "DINO head features differ for arm={} (minimum per-head cosine={:.6f}, threshold={:.6f}). "
                "Do not compare these arms as if they shared identical visual features. "
                "Inspect preprocessing/checkpoints or lower --min-head-cosine deliberately.".format(
                    arm, min_cosines[arm], min_head_cosine
                )
            )
    return min_cosines


def batched_caption_head_scores(projector, text_features, head_features, device):
    """Per-image cosine matrix [B,5,12], using the trained text projector."""
    text_features = text_features.to(device=device, dtype=torch.float32)
    head_features = head_features.to(device=device, dtype=torch.float32)
    batch_size, num_captions, clip_dim = text_features.shape
    if num_captions != 5 or clip_dim != 512 or head_features.shape[1:] != (12, 768):
        raise ValueError("Unexpected batch dimensions: text={}, heads={}".format(
            tuple(text_features.shape), tuple(head_features.shape)
        ))

    projected = projector.project_clip_txt(text_features.reshape(-1, clip_dim))
    if projected.shape[-1] != 768:
        raise ValueError("Projector output must be 768-D, got {}".format(projected.shape[-1]))
    projected = F.normalize(projected.float(), p=2, dim=-1).reshape(batch_size, num_captions, 768)
    heads = F.normalize(head_features.float(), p=2, dim=-1)
    similarities = torch.einsum("bcd,bhd->bch", projected, heads)

    top = similarities.topk(k=2, dim=-1)
    return {
        "similarities": similarities.detach().cpu(),
        "best_heads": top.indices[..., 0].detach().cpu(),
        "top1_scores": top.values[..., 0].detach().cpu(),
        "top2_heads": top.indices[..., 1].detach().cpu(),
        "top2_scores": top.values[..., 1].detach().cpu(),
        "margins": (top.values[..., 0] - top.values[..., 1]).detach().cpu(),
    }


def make_description(scores, captions, annotation_ids):
    """Create the same core schema consumed by Caption_Head_Visualization.save_visualizations."""
    similarities = scores["similarities"]
    best = scores["best_heads"].tolist()
    counts = [best.count(h) for h in range(12)]
    rows = []
    for index, caption in enumerate(captions):
        head_index = int(best[index])
        ann_id = annotation_ids[index]
        rows.append({
            "caption_index": index,
            "annotation_id": int(ann_id) if ann_id is not None else None,
            "caption": caption,
            "head_index": head_index,
            "head_number": head_index + 1,
            "cosine_similarity": round(float(scores["top1_scores"][index]), 8),
            "second_head_index": int(scores["top2_heads"][index]),
            "second_head_number": int(scores["top2_heads"][index]) + 1,
            "second_best_cosine": round(float(scores["top2_scores"][index]), 8),
            "top1_top2_margin": round(float(scores["margins"][index]), 8),
            "all_head_cosines": [round(float(v), 8) for v in similarities[index].tolist()],
        })
    return {
        "num_captions": len(captions),
        "num_heads": 12,
        "head_use_counts": counts,
        "unique_heads_used": sum(value > 0 for value in counts),
        "unused_head_numbers": [index + 1 for index, value in enumerate(counts) if value == 0],
        "mean_matching_margin": round(float(scores["margins"].mean()), 8),
        "mean_top1_cosine": round(float(scores["top1_scores"].mean()), 8),
        "captions": rows,
    }


def normalized_entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [value / total for value in counts if value > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(12)


def summarize_arm(arm, image_count, head_counts, top1_values, margin_values,
                  unique_per_image, per_slot_scores, per_slot_margins, per_slot_heads):
    caption_count = image_count * 5
    return {
        "arm": arm,
        "num_images": image_count,
        "num_captions": caption_count,
        "head_use_counts_1_to_12": [int(x) for x in head_counts],
        "head_use_ratio_1_to_12": [round(float(x) / max(1, caption_count), 8) for x in head_counts],
        "normalized_head_usage_entropy": round(normalized_entropy(head_counts), 8),
        "mean_unique_heads_per_image": round(sum(unique_per_image) / max(1, len(unique_per_image)), 8),
        "mean_top1_cosine": round(sum(top1_values) / max(1, len(top1_values)), 8),
        "std_top1_cosine": round(float(torch.tensor(top1_values).std(unbiased=False)), 8) if top1_values else 0.0,
        "mean_top1_top2_margin": round(sum(margin_values) / max(1, len(margin_values)), 8),
        "per_caption_slot": [
            {
                "slot": slot + 1,
                "mean_top1_cosine": round(sum(per_slot_scores[slot]) / max(1, len(per_slot_scores[slot])), 8),
                "mean_top1_top2_margin": round(sum(per_slot_margins[slot]) / max(1, len(per_slot_margins[slot])), 8),
                "head_use_counts_1_to_12": [int(x) for x in per_slot_heads[slot]],
            }
            for slot in range(5)
        ],
    }


def pairwise_head_set_overlap(head_sets):
    results = {}
    for i, arm_a in enumerate(ARMS):
        for arm_b in ARMS[i + 1:]:
            scores = []
            for image_id in head_sets[arm_a]:
                a = head_sets[arm_a][image_id]
                b = head_sets[arm_b][image_id]
                union = a | b
                scores.append(len(a & b) / len(union) if union else 1.0)
            results["{}__{}".format(arm_a, arm_b)] = {
                "mean_per_image_head_set_jaccard": round(sum(scores) / max(1, len(scores)), 8),
                "num_images": len(scores),
            }
    return results


def parse_int_list(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]



def write_html_gallery(output_dir: Path, image_ids, arms):
    sections = []

    for image_id in image_ids:
        cards = []

        for arm in arms:
            rel = "images/{}/{}/".format(image_id, arm)

            cards.append(
                '<article><h3>{}</h3>'
                '<a href="{}overview.png"><img src="{}overview.png" alt="{} overview"></a>'
                '<p><a href="{}similarity_matrix.png">Caption × 12-head cosine matrix</a> · '
                '<a href="{}all_12_heads.png">All 12 heads</a></p></article>'.format(
                    html.escape(arm), rel, rel,
                    html.escape(arm), rel, rel
                )
            )

        sections.append(
            '<section><h2>COCO image {}</h2><div class="grid">{}</div></section>'.format(
                image_id, "".join(cards)
            )
        )

    # 使用专用占位符，不再使用 str.format() 处理整个 HTML。
    # 这样 CSS 中的 {} 不会被 Python 误识别为格式化变量。
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subset-5K Caption–Head Comparison</title>

<style>
body {
    font-family: Arial, sans-serif;
    margin: 24px;
    background: #f5f7fa;
    color: #20252b;
}

h1 {
    margin-bottom: 8px;
}

.note {
    color: #56616d;
    margin-bottom: 24px;
}

section {
    background: white;
    border-radius: 12px;
    padding: 18px;
    margin: 22px 0;
    box-shadow: 0 1px 5px #0001;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
}

article {
    border: 1px solid #e4e8ed;
    border-radius: 8px;
    padding: 12px;
    min-width: 0;
}

article img {
    width: 100%;
    height: auto;
    border: 1px solid #e4e8ed;
}

a {
    color: #1769aa;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}
</style>

</head>
<body>

<h1>Subset-5K Caption–Attention Head Correspondence</h1>

<p class="note">
Each condition uses its own trained projector.
Heatmaps visualize DINO CLS-to-patch attention for
the head selected by caption-to-head cosine matching.
</p>

__GALLERY_SECTIONS__

</body>
</html>
"""

    # 只替换指定占位符，保留 CSS 原有花括号。
    page = page.replace(
        "__GALLERY_SECTIONS__",
        "".join(sections)
    )

    (output_dir / "index.html").write_text(
        page,
        encoding="utf-8"
    )



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/5k_rich_diversity/work"),
                        help="Default directory containing {arm}_{train,val}.pth for all arms.")
    for arm in ARMS:
        parser.add_argument("--{}-work-dir".format(arm.replace("_", "-")), type=Path, default=None,
                            help="Optional per-arm feature directory; overrides --work-dir.")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--config", type=Path, default=Path("configs/vitb_mlp_infonce.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/Subset5K_Caption_Head_Comparison"))
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Debug: number of sorted image IDs; 0 means all.")
    parser.add_argument("--batch-images", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-root", type=Path, default=Path("outputs/5k_rich_diversity/work/images"))
    parser.add_argument("--no-visualize", action="store_true",
                        help="Compute all caption/head matches and statistics without re-extracting spatial maps.")
    parser.add_argument("--viz-limit", type=int, default=20,
                        help="Number of first sorted images to visualize; 0 disables visualization.")
    parser.add_argument("--viz-image-ids", default="", help="Comma-separated IDs; overrides --viz-limit.")
    parser.add_argument("--attention-norm", choices=("per-head", "global"), default="per-head")
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--min-head-cosine", type=float, default=0.999)
    parser.add_argument("--allow-split-mismatch", action="store_true")
    parser.add_argument("--allow-head-mismatch", action="store_true",
                        help="Continue after detecting different cached DINO head features across arms.")
    parser.add_argument("--strict-cache", action="store_true",
                        help="Fail visualization if cached and freshly extracted DINO heads disagree.")
    parser.add_argument("--dinov2-repo", default="facebookresearch/dinov2")
    parser.add_argument("--hub-source", choices=("github", "local"), default="github")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.limit < 0 or args.batch_images <= 0 or args.viz_limit < 0:
        parser.error("--limit and --viz-limit must be >=0; --batch-images must be >0.")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0,1].")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu for a slower run.")
    return args


def main():
    args = parse_args()
    root = Path.cwd()
    config_path = (root / args.config).resolve()
    output_dir = (root / args.output).resolve()
    weights_dir = (root / args.weights_dir).resolve()

    if not config_path.is_file():
        raise FileNotFoundError("Missing Talk2DINO config: {}".format(config_path))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Output directory is not empty: {}. Choose another --output or pass --overwrite.".format(output_dir)
        )

    weight_paths = {arm: weights_dir / WEIGHT_FILES[arm] for arm in ARMS}
    for arm, path in weight_paths.items():
        if not path.is_file():
            raise FileNotFoundError("Missing {} projector checkpoint: {}".format(arm, path))

    arm_data = {}
    for arm in ARMS:
        work_dir = work_dir_for(args, arm).resolve()
        print("[LOAD] {} <- {}".format(arm, work_dir), flush=True)
        arm_data[arm] = load_arm_data(work_dir, arm, args.split)

    head_cosines = validate_four_arms(
        arm_data,
        allow_split_mismatch=args.allow_split_mismatch,
        min_head_cosine=args.min_head_cosine,
    )
    image_ids = arm_data["raw"]["image_ids"]
    if args.limit:
        image_ids = image_ids[:args.limit]
    if not image_ids:
        raise ValueError("No images selected.")

    # Use one canonical copy of the frozen DINO features for every arm.
    head_tensor_cpu = torch.stack([
        torch.as_tensor(arm_data["raw"]["images"][image_id]["disentangled_self_attn"]).float()
        for image_id in image_ids
    ])

    viz_ids = []
    if not args.no_visualize:
        explicit_viz_ids = parse_int_list(args.viz_image_ids)
        if explicit_viz_ids:
            missing = sorted(set(explicit_viz_ids) - set(image_ids))
            if missing:
                raise ValueError("Visualization IDs are not in selected feature set: {}".format(missing))
            viz_ids = list(dict.fromkeys(explicit_viz_ids))
        elif args.viz_limit:
            viz_ids = image_ids[:args.viz_limit]

        if viz_ids and not args.image_root.is_dir():
            raise NotADirectoryError(
                "Image root not found: {}. Use --image-root or --no-visualize.".format(args.image_root)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "Subset5K_Caption_Head_Comparison",
        "split": args.split,
        "num_images": len(image_ids),
        "image_ids": image_ids,
        "arms": list(ARMS),
        "weights": {
            arm: {
                "path": str(weight_paths[arm]),
                "sha256": sha256_file(weight_paths[arm]),
            } for arm in ARMS
        },
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "feature_dirs": {arm: arm_data[arm]["work_dir"] for arm in ARMS},
        "shared_dino_head_feature_min_cosine": head_cosines,
        "device": args.device,
        "batch_images": args.batch_images,
        "visualized_image_ids": viz_ids,
        "attention_map_note": "Heatmaps are DINO CLS-to-patch attention maps; captions select a head but do not generate text-conditioned spatial maps.",
    }
    (output_dir / "run_config.json").write_text(json_dump(config), encoding="utf-8")

    per_arm_summary = {}
    head_sets = {arm: {} for arm in ARMS}
    viz_cache = {arm: {} for arm in ARMS}
    per_image_rows = []

    for arm in ARMS:
        print("[MATCH] {}".format(arm), flush=True)
        projector = load_projector(config_path, weight_paths[arm], args.device)
        head_counts = [0] * 12
        top1_values, margin_values, unique_per_image = [], [], []
        per_slot_scores = [[] for _ in range(5)]
        per_slot_margins = [[] for _ in range(5)]
        per_slot_heads = [[0] * 12 for _ in range(5)]
        matches_path = output_dir / "matches_{}.jsonl".format(arm)

        with matches_path.open("w", encoding="utf-8") as matches_file, torch.no_grad():
            for start in range(0, len(image_ids), args.batch_images):
                batch_ids = image_ids[start:start + args.batch_images]
                text_batch = torch.stack([
                    arm_data[arm]["text_features"][image_id] for image_id in batch_ids
                ])
                head_batch = head_tensor_cpu[start:start + len(batch_ids)]
                batch_scores = batched_caption_head_scores(projector, text_batch, head_batch, args.device)

                for local_index, image_id in enumerate(batch_ids):
                    scores = {
                        key: value[local_index] for key, value in batch_scores.items()
                    }
                    captions = arm_data[arm]["captions"][image_id]
                    ann_ids = arm_data[arm]["annotation_ids"][image_id]
                    description = make_description(scores, captions, ann_ids)
                    description.update({"image_id": int(image_id), "arm": arm})

                    matches_file.write(json.dumps(description, ensure_ascii=False, allow_nan=False) + "\n")

                    best = scores["best_heads"].tolist()
                    head_set = set(int(x) for x in best)
                    head_sets[arm][image_id] = head_set
                    unique_per_image.append(len(head_set))
                    for caption_index, head_index in enumerate(best):
                        head_counts[int(head_index)] += 1
                        top1 = float(scores["top1_scores"][caption_index])
                        margin = float(scores["margins"][caption_index])
                        top1_values.append(top1)
                        margin_values.append(margin)
                        per_slot_scores[caption_index].append(top1)
                        per_slot_margins[caption_index].append(margin)
                        per_slot_heads[caption_index][int(head_index)] += 1

                    per_image_rows.append({
                        "image_id": int(image_id),
                        "arm": arm,
                        "num_unique_heads": len(head_set),
                        "mean_top1_cosine": round(float(scores["top1_scores"].mean()), 8),
                        "mean_top1_top2_margin": round(float(scores["margins"].mean()), 8),
                        "head_use_counts_1_to_12": json.dumps(description["head_use_counts"]),
                    })

                    if image_id in viz_ids:
                        viz_cache[arm][image_id] = {
                            "scores": scores,
                            "description": description,
                        }

        per_arm_summary[arm] = summarize_arm(
            arm, len(image_ids), head_counts, top1_values, margin_values, unique_per_image,
            per_slot_scores, per_slot_margins, per_slot_heads
        )
        del projector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cross_arm = {
        "head_set_jaccard": pairwise_head_set_overlap(head_sets),
        "interpretation": (
            "Per-image Jaccard compares the sets of heads selected by the five captions in two arms. "
            "It is a set-level overlap measure, not proof that caption slot i has the same meaning across arms."
        ),
    }

    summary = {
        "num_images": len(image_ids),
        "num_captions_per_arm": len(image_ids) * 5,
        "arms": per_arm_summary,
        "cross_arm_comparison": cross_arm,
        "shared_dino_head_feature_min_cosine": head_cosines,
    }
    (output_dir / "summary.json").write_text(json_dump(summary), encoding="utf-8")
    (output_dir / "cross_arm_comparison.json").write_text(json_dump(cross_arm), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arm", "num_images", "num_captions", "mean_unique_heads_per_image",
            "mean_top1_cosine", "std_top1_cosine", "mean_top1_top2_margin",
            "normalized_head_usage_entropy"
        ] + ["head_{}_count".format(i + 1) for i in range(12)])
        for arm in ARMS:
            item = per_arm_summary[arm]
            writer.writerow([
                arm, item["num_images"], item["num_captions"], item["mean_unique_heads_per_image"],
                item["mean_top1_cosine"], item["std_top1_cosine"], item["mean_top1_top2_margin"],
                item["normalized_head_usage_entropy"]
            ] + item["head_use_counts_1_to_12"])

    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "arm", "num_unique_heads", "mean_top1_cosine",
            "mean_top1_top2_margin", "head_use_counts_1_to_12"
        ])
        writer.writeheader()
        writer.writerows(per_image_rows)

    viz_errors = []
    if viz_ids:
        print("[VIS] Re-extracting DINO spatial attention for {} selected images.".format(len(viz_ids)), flush=True)
        extractor = DinoHeadExtractor(
            device=args.device, size=448, repo=args.dinov2_repo, hub_source=args.hub_source
        )
        try:
            for index, image_id in enumerate(viz_ids, start=1):
                try:
                    image_record = arm_data["raw"]["images"][image_id]
                    image_path = resolve_image(args.image_root, image_record)
                    with Image.open(image_path) as im:
                        original = im.convert("RGB").copy()

                    new_heads, attention_maps, crop = extractor.extract(original)
                    cached_heads = torch.as_tensor(
                        arm_data["raw"]["images"][image_id]["disentangled_self_attn"]
                    ).float()
                    cache_cosines = F.cosine_similarity(cached_heads, new_heads.float(), dim=-1)
                    min_cache_cosine = float(cache_cosines.min())
                    if min_cache_cosine < 0.99:
                        message = (
                            "Cached/re-extracted head feature mismatch for image_id={}: min cosine={:.6f}"
                            .format(image_id, min_cache_cosine)
                        )
                        if args.strict_cache:
                            raise ValueError(message)
                        print("[WARNING] " + message, file=sys.stderr, flush=True)

                    for arm in ARMS:
                        cached = viz_cache[arm][image_id]
                        description = copy.deepcopy(cached["description"])
                        description.update({
                            "image_file": str(image_path.resolve()),
                            "cached_recomputed_head_cosine": [round(float(x), 7) for x in cache_cosines],
                            "minimum_cached_recomputed_head_cosine": round(min_cache_cosine, 7),
                            "normalized_heatmap_is_semantic_mask": False,
                        })
                        arm_out = output_dir / "images" / str(image_id) / arm
                        save_visualizations(
                            crop, attention_maps, cached["scores"], description, arm_out,
                            alpha=args.alpha, norm_mode=args.attention_norm,
                            all_heads=True, full_resized=extractor.resize(original)
                        )
                        (arm_out / "diagnostics.json").write_text(
                            json_dump(description), encoding="utf-8"
                        )
                    print("[VIS] {}/{} image_id={}".format(index, len(viz_ids), image_id), flush=True)
                except Exception as exc:
                    error = {
                        "image_id": int(image_id),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    viz_errors.append(error)
                    print("[VIS-ERROR] image_id={} {}".format(image_id, exc), file=sys.stderr, flush=True)
                    if args.fail_fast:
                        raise
        finally:
            extractor.close()

        (output_dir / "visualization_errors.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in viz_errors),
            encoding="utf-8"
        )
        successful_viz_ids = [image_id for image_id in viz_ids if
                              (output_dir / "images" / str(image_id) / "raw" / "overview.png").is_file()]
        write_html_gallery(output_dir, successful_viz_ids, ARMS)

    print("[DONE] {}".format(output_dir), flush=True)
    print("[SUMMARY] {}".format(output_dir / "summary.json"), flush=True)
    if viz_ids:
        print("[GALLERY] {}".format(output_dir / "index.html"), flush=True)


if __name__ == "__main__":
    main()
