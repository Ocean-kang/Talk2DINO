#!/usr/bin/env python3
"""Reuse existing Local-CoCu concepts to prepare Talk2DINO training PTH.

5plus1: preserve five original captions/features, add ONE concept-composition caption.
5: assign EACH new concept to the semantically farthest of five original CLIP captions,
   then re-encode ONLY captions whose text changes (no extra training instances).
No new concept mining or modification of the original Talk2DINO code is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn.functional as F


def norm_tokens(value: str) -> tuple[str, ...]:
    """Small lexical guard, deliberately independent from the original package."""
    mapping = {"men": "man", "women": "woman", "people": "person", "children": "child",
               "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth"}
    words = []
    for word in re.findall(r"[a-z0-9]+", str(value).lower()):
        word = mapping.get(word, word)
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ous")):
            word = word[:-1]
        words.append(word)
    return tuple(words)


def already_mentioned(concept: str, captions: list[str]) -> bool:
    words = norm_tokens(concept)
    if not words:
        return True
    for caption in captions:
        hay = norm_tokens(caption)
        if any(hay[i:i+len(words)] == words for i in range(len(hay)-len(words)+1)):
            return True
    return False


def choose_novel(rows: Any, raw: list[str], maximum: int) -> list[dict]:
    """Highest source scores first, stable ties; remove duplicates and explicit mentions."""
    if isinstance(rows, dict):
        rows = rows.get("concepts", rows.get("matched", []))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list of concepts, got {type(rows).__name__}")
    ranked = []
    for i, item in enumerate(rows):
        entry = {"concept": item} if isinstance(item, str) else item
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid concept row {item!r}")
        concept = str(entry.get("concept", "")).strip()
        if not concept:
            continue
        try:
            score = float(entry.get("score", 0.0))
        except (ValueError, TypeError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        ranked.append(((-score, i), concept, score, entry))
    ranked.sort(key=lambda x: x[0])
    result, seen = [], set()
    for _, concept, score, item in ranked:
        key = norm_tokens(concept)
        if not key or key in seen or already_mentioned(concept, raw):
            continue
        seen.add(key)
        result.append({"concept": concept, "score": score,
                       "best_head": item.get("best_head"), "source": item})
        if len(result) >= maximum:
            break
    return result


def combine_concepts(concepts: list[str]) -> str:
    """One new CLIP-compatible sentence, not one sentence per concept."""
    if not concepts:
        raise ValueError("Cannot compose a concept caption from an empty list")
    if len(concepts) == 1:
        combined = concepts[0]
    elif len(concepts) == 2:
        combined = f"{concepts[0]} and {concepts[1]}"
    else:
        combined = ", ".join(concepts[:-1]) + f", and {concepts[-1]}"
    return f"A photo showing {combined}."


def append_concepts(caption: str, concepts: list[str]) -> str:
    if not concepts:
        return caption
    details = ", ".join(concepts)
    return f"{caption.rstrip().rstrip('.!?')}. Also visible in the image: {details}."


def farthest_caption(caption_vectors: torch.Tensor, concept_vector: torch.Tensor) -> tuple[int, list[float]]:
    """argmin_i cos(CLIP(raw caption_i), CLIP('a photo of a {concept}'))."""
    raw = F.normalize(caption_vectors.float(), dim=-1)
    concept = F.normalize(concept_vector.float().reshape(-1), dim=0)
    similarities = raw @ concept
    return int(similarities.argmin().item()), [float(x) for x in similarities.tolist()]


def load_source(path: Path) -> dict[int, dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Source concepts JSON must be image_id keyed dictionary")
    result = {}
    for key, value in obj.items():
        if not str(key).isdigit():
            continue  # top-level metadata, if present
        image_id = int(key)
        if isinstance(value, dict):
            if "image_id" in value and int(value["image_id"]) != image_id:
                raise ValueError(f"Image ID mismatch at {key}")
            result[image_id] = value
        elif isinstance(value, list):
            result[image_id] = {"concepts": value}
        else:
            raise ValueError(f"Unexpected entry at image {key}: {type(value).__name__}")
    if not result:
        raise ValueError(f"No image concepts found in {path}")
    return result


@torch.no_grad()
def embed(model, clip_module, sentences: list[str], device: str, batch_size: int) -> torch.Tensor:
    if not sentences:
        return torch.empty((0, 512), dtype=torch.float32)
    chunks = []
    for i in range(0, len(sentences), batch_size):
        t = clip_module.tokenize(sentences[i:i + batch_size], truncate=True).to(device)
        chunks.append(model.encode_text(t).detach().float().cpu())
        if (i // batch_size) % 100 == 0:
            print(f"[CLIP] {min(i + batch_size, len(sentences))}/{len(sentences)}", flush=True)
    return torch.cat(chunks, dim=0)


def plan_structure(base: dict, source: dict[int, dict], max_concepts: int,
                   missing_policy: str) -> tuple[list[dict], list[dict], list[dict], Counter]:
    """Plan edits without GPU. Returns five raw ann per image and per-image decisions."""
    images = base["images"]
    ids = [int(im["id"]) for im in images]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs in base train.pth")
    group: dict[int, list] = defaultdict(list)
    for ann in base["annotations"]:
        group[int(ann["image_id"])].append(ann)
    missing = set(ids) - set(source)
    if missing and missing_policy == "error":
        raise ValueError(f"{len(missing)} training image IDs lack extracted concepts, e.g. {sorted(missing)[:10]}. "
                         "Use --missing-policy keep_raw/subset explicitly.")
    selected_images, base_annotations, plans = [], [], []
    histogram = Counter()
    for im in images:
        iid = int(im["id"])
        if iid not in source and missing_policy == "subset":
            continue
        original = group[iid][:5]
        if len(original) != 5 or any("caption" not in ann or "ann_feats" not in ann for ann in original):
            raise ValueError(f"Image {iid} must have >=5 original caption/ann_feats entries")
        if "disentangled_self_attn" not in im:
            raise ValueError(f"Image {iid} lacks disentangled_self_attn")
        raw = [str(x["caption"]) for x in original]
        content = source.get(iid, {})
        concepts = choose_novel(content.get("concepts", []), raw, max_concepts)
        histogram[len(concepts)] += 1
        selected_images.append(im)
        base_annotations.extend(original)
        plans.append({"image_id": iid, "raw_captions": raw,
                      "selected_concepts": concepts,
                      "raw_ann_ids": [int(a["id"]) for a in original]})
    if not selected_images:
        raise ValueError("No train images remain")
    return selected_images, base_annotations, plans, histogram


def build(base_path: Path, concepts_path: Path, out_path: Path, manifest_path: Path,
          mode: str, maximum: int, clip_model: str, device: str, batch_size: int,
          missing_policy: str) -> dict:
    if mode not in ("5plus1", "5"):
        raise ValueError(mode)
    if maximum < 1 or batch_size < 1:
        raise ValueError("max_concepts and batch_size must be positive")
    print("[INPUT]", base_path, concepts_path, flush=True)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    source = load_source(concepts_path)
    images, raw_annotations, plans, hist = plan_structure(base, source, maximum, missing_policy)
    if not sum(k * n for k, n in hist.items()):
        raise RuntimeError("No novel concepts survived; refusing to train an identical-to-baseline model")

    import clip  # use precisely the repository's CLIP ViT-B/16 implementation
    model, _ = clip.load(clip_model, device=device, jit=False)
    model.eval()
    new_annotations = list(raw_annotations)
    next_id = max(int(a["id"]) for a in base["annotations"]) + 1
    audit = {}
    changed_rows = []  # references to newly created/modified annotations
    changed_texts = []

    # Only mode=5 requires concept-caption semantic similarity.
    concept_embeddings = {}
    if mode == "5":
        vocabulary = sorted({r["concept"] for p in plans for r in p["selected_concepts"]})
        features = embed(model, clip, [f"a photo of a {c}" for c in vocabulary], device, batch_size)
        concept_embeddings = {c: feat for c, feat in zip(vocabulary, features)}

    for pos, p in enumerate(plans):
        iid = p["image_id"]
        selected = p["selected_concepts"]
        raw = p["raw_captions"]
        augmented = list(raw)
        assignments = []
        ids = p["raw_ann_ids"]
        if mode == "5plus1":
            if selected:
                caption = combine_concepts([r["concept"] for r in selected])
                row = {"id": next_id, "image_id": iid, "caption": caption,
                       "cocu_new_train_mode": "5plus1",
                       "cocu_new_concepts": [r["concept"] for r in selected]}
                new_annotations.append(row)
                changed_rows.append(row)
                changed_texts.append(caption)
                augmented.append(caption)
                next_id += 1
        else:
            records = raw_annotations[pos * 5:pos * 5 + 5]
            base_vectors = torch.stack([torch.as_tensor(a["ann_feats"]).float().reshape(-1)
                                        for a in records])
            concept_groups = [[] for _ in range(5)]
            for r in selected:
                concept = r["concept"]
                index, sims = farthest_caption(base_vectors, concept_embeddings[concept])
                concept_groups[index].append(concept)
                assignments.append({"concept": concept, "caption_index": index,
                                    "similarities_to_raw_captions": [round(x, 6) for x in sims],
                                    "min_similarity": round(sims[index], 6)})
            for j, additions in enumerate(concept_groups):
                if additions:
                    caption = append_concepts(raw[j], additions)
                    row = dict(records[j])  # never mutate original base tensor/cache
                    row["caption"] = caption
                    row["cocu_original_caption"] = raw[j]
                    row["cocu_new_concepts"] = additions
                    row["cocu_new_train_mode"] = "5"
                    new_annotations[pos * 5 + j] = row
                    changed_rows.append(row)
                    changed_texts.append(caption)
                    augmented[j] = caption
        audit[str(iid)] = {"image_id": iid, "raw_captions": raw,
                           "selected_concepts": [{"concept": r["concept"], "score": r["score"],
                                                   "best_head": r["best_head"]} for r in selected],
                           "augmented_captions": augmented, "assignments": assignments,
                           "annotation_ids": ids + ([next_id - 1] if mode == "5plus1" and selected else [])}
        if (pos + 1) % 10000 == 0:
            print(f"[PLAN] {pos+1}/{len(plans)}", flush=True)

    features = embed(model, clip, changed_texts, device, batch_size)
    for row, feat in zip(changed_rows, features):
        row["ann_feats"] = feat
    del model

    counts = Counter(int(a["image_id"]) for a in new_annotations)
    expected = set(int(x["id"]) for x in images)
    if set(counts) != expected:
        raise AssertionError("Output image / annotation ID coverage mismatch")
    if mode == "5" and any(v != 5 for v in counts.values()):
        raise AssertionError("5-caption mode must have exactly five annotations per image")
    if mode == "5plus1" and any(v not in (5, 6) for v in counts.values()):
        raise AssertionError("5+1 mode must have 5 or 6 annotations (6 when >=1 novel concept)")
    out = dict(base)
    out["images"] = images
    out["annotations"] = new_annotations
    metadata = {"mode": mode, "policy": "max concept-caption CLIP cosine distance" if mode == "5" else "5 raw + 1 composed concept caption",
                "max_concepts_per_image": maximum, "concepts_source": str(concepts_path),
                "clip_model": clip_model, "student_new_initialization": True,
                "teacher_used_during_training": False, "image_count": len(images),
                "annotation_count": len(new_annotations), "reencoded_caption_count": len(changed_rows),
                "histogram_novel_concepts": dict(sorted(hist.items())),
                "images_without_novel_concepts": hist[0], "missing_policy": missing_policy}
    out["local_cocu_new_train_meta"] = metadata
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"meta": metadata, "images": audit},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SAVED]", out_path, "images=", len(images), "annotations=", len(new_annotations), flush=True)
    print("[SAVED]", manifest_path, flush=True)
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-train", default="outputs/coco2014_b14/train.pth")
    p.add_argument("--concepts", required=True, help="Existing concepts_by_image.json OR matched_concepts_detailed.json")
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--mode", required=True, choices=["5plus1", "5"])
    p.add_argument("--max-concepts", type=int, default=3)
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--missing-policy", choices=["error", "keep_raw", "subset"], default="error")
    args = p.parse_args()
    build(Path(args.base_train), Path(args.concepts), Path(args.output), Path(args.manifest),
          args.mode, args.max_concepts, args.clip_model, args.device, args.batch_size, args.missing_policy)


if __name__ == "__main__":
    main()
