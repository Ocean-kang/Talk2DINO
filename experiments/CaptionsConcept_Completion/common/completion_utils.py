#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_IRREGULAR = {
    "men": "man",
    "women": "woman",
    "people": "person",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
}


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def manifest_index(payload: dict) -> Dict[int, dict]:
    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError("Expected manifest format: {'meta': ..., 'images': [...]}")
    out = {}
    for rec in payload["images"]:
        image_id = int(rec["image_id"])
        if image_id in out:
            raise ValueError(f"Duplicate image_id in manifest: {image_id}")
        out[image_id] = rec
    return out


def prepare_base(base: dict, keep_original: int = 5):
    if "images" not in base or "annotations" not in base:
        raise KeyError("Base train.pth must contain 'images' and 'annotations'.")

    image_ids = [int(x["id"]) for x in base["images"]]
    grouped = defaultdict(list)
    for ann in base["annotations"]:
        grouped[int(ann["image_id"])].append(ann)

    missing = [iid for iid in image_ids if len(grouped[iid]) < keep_original]
    if missing:
        raise RuntimeError(
            f"{len(missing)} images have fewer than {keep_original} annotations. "
            f"First: {missing[:20]}"
        )

    missing_text = []
    missing_feat = []
    for iid in image_ids:
        for ann in grouped[iid][:keep_original]:
            if "caption" not in ann:
                missing_text.append(ann.get("id"))
            if "ann_feats" not in ann:
                missing_feat.append(ann.get("id"))
    if missing_text:
        raise RuntimeError(f"Raw annotations without caption. First ids: {missing_text[:20]}")
    if missing_feat:
        raise RuntimeError(f"Raw annotations without ann_feats. First ids: {missing_feat[:20]}")

    return image_ids, grouped


def validate_manifest_against_base(
    image_ids: Sequence[int],
    grouped,
    manifest_by_id: Dict[int, dict],
    keep_original: int = 5,
):
    base_set = set(image_ids)
    mani_set = set(manifest_by_id)
    missing = sorted(base_set - mani_set)
    extra = sorted(mani_set - base_set)
    if missing or extra:
        raise RuntimeError(
            "Manifest/base image-id mismatch. "
            f"missing_in_manifest={len(missing)}, extra_in_manifest={len(extra)}, "
            f"first_missing={missing[:10]}, first_extra={extra[:10]}"
        )

    caption_mismatch = []
    for iid in image_ids:
        raw = [str(x["caption"]) for x in grouped[iid][:keep_original]]
        mani = [str(x) for x in manifest_by_id[iid].get("original_captions", [])]
        if len(mani) != keep_original:
            caption_mismatch.append((iid, "manifest_count", len(mani)))
            continue
        norm = lambda s: " ".join(s.strip().split())
        if [norm(x) for x in raw] != [norm(x) for x in mani]:
            caption_mismatch.append((iid, "text_mismatch", None))
        if len(caption_mismatch) >= 20:
            break
    if caption_mismatch:
        raise RuntimeError(
            "Manifest raw captions do not match the first five base annotations. "
            f"First mismatches: {caption_mismatch}"
        )


def concept_items(record: dict) -> List[dict]:
    out = []
    seen = set()
    for item in record.get("curated", []):
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept", "")).strip()
        if not concept:
            continue
        key = concept.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "concept": concept,
            "score": item.get("score"),
            "rank": item.get("rank"),
            "best_head": item.get("best_head"),
        })
    return out


def _singularize(tok: str) -> str:
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("ses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def normalized_tokens(text: str) -> List[str]:
    return [_singularize(x) for x in _TOKEN_RE.findall(text.lower())]


def lexically_redundant(concept: str, raw_captions: Sequence[str]) -> bool:
    c = normalized_tokens(concept)
    if not c:
        return True
    for caption in raw_captions:
        t = normalized_tokens(caption)
        if len(c) == 1 and c[0] in t:
            return True
        if len(c) > 1:
            for i in range(0, len(t) - len(c) + 1):
                if t[i:i + len(c)] == c:
                    return True
    return False


def list_phrase(items: Sequence[str]) -> str:
    items = [x.strip() for x in items if x and x.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def enrich_caption(raw: str, concepts: Sequence[str]) -> str:
    concepts = [x for x in concepts if x]
    if not concepts:
        return raw
    raw = raw.strip()
    raw = raw.rstrip(" .!?;:") + "."
    noun = "element is" if len(concepts) == 1 else "elements include"
    return f"{raw} Additional visible {noun} {list_phrase(concepts)}."


def supplementary_caption(anchor: str, concepts: Sequence[str]) -> str:
    anchor = anchor.strip().rstrip(" .!?;:") + "."
    return (
        f"{anchor} Additional visible elements in the scene include "
        f"{list_phrase(concepts)}."
    )


class CLIPTextEncoder:
    def __init__(self, model_name: str = "ViT-B/16", device: str = "cuda"):
        try:
            import clip
        except ImportError as e:
            raise RuntimeError(
                "The Talk2DINO environment must provide the OpenAI CLIP package "
                "used by the repository (import clip)."
            ) from e
        self.clip = clip
        self.device = device
        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 512) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 512))
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            tokens = self.clip.tokenize(batch, truncate=True).to(self.device)
            feats = self.model.encode_text(tokens).cpu()
            chunks.append(feats)
        return torch.cat(chunks, dim=0)


def normalized(feats: torch.Tensor) -> torch.Tensor:
    feats = feats.float()
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def match_concepts_to_raw(
    raw_anns: Sequence[dict],
    concept_feats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_feats = torch.stack([x["ann_feats"].detach().cpu() for x in raw_anns])
    sim = normalized(concept_feats) @ normalized(raw_feats).T
    best_sim, best_idx = sim.max(dim=1)
    return best_idx, best_sim


def relative_visual_scores(items: Sequence[dict]) -> List[float]:
    vals = []
    for x in items:
        s = x.get("score")
        vals.append(None if s is None else float(s))
    finite = [x for x in vals if x is not None and math.isfinite(x)]
    if not finite:
        return [1.0] * len(items)
    max_score = max(finite)
    min_score = min(finite)
    if max_score <= 0:
        # Rank-based fallback for non-positive scores.
        return [1.0 - (i / max(1, len(items))) * 0.2 for i in range(len(items))]
    return [
        1.0 if x is None else max(0.0, float(x) / max_score)
        for x in vals
    ]


def residual_filter(
    raw_captions: Sequence[str],
    items: Sequence[dict],
    concept_raw_sim: torch.Tensor,
    min_visual_ratio: float = 0.85,
    semantic_floor: float = 0.18,
    consistency_margin: float = 0.20,
    redundancy_ceiling: float = 0.90,
    max_concepts: int = 3,
):
    if len(items) == 0:
        return [], []

    max_sims = concept_raw_sim.max(dim=1).values.float().tolist()
    best_semantic = max(max_sims) if max_sims else -1.0
    visual = relative_visual_scores(items)

    kept = []
    audit = []
    for i, item in enumerate(items):
        sim = float(max_sims[i])
        v = float(visual[i])
        lex_red = lexically_redundant(item["concept"], raw_captions)
        reasons = []
        if v < min_visual_ratio:
            reasons.append("low_visual_confidence")
        if lex_red:
            reasons.append("lexically_redundant")
        if sim < semantic_floor:
            reasons.append("semantic_floor")
        if sim < best_semantic - consistency_margin:
            reasons.append("semantic_outlier")
        if sim > redundancy_ceiling:
            reasons.append("semantic_redundancy")

        residual_score = v * max(0.0, sim) * max(0.0, 1.0 - sim)
        row = dict(item)
        row.update({
            "visual_ratio": v,
            "max_raw_similarity": sim,
            "residual_score": residual_score,
            "filter_reasons": reasons,
            "kept": len(reasons) == 0,
        })
        audit.append(row)
        if not reasons:
            kept.append(row)

    kept.sort(key=lambda x: x["residual_score"], reverse=True)
    kept = kept[:max_concepts]
    kept_keys = {x["concept"].lower() for x in kept}
    for row in audit:
        row["kept"] = row["concept"].lower() in kept_keys
        if len(row["filter_reasons"]) == 0 and not row["kept"]:
            row["filter_reasons"] = ["residual_rank_cut"]
    return kept, audit


def copy_raw_annotation(ann: dict, caption: str | None = None, feat: torch.Tensor | None = None):
    out = dict(ann)
    if caption is not None:
        out["caption"] = caption
    if feat is not None:
        out["ann_feats"] = feat
    return out


def write_jsonl(rows: Iterable[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
