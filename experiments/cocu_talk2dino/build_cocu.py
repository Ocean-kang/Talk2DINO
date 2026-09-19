#!/usr/bin/env python3
"""Build CoCu-style curated caption supervision for Talk2DINO without touching base code.

Two commands:
  init-pth  COCO captions JSON -> torch-loadable PTH for this Talk2DINO fork.
  build     MSCOCO train2014 + CoCu gallery -> side-car curated_annotations.pt.

The build path intentionally reuses OpenAI CLIP and PyTorch already required by
Talk2DINO. It does not require FAISS/NLTK/TextBlob/scikit-learn.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_IRREGULAR_PLURALS = {
    "person": "people", "man": "men", "woman": "women", "child": "children",
    "mouse": "mice", "goose": "geese", "tooth": "teeth", "foot": "feet",
}


def _load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_strings(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _flatten_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _flatten_strings(value)


def _pluralize_phrase(phrase: str) -> str:
    """Small dependency-free approximation of TextBlob's final-word pluralization."""
    words = phrase.split()
    if not words:
        return phrase
    word = words[-1]
    # ponytail: common English morphology is enough for COCO; install TextBlob and
    # swap this helper only if byte-for-byte lexical parity with original CoCu is needed.
    if word in _IRREGULAR_PLURALS:
        plural = _IRREGULAR_PLURALS[word]
    elif word.endswith(("s", "x", "z", "ch", "sh")):
        plural = word + "es"
    elif len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        plural = word[:-1] + "ies"
    elif word.endswith("fe"):
        plural = word[:-2] + "ves"
    elif word.endswith("f"):
        plural = word[:-1] + "ves"
    else:
        plural = word + "s"
    words[-1] = plural
    return " ".join(words)


def load_gallery(path: str | Path) -> list[str]:
    raw = _load_json(path)
    seen: dict[str, None] = {}
    for concept in _flatten_strings(raw):
        concept = " ".join(concept.lower().strip().split())
        if concept:
            seen.setdefault(concept, None)
    for concept in list(seen):
        seen.setdefault(_pluralize_phrase(concept), None)
    return list(seen)


def caption_concepts(caption: str, gallery: set[str], max_ngram: int = 4) -> list[str]:
    # ponytail: regex tokenization covers COCO captions and avoids an NLTK model download;
    # use nltk.word_tokenize if exact tokenizer parity becomes an experimental requirement.
    tokens = _TOKEN_RE.findall(caption.lower())
    found: list[str] = []
    used: set[str] = set()
    for n in range(1, min(max_ngram, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            if gram in gallery and gram not in used:
                found.append(gram)
                used.add(gram)
    return found


def cmd_init_pth(args):
    data = _load_json(args.captions_json)
    if "images" not in data or "annotations" not in data:
        raise ValueError("Expected COCO captions JSON with 'images' and 'annotations'.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f"Saved {len(data['images'])} images / {len(data['annotations'])} captions -> {out}")


def _resolve_image(images_dir: Path, item: dict) -> Path:
    p = images_dir / item["file_name"]
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


@torch.inference_mode()
def extract_image_features(model, preprocess, images: list[dict], images_dir: Path,
                           batch_size: int, device: str) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, len(images), batch_size), desc="CLIP image features"):
        batch_items = images[start:start + batch_size]
        batch = []
        for item in batch_items:
            with Image.open(_resolve_image(images_dir, item)) as im:
                batch.append(preprocess(im.convert("RGB")))
        x = torch.stack(batch).to(device, non_blocking=True)
        feat = model.encode_image(x).float()
        feat = F.normalize(feat, dim=-1)
        chunks.append(feat.cpu().half())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def extract_text_features(model, concepts: list[str], batch_size: int, device: str):
    import clip
    raw_chunks, norm_chunks = [], []
    for start in tqdm(range(0, len(concepts), batch_size), desc="CLIP concept features"):
        prompts = [f"a photo of a {x}" for x in concepts[start:start + batch_size]]
        tokens = clip.tokenize(prompts, truncate=True).to(device)
        raw = model.encode_text(tokens).float()
        raw_chunks.append(raw.cpu().half())
        norm_chunks.append(F.normalize(raw, dim=-1).cpu().half())
    return torch.cat(raw_chunks), torch.cat(norm_chunks)


@torch.inference_mode()
def exact_topk_neighbors(image_features: torch.Tensor, top_k: int, query_batch: int,
                         device: str, include_self: bool) -> torch.Tensor:
    """Exact cosine top-k in chunks; avoids a new FAISS dependency for COCO-sized data."""
    n = image_features.shape[0]
    wanted = top_k if include_self else top_k + 1
    if wanted >= n:
        raise ValueError(f"top_k={top_k} is too large for {n} images")
    # ponytail: exact O(N^2) retrieval is practical for COCO (~82k x 512) on a 4090;
    # switch to FAISS only when scaling to CC3M/CC12M-sized galleries.
    bank = image_features.float().to(device)
    result = torch.empty((n, top_k), dtype=torch.int32)
    for start in tqdm(range(0, n, query_batch), desc="Exact image-image retrieval"):
        end = min(start + query_batch, n)
        q = bank[start:end]
        sims = q @ bank.T
        if not include_self:
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            sims[rows, cols] = -torch.inf
            idx = sims.topk(top_k, dim=1).indices
        else:
            idx = sims.topk(wanted, dim=1).indices[:, :top_k]
        result[start:end] = idx.cpu().to(torch.int32)
        del sims, idx
    return result


def cosine_kmeans(x: torch.Tensor, k: int, priority: torch.Tensor, max_iter: int = 12) -> torch.Tensor:
    """Tiny deterministic spherical k-means returning a cluster id per row."""
    n = x.shape[0]
    if k >= n:
        return torch.arange(n)
    x = F.normalize(x.float(), dim=-1)
    centers = [int(priority.argmax())]
    while len(centers) < k:
        sims = x @ x[centers].T
        nearest = sims.max(dim=1).values
        nearest[centers] = 1.0
        centers.append(int(nearest.argmin()))
    centers_t = x[centers].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(max_iter):
        new_labels = (x @ centers_t.T).argmax(dim=1)
        if torch.equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        next_centers = centers_t.clone()
        for c in range(k):
            members = x[labels == c]
            if len(members):
                next_centers[c] = F.normalize(members.mean(dim=0), dim=0)
        centers_t = next_centers
    return labels


def select_diverse(cand_ids: list[int], scores: torch.Tensor, concept_features: torch.Tensor,
                   count: int) -> list[tuple[int, float]]:
    if not cand_ids:
        return []
    if len(cand_ids) <= count:
        order = scores.argsort(descending=True)
        return [(cand_ids[int(i)], float(scores[int(i)])) for i in order]
    x = concept_features[cand_ids].float()
    labels = cosine_kmeans(x, count, scores)
    picked: list[tuple[int, float]] = []
    for cluster in range(count):
        member_idx = torch.where(labels == cluster)[0]
        if len(member_idx) == 0:
            continue
        local = scores[member_idx]
        best_local = int(member_idx[int(local.argmax())])
        picked.append((cand_ids[best_local], float(scores[best_local])))
    picked.sort(key=lambda z: z[1], reverse=True)
    return picked


def build_image_semantics(data: dict, images: list[dict], gallery_set: set[str]):
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        captions_by_image[int(ann["image_id"])].append(ann["caption"])
    semantics: list[list[str]] = []
    for item in tqdm(images, desc="Caption -> gallery concepts"):
        used: dict[str, None] = {}
        for caption in captions_by_image[int(item["id"])]:
            for concept in caption_concepts(caption, gallery_set):
                used.setdefault(concept, None)
        semantics.append(list(used))
    return semantics


def score_archive(anchor_idx: int, neighbor_ids: list[int], semantics: list[list[str]],
                  image_features: torch.Tensor, concept_features: torch.Tensor,
                  concept2idx: dict[str, int], filter_anchor: bool):
    anchor_sem = set(semantics[anchor_idx])
    archive: dict[str, None] = {}
    for nidx in neighbor_ids:
        for concept in semantics[nidx]:
            if not filter_anchor or concept not in anchor_sem:
                archive.setdefault(concept, None)
    if not archive:
        return [], torch.empty(0)

    concepts = list(archive)
    cand_ids = [concept2idx[c] for c in concepts]
    cfeat = concept_features[cand_ids].float()
    refs = image_features[neighbor_ids + [anchor_idx]].float()
    sims = cfeat @ refs.T  # [concept, K + anchor]

    mask = torch.zeros_like(sims, dtype=torch.bool)
    for col, nidx in enumerate(neighbor_ids):
        sem = set(semantics[nidx])
        mask[:, col] = torch.tensor([c in sem for c in concepts])
    mask[:, -1] = True

    global_prob = sims[:, -1]
    denom = (sims * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    eps = 1e-6
    safe_denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    local_prob = global_prob / safe_denom
    masked = sims.masked_fill(~mask, -torch.inf)
    max_prob = masked.max()
    if not torch.isfinite(max_prob) or max_prob.abs() < eps:
        max_prob = torch.tensor(eps)
    score = global_prob / max_prob + local_prob
    return cand_ids, score


def cmd_build(args):
    import clip

    captions_json = Path(args.captions_json)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_json(captions_json)
    images = list(data["images"])
    if args.limit:
        images = images[:args.limit]
    valid_ids = {int(x["id"]) for x in images}
    data = dict(data)
    data["annotations"] = [x for x in data["annotations"] if int(x["image_id"]) in valid_ids]

    concepts = load_gallery(args.gallery)
    gallery_set = set(concepts)
    print(f"Images: {len(images):,}; gallery concepts incl. plurals: {len(concepts):,}")
    semantics = build_image_semantics(data, images, gallery_set)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    model, preprocess = clip.load(args.model, device=device, jit=False)
    model.eval()

    image_cache = out_dir / "clip_image_features.pt"
    if image_cache.exists() and not args.overwrite:
        cached = torch.load(image_cache, map_location="cpu")
        if cached["image_ids"] != [int(x["id"]) for x in images]:
            raise ValueError("Image cache does not match this COCO subset. Use --overwrite or a new out-dir.")
        image_features = cached["features"]
        print(f"Loaded {image_cache}")
    else:
        image_features = extract_image_features(model, preprocess, images, images_dir, args.image_batch, device)
        torch.save({"image_ids": [int(x["id"]) for x in images], "features": image_features}, image_cache)

    concept_cache = out_dir / "clip_concept_features.pt"
    if concept_cache.exists() and not args.overwrite:
        cached = torch.load(concept_cache, map_location="cpu")
        if cached["concepts"] != concepts:
            raise ValueError("Concept cache does not match gallery. Use --overwrite or a new out-dir.")
        concept_raw, concept_norm = cached["raw"], cached["normalized"]
        print(f"Loaded {concept_cache}")
    else:
        concept_raw, concept_norm = extract_text_features(model, concepts, args.text_batch, device)
        torch.save({"concepts": concepts, "raw": concept_raw, "normalized": concept_norm}, concept_cache)

    neighbors_cache = out_dir / f"neighbors_k{args.top_k}.pt"
    if neighbors_cache.exists() and not args.overwrite:
        cached = torch.load(neighbors_cache, map_location="cpu")
        neighbors = cached["indices"]
        print(f"Loaded {neighbors_cache}")
    else:
        neighbors = exact_topk_neighbors(image_features, args.top_k, args.retrieval_batch, device, args.include_self)
        torch.save({"indices": neighbors, "include_self": args.include_self}, neighbors_cache)

    concept2idx = {c: i for i, c in enumerate(concepts)}
    curated = []
    next_ann_id = max((int(a["id"]) for a in data["annotations"]), default=0) + 1
    inspect_path = out_dir / "curated_captions.jsonl"
    with open(inspect_path, "w", encoding="utf-8") as inspect:
        for image_idx, item in enumerate(tqdm(images, desc="CoCu scoring + diversity")):
            neigh = [int(x) for x in neighbors[image_idx].tolist() if int(x) != image_idx]
            cand_ids, scores = score_archive(
                image_idx, neigh, semantics, image_features, concept_norm,
                concept2idx, filter_anchor=not args.keep_existing,
            )
            picked = select_diverse(cand_ids, scores, concept_norm, args.num_curated)
            row = {"image_id": int(item["id"]), "curated": []}
            for cid, score in picked:
                concept = concepts[cid]
                caption = f"a photo of a {concept}"
                ann = {
                    "id": next_ann_id,
                    "image_id": int(item["id"]),
                    "caption": caption,
                    "concept": concept,
                    "cocu_score": score,
                    "source": "cocu",
                    "ann_feats": concept_raw[cid].clone(),
                }
                curated.append(ann)
                row["curated"].append({"concept": concept, "caption": caption, "score": score})
                next_ann_id += 1
            inspect.write(json.dumps(row, ensure_ascii=False) + "\n")

    sidecar = {
        "annotations": curated,
        "meta": {
            "method": "CoCu-Talk2DINO adaptation",
            "captions_json": str(captions_json),
            "gallery": str(args.gallery),
            "clip_model": args.model,
            "top_k": args.top_k,
            "num_curated": args.num_curated,
            "include_self": args.include_self,
            "filter_anchor_existing_concepts": not args.keep_existing,
            "notes": [
                "MSCOCO image-level semantics are the union of concepts from all captions for an image.",
                "Retrieval is exact chunked cosine in PyTorch instead of FAISS.",
                "Diverse concepts are deterministic top-score-per-cluster for static Talk2DINO text features.",
            ],
        },
    }
    out_file = out_dir / "curated_annotations.pt"
    torch.save(sidecar, out_file)
    per_image = len(curated) / max(1, len(images))
    print(f"Saved {len(curated):,} curated annotations ({per_image:.2f}/image) -> {out_file}")
    print(f"Human-readable inspection -> {inspect_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-pth", help="Convert COCO captions JSON to torch PTH for current dino_extraction_v2.py")
    p.add_argument("--captions-json", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_init_pth)

    p = sub.add_parser("build", help="Build CoCu curated side-car annotations")
    p.add_argument("--captions-json", required=True)
    p.add_argument("--images-dir", required=True, help="Directory containing COCO train2014 images")
    p.add_argument("--gallery", required=True, help="rewrite/rewrite/gallery.json from the CoCu repo")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="ViT-B/16")
    p.add_argument("--top-k", type=int, default=8, help="Number of external visual neighbors")
    p.add_argument("--num-curated", type=int, default=3)
    p.add_argument("--image-batch", type=int, default=128)
    p.add_argument("--text-batch", type=int, default=512)
    p.add_argument("--retrieval-batch", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="Smoke-test on first N images; 0 means all")
    p.add_argument("--include-self", action="store_true", help="Include anchor in retrieval, closer to same-index CoCu behavior")
    p.add_argument("--keep-existing", action="store_true", help="Do not remove concepts already in the anchor's original captions")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_build)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
