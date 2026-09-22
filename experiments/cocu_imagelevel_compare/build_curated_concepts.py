#!/usr/bin/env python3
"""
CoCu-style curated concept extraction on MSCOCO2014.

Backends
--------
1) clip
   CLIP image embedding <-> CLIP text embedding

2) talk2dino_image
   DINOv2 CLS image embedding <-> Projector(CLIP text embedding)

Design goals
------------
- New experiment only; does not modify Talk2DINO base code.
- Uses an exact gallery as-is: no plural expansion, stemming, lemmatization,
  NLTK tokenization, or TextBlob.
- Reproduces the core CoCu curation equation:
      score(c) = global(c) / max_prob + local(c)
      local(c) = sim(c, anchor) / mean(sim(c, reference_images + anchor))
- Source-like retrieval includes the anchor itself by default when the search
  bank is the same dataset. Use --exclude-self only for an ablation.
- COCO has multiple captions/image, so per-image source semantics are the union
  of exact gallery concepts found across all original captions.
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


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_phrase(text: str) -> str:
    # Lexical normalization only. No stemming/pluralization/linguistic tokenizer.
    return " ".join(_WORD_RE.findall(text.lower()))


def load_exact_gallery(path: str | Path) -> list[str]:
    raw = load_json(path)

    if isinstance(raw, dict) and isinstance(raw.get("concept"), list):
        values = raw["concept"]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(
            "Expected gallery JSON as {'concept': [...]} or a flat JSON list."
        )

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        concept = normalize_phrase(value)
        if concept and concept not in seen:
            seen.add(concept)
            out.append(concept)

    if not out:
        raise ValueError("Gallery is empty after normalization.")
    return out


def exact_caption_concepts(
    caption: str,
    gallery_set: set[str],
    max_words: int,
) -> list[str]:
    """
    Exact phrase matching after lowercase/punctuation normalization.

    This deliberately does NOT add plural variants and does NOT use NLTK,
    TextBlob, stemming, or lemmatization.
    """
    words = _WORD_RE.findall(caption.lower())
    found: list[str] = []
    used: set[str] = set()

    for n in range(1, min(max_words, len(words)) + 1):
        for start in range(0, len(words) - n + 1):
            phrase = " ".join(words[start:start + n])
            if phrase in gallery_set and phrase not in used:
                used.add(phrase)
                found.append(phrase)
    return found


def build_caption_maps(data: dict):
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        captions_by_image[int(ann["image_id"])].append(str(ann["caption"]))
    return captions_by_image


def build_image_semantics(
    images: list[dict],
    captions_by_image: dict[int, list[str]],
    gallery: list[str],
):
    gallery_set = set(gallery)
    max_words = max(len(x.split()) for x in gallery)
    semantics: list[list[str]] = []

    for item in tqdm(images, desc="Original captions -> exact gallery concepts"):
        image_id = int(item["id"])
        used: dict[str, None] = {}
        for caption in captions_by_image.get(image_id, []):
            for concept in exact_caption_concepts(
                caption, gallery_set=gallery_set, max_words=max_words
            ):
                used.setdefault(concept, None)
        semantics.append(list(used))
    return semantics


def resolve_image(images_dir: Path, item: dict) -> Path:
    path = images_dir / item["file_name"]
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


@torch.inference_mode()
def extract_clip_image_features(
    model,
    preprocess,
    images: list[dict],
    images_dir: Path,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(
        range(0, len(images), batch_size),
        desc="CLIP image features",
    ):
        batch_items = images[start:start + batch_size]
        tensors = []
        for item in batch_items:
            with Image.open(resolve_image(images_dir, item)) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(device, non_blocking=True)
        feats = model.encode_image(batch).float()
        feats = F.normalize(feats, dim=-1)
        chunks.append(feats.cpu().half())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def extract_clip_text_features(
    model,
    concepts: list[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    import clip

    chunks = []
    for start in tqdm(
        range(0, len(concepts), batch_size),
        desc="CLIP concept features",
    ):
        part = concepts[start:start + batch_size]
        prompts = [f"a photo of a {concept}" for concept in part]
        tokens = clip.tokenize(prompts, truncate=True).to(device)
        feats = model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)
        chunks.append(feats.cpu().half())
    return torch.cat(chunks, dim=0)


def unwrap_state_dict(checkpoint):
    state = checkpoint
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError("Unsupported checkpoint format: expected a state_dict-like object.")

    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "projector.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_projector(config_path: str, weight_path: str, device: str):
    from src.model import ProjectionLayer

    projector = ProjectionLayer.from_config(config_path)

    checkpoint = torch.load(
        weight_path,
        map_location="cpu",
    )

    state_dict = unwrap_state_dict(checkpoint)

    # ProjectionLayer overrides load_state_dict() and returns None,
    # so do not unpack the return value.
    # strict=True guarantees that a mismatched checkpoint raises immediately.
    projector.load_state_dict(
        state_dict,
        strict=True,
    )

    projector.to(device)
    projector.eval()

    print(
        f"Loaded projector: {weight_path} | "
        f"config: {config_path}"
    )

    return projector


def _feature_from_image_item(item: dict):
    candidate_keys = (
        "dino_features",
        "x_norm_clstoken",
        "cls_token",
        "cls",
    )
    for key in candidate_keys:
        if key in item:
            feat = item[key]
            if not torch.is_tensor(feat):
                feat = torch.as_tensor(feat)
            feat = feat.float().squeeze()
            if feat.ndim != 1:
                raise ValueError(
                    f"Feature under key '{key}' must become 1-D after squeeze; "
                    f"got shape {tuple(feat.shape)}"
                )
            return feat, key
    return None, None


def load_dino_cls_from_pth(
    pth_path: str | Path,
    expected_images: list[dict],
) -> torch.Tensor:
    """
    Preferred path for Talk2DINO-image.

    Use the exact CLS-feature PTH that was used to train the image-level projector.
    This avoids any preprocessing/resolution mismatch during curation.
    """
    data = torch.load(pth_path, map_location="cpu")
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError("DINO feature PTH must contain data['images'].")

    feature_by_id: dict[int, torch.Tensor] = {}
    detected_key = None

    for item in data["images"]:
        image_id = int(item.get("id", item.get("image_id", -1)))
        feat, key = _feature_from_image_item(item)
        if feat is not None:
            feature_by_id[image_id] = feat
            detected_key = detected_key or key

    if not feature_by_id:
        raise ValueError(
            "No CLS feature found. Expected one of: "
            "dino_features, x_norm_clstoken, cls_token, cls."
        )

    rows = []
    missing_ids = []
    for item in expected_images:
        image_id = int(item["id"])
        if image_id not in feature_by_id:
            missing_ids.append(image_id)
            continue
        rows.append(feature_by_id[image_id])

    if missing_ids:
        preview = missing_ids[:10]
        raise ValueError(
            f"DINO PTH is missing {len(missing_ids)} COCO image ids. "
            f"First missing ids: {preview}"
        )

    features = torch.stack(rows, dim=0).float()
    features = F.normalize(features, dim=-1).half()
    print(
        f"Loaded DINO CLS features from {pth_path} "
        f"(key='{detected_key}', shape={tuple(features.shape)})"
    )
    return features


@torch.inference_mode()
def extract_talk2dino_text_features(
    concepts: list[str],
    clip_model,
    projector,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    import clip

    chunks = []
    for start in tqdm(
        range(0, len(concepts), batch_size),
        desc="Talk2DINO-image concept features",
    ):
        part = concepts[start:start + batch_size]
        prompts = [f"a photo of a {concept}" for concept in part]
        tokens = clip.tokenize(prompts, truncate=True).to(device)

        clip_text = clip_model.encode_text(tokens).float()
        dino_text = projector.project_clip_txt(clip_text).float()
        dino_text = F.normalize(dino_text, dim=-1)
        chunks.append(dino_text.cpu().half())

    return torch.cat(chunks, dim=0)


def cache_load(path: Path, expected_meta: dict):
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or obj.get("meta") != expected_meta:
        return None
    return obj["features"]


def cache_save(path: Path, meta: dict, features: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "features": features}, path)


@torch.inference_mode()
def exact_topk_neighbors(
    image_features: torch.Tensor,
    top_k: int,
    query_batch: int,
    device: str,
    include_self: bool,
) -> torch.Tensor:
    """
    Exact cosine retrieval.

    CoCu source behavior with data == index_data naturally includes self.
    Therefore include_self=True is the default for this experiment.
    """
    n = image_features.shape[0]
    if top_k <= 0 or top_k >= n:
        raise ValueError(f"top_k must be in [1, {n - 1}], got {top_k}")

    bank = F.normalize(image_features.float(), dim=-1).to(device)
    result = torch.empty((n, top_k), dtype=torch.int32)

    for start in tqdm(
        range(0, n, query_batch),
        desc="Exact image-image retrieval",
    ):
        end = min(start + query_batch, n)
        query = bank[start:end]
        sims = query @ bank.T

        if not include_self:
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            sims[rows, cols] = -torch.inf

        indices = sims.topk(top_k, dim=1).indices
        result[start:end] = indices.cpu().to(torch.int32)

        del sims, indices

    return result


def cocu_score_archive(
    anchor_idx: int,
    neighbor_ids: list[int],
    semantics: list[list[str]],
    image_features: torch.Tensor,
    concept_features: torch.Tensor,
    concept2idx: dict[str, int],
):
    """
    Vectorized equivalent of the central scoring logic in rewrite/curation.py.

    Important source-like detail:
    - neighbor_ids may contain anchor_idx itself;
    - anchor is appended again as the final global-reference column.
      This mirrors original CoCu when searching the same index.
    """
    archive: dict[str, None] = {}
    for nidx in neighbor_ids:
        for concept in semantics[nidx]:
            archive.setdefault(concept, None)

    if not archive:
        return []

    concepts = list(archive)
    cand_ids = [concept2idx[c] for c in concepts]

    # [C, D]
    cfeat = concept_features[cand_ids].float()

    # CoCu: reference images are retrieved images carrying concept + anchor.
    refs = image_features[neighbor_ids + [anchor_idx]].float()
    sims = cfeat @ refs.T  # [C, L+1]

    mask = torch.zeros_like(sims, dtype=torch.bool)
    for col, nidx in enumerate(neighbor_ids):
        neighbor_sem = set(semantics[nidx])
        mask[:, col] = torch.tensor(
            [c in neighbor_sem for c in concepts],
            dtype=torch.bool,
        )
    mask[:, -1] = True  # anchor is always appended for global similarity.

    global_prob = sims[:, -1]

    count = mask.sum(dim=1).clamp_min(1)
    denom = (sims * mask).sum(dim=1) / count

    eps = 1e-6
    denom = torch.where(
        denom.abs() < eps,
        torch.where(denom < 0, -torch.full_like(denom, eps), torch.full_like(denom, eps)),
        denom,
    )
    local_prob = global_prob / denom

    valid_sims = sims.masked_fill(~mask, -torch.inf)
    # Original code starts max_prob at 0.0.
    max_prob = max(0.0, float(valid_sims.max()))
    max_prob = max(max_prob, eps)

    score = global_prob / max_prob + local_prob

    ranked = [
        {"concept": concept, "score": float(value)}
        for concept, value in zip(concepts, score.tolist())
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def write_json_header(fp, meta: dict):
    fp.write('{\n  "meta": ')
    json.dump(meta, fp, ensure_ascii=False, indent=2)
    fp.write(',\n  "images": [\n')


def write_json_footer(fp):
    fp.write('\n  ]\n}\n')


def build_features(args, images, image_ids, concepts, tag, cache_dir: Path):
    import clip

    image_meta = {
        "backend": args.backend,
        "image_ids": image_ids,
        "weight": str(args.projector_weight or ""),
        "dino_features_pth": str(args.dino_features_pth or ""),
        "clip_model": args.clip_model,
    }
    concept_meta = {
        "backend": args.backend,
        "concepts": concepts,
        "weight": str(args.projector_weight or ""),
        "projector_config": str(args.projector_config or ""),
        "clip_model": args.clip_model,
    }

    image_cache = cache_dir / f"{tag}_image_features.pt"
    concept_cache = cache_dir / f"{tag}_concept_features.pt"

    image_features = None if args.overwrite else cache_load(image_cache, image_meta)
    concept_features = None if args.overwrite else cache_load(concept_cache, concept_meta)

    if args.backend == "clip":
        model, preprocess = clip.load(args.clip_model, device=args.device, jit=False)
        model.eval()

        if image_features is None:
            if not args.images_dir:
                raise ValueError("--images-dir is required for --backend clip")
            image_features = extract_clip_image_features(
                model=model,
                preprocess=preprocess,
                images=images,
                images_dir=Path(args.images_dir),
                batch_size=args.image_batch,
                device=args.device,
            )
            cache_save(image_cache, image_meta, image_features)
        else:
            print(f"Loaded image cache: {image_cache}")

        if concept_features is None:
            concept_features = extract_clip_text_features(
                model=model,
                concepts=concepts,
                batch_size=args.text_batch,
                device=args.device,
            )
            cache_save(concept_cache, concept_meta, concept_features)
        else:
            print(f"Loaded concept cache: {concept_cache}")

        del model

    elif args.backend == "talk2dino_image":
        if not args.projector_weight:
            raise ValueError("--projector-weight is required for talk2dino_image")
        if not args.projector_config:
            raise ValueError("--projector-config is required for talk2dino_image")
        if not args.dino_features_pth:
            raise ValueError(
                "--dino-features-pth is required for talk2dino_image. "
                "Use the exact CLS-feature PTH used to train the image-level projector."
            )

        if image_features is None:
            image_features = load_dino_cls_from_pth(
                args.dino_features_pth,
                expected_images=images,
            )
            cache_save(image_cache, image_meta, image_features)
        else:
            print(f"Loaded image cache: {image_cache}")

        if concept_features is None:
            clip_model, _ = clip.load(
                args.clip_model,
                device=args.device,
                jit=False,
            )
            clip_model.eval()

            projector = load_projector(
                config_path=args.projector_config,
                weight_path=args.projector_weight,
                device=args.device,
            )

            concept_features = extract_talk2dino_text_features(
                concepts=concepts,
                clip_model=clip_model,
                projector=projector,
                batch_size=args.text_batch,
                device=args.device,
            )
            cache_save(concept_cache, concept_meta, concept_features)

            del clip_model, projector
        else:
            print(f"Loaded concept cache: {concept_cache}")

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if image_features.shape[1] != concept_features.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match: "
            f"image={image_features.shape[1]}, concept={concept_features.shape[1]}"
        )

    return image_features, concept_features


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        required=True,
        choices=["clip", "talk2dino_image"],
    )
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)

    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory containing train2014 images; needed by CLIP backend.",
    )
    parser.add_argument("--clip-model", default="ViT-B/16")

    parser.add_argument(
        "--projector-config",
        default="configs/vitb_mlp_infonce.yaml",
    )
    parser.add_argument("--projector-weight", default=None)
    parser.add_argument(
        "--dino-features-pth",
        default=None,
        help=(
            "PTH containing per-image DINO CLS under dino_features/"
            "x_norm_clstoken. Prefer the exact PTH used to train the "
            "image-level projector."
        ),
    )

    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--num-curated",
        type=int,
        default=3,
        help="Top-N concepts written to curated_concepts; 0 keeps all.",
    )
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Ablation only. Original CoCu same-index retrieval includes self.",
    )
    parser.add_argument(
        "--save-all-scored",
        action="store_true",
        help="Also write full sorted scored_archive for each image.",
    )

    parser.add_argument("--image-batch", type=int, default=128)
    parser.add_argument("--text-batch", type=int, default=512)
    parser.add_argument("--retrieval-batch", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    data = load_json(args.captions_json)
    if "images" not in data or "annotations" not in data:
        raise ValueError("Expected COCO captions JSON with images + annotations.")

    images = list(data["images"])
    if args.limit:
        images = images[:args.limit]

    valid_ids = {int(item["id"]) for item in images}
    annotations = [
        ann for ann in data["annotations"]
        if int(ann["image_id"]) in valid_ids
    ]
    data = dict(data)
    data["annotations"] = annotations

    image_ids = [int(item["id"]) for item in images]
    captions_by_image = build_caption_maps(data)

    concepts = load_exact_gallery(args.gallery)
    print(
        f"Images: {len(images):,} | captions: {len(annotations):,} | "
        f"exact gallery concepts: {len(concepts):,}"
    )
    print(
        "Gallery policy: exact only; no plural expansion / stemming / "
        "lemmatization / NLTK / TextBlob."
    )

    semantics = build_image_semantics(
        images=images,
        captions_by_image=captions_by_image,
        gallery=concepts,
    )

    if args.backend == "clip":
        tag = "clip_vitb16"
    else:
        tag = "talk2dino_image_" + Path(args.projector_weight).stem

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_features, concept_features = build_features(
        args=args,
        images=images,
        image_ids=image_ids,
        concepts=concepts,
        tag=tag,
        cache_dir=cache_dir,
    )

    neighbors_path = cache_dir / (
        f"{tag}_neighbors_k{args.top_k}_"
        f"{'exclude_self' if args.exclude_self else 'include_self'}.pt"
    )

    neighbors = None
    if neighbors_path.exists() and not args.overwrite:
        cached = torch.load(neighbors_path, map_location="cpu")
        if cached.get("image_ids") == image_ids:
            neighbors = cached["indices"]
            print(f"Loaded neighbor cache: {neighbors_path}")

    if neighbors is None:
        neighbors = exact_topk_neighbors(
            image_features=image_features,
            top_k=args.top_k,
            query_batch=args.retrieval_batch,
            device=args.device,
            include_self=not args.exclude_self,
        )
        torch.save(
            {
                "image_ids": image_ids,
                "indices": neighbors,
                "include_self": not args.exclude_self,
            },
            neighbors_path,
        )

    concept2idx = {c: i for i, c in enumerate(concepts)}

    meta = {
        "method": "CoCu-style curated concept extraction",
        "backend": args.backend,
        "captions_json": str(args.captions_json),
        "gallery": str(args.gallery),
        "gallery_policy": "exact match only; no plural expansion",
        "clip_model": args.clip_model,
        "projector_config": (
            str(args.projector_config)
            if args.backend == "talk2dino_image"
            else None
        ),
        "projector_weight": (
            str(args.projector_weight)
            if args.backend == "talk2dino_image"
            else None
        ),
        "dino_features_pth": (
            str(args.dino_features_pth)
            if args.backend == "talk2dino_image"
            else None
        ),
        "top_k": args.top_k,
        "include_self": not args.exclude_self,
        "num_curated": args.num_curated,
        "coco_semantics": "union of exact concepts across all original captions per image",
        "score": "global/max_prob + global/mean(reference+anchor similarities)",
        "curated_concepts": "source-ranked top-N; may overlap original concepts",
        "new_curated_concepts": "top-N after removing concepts already present in original captions",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        write_json_header(fp, meta)

        for image_idx, item in enumerate(
            tqdm(images, desc=f"CoCu scoring [{tag}]")
        ):
            neigh = [int(x) for x in neighbors[image_idx].tolist()]
            ranked = cocu_score_archive(
                anchor_idx=image_idx,
                neighbor_ids=neigh,
                semantics=semantics,
                image_features=image_features,
                concept_features=concept_features,
                concept2idx=concept2idx,
            )

            original_set = set(semantics[image_idx])
            new_ranked = [
                row for row in ranked
                if row["concept"] not in original_set
            ]

            if args.num_curated > 0:
                curated = ranked[:args.num_curated]
                new_curated = new_ranked[:args.num_curated]
            else:
                curated = ranked
                new_curated = new_ranked

            image_id = int(item["id"])
            row = {
                "image_id": image_id,
                "file_name": item.get("file_name"),
                "captions": captions_by_image.get(image_id, []),
                "original_concepts": semantics[image_idx],
                "curated_concepts": curated,
                "new_curated_concepts": new_curated,
            }
            if args.save_all_scored:
                row["scored_archive"] = ranked

            if image_idx:
                fp.write(",\n")
            fp.write("    ")
            json.dump(row, fp, ensure_ascii=False)

        write_json_footer(fp)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
