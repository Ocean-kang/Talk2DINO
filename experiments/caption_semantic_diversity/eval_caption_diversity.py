#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP-based semantic-diversity evaluation for Talk2DINO's 5k caption variants.

Reuses the same OpenAI CLIP API used by Talk2DINO:
    model, preprocess = clip.load("ViT-B/16", ...)
    tokens = clip.tokenize(texts, truncate=True)
    features = model.encode_text(tokens)

No training, LLM, or data/weight download is performed by this script.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


METHOD_KEYS = {
    "raw": "raw_captions",
    "rich": "rich_captions",
    "rich_weaken": "rich_weaken_captions",
    "rich_diversity": "rich_diversity_captions",
}
METHOD_ORDER = ["raw", "rich", "rich_weaken", "rich_diversity"]
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_caption_records(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            "Expected a JSON list: [{image_id, raw_captions, rich_captions, "
            "rich_weaken_captions, rich_diversity_captions}, ...]."
        )
    if not records:
        raise ValueError("Caption JSON is empty.")

    seen_ids = set()
    clean = []
    for row_idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Record {row_idx} is not a JSON object.")
        if "image_id" not in row:
            raise ValueError(f"Record {row_idx} has no image_id.")
        image_id = row["image_id"]
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id found: {image_id}")
        seen_ids.add(image_id)

        item = {"image_id": image_id}
        for method, key in METHOD_KEYS.items():
            captions = row.get(key)
            if not isinstance(captions, list) or len(captions) != 5:
                raise ValueError(
                    f"image_id={image_id}: '{key}' must contain exactly 5 captions."
                )
            if not all(isinstance(x, str) and x.strip() for x in captions):
                raise ValueError(
                    f"image_id={image_id}: '{key}' contains an empty/non-string caption."
                )
            item[method] = [x.strip() for x in captions]
        clean.append(item)
    return clean


def _expected_cached_clip_path(model_name, cache_dir):
    """Resolve OpenAI CLIP's expected cache file without triggering a download."""
    models = None
    try:
        from clip.clip import _MODELS as models
    except Exception:
        try:
            models = clip._MODELS
        except Exception:
            models = None

    if models is not None and model_name in models:
        filename = Path(urlparse(models[model_name]).path).name
    else:
        filename = model_name.replace("/", "-") + ".pt"
    return Path(cache_dir).expanduser() / filename


def _build_preprocess(input_resolution):
    # Same normalization constants and resize/center-crop convention as OpenAI CLIP.
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    return T.Compose([
        T.Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(input_resolution),
        lambda image: image.convert("RGB"),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def load_clip_model(model_name, device, checkpoint=None, cache_dir=None):
    if checkpoint:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CLIP checkpoint does not exist: {checkpoint}")
        # Accept either an OpenAI CLIP TorchScript archive or a plain state_dict.
        try:
            model = torch.jit.load(str(checkpoint), map_location=device).eval()
        except Exception:
            try:
                state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(str(checkpoint), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                from clip.model import build_model
            except Exception:
                from clip.clip import build_model
            model = build_model(state).to(device).eval()
        resolution = int(model.visual.input_resolution)
        preprocess = _build_preprocess(resolution)
        return model, preprocess, str(checkpoint)

    cache_dir = Path(cache_dir or Path.home() / ".cache" / "clip").expanduser()
    expected = _expected_cached_clip_path(model_name, cache_dir)
    if not expected.is_file():
        raise FileNotFoundError(
            "No cached CLIP weight was found, and automatic downloads are disabled.\n"
            f"Expected cache file: {expected}\n"
            "Either place the existing OpenAI CLIP weight there or pass "
            "--clip-checkpoint /path/to/ViT-B-16.pt."
        )
    # This is the same loader used in Talk2DINO/text_features_extraction.py.
    model, preprocess = clip.load(
        model_name, device=device, jit=False, download_root=str(cache_dir)
    )
    model.eval()
    return model, preprocess, str(expected)


def encode_texts(model, texts, device, batch_size):
    all_features = []
    token_counts = []
    truncated_flags = []

    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text encoding"):
        batch_texts = texts[start:start + batch_size]
        # Reuse Talk2DINO's clip.tokenize(..., truncate=True) behavior.
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens).float()
            features = torch.nn.functional.normalize(features, p=2, dim=-1)

        # OpenAI CLIP uses SOT/EOT plus zero padding. This counts content BPE tokens.
        content_counts = (tokens != 0).sum(dim=1).detach().cpu().numpy() - 2
        content_counts = np.maximum(content_counts, 0).astype(np.int32)
        token_counts.extend(content_counts.tolist())
        truncated_flags.extend((content_counts >= 75).tolist())
        all_features.append(features.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_features, axis=0), np.asarray(token_counts), np.asarray(truncated_flags)


def effective_rank(centered_features):
    # Centering removes the shared mean direction. With K=5, rank is at most K-1=4.
    singular_values = np.linalg.svd(centered_features, full_matrices=False, compute_uv=False)
    energy = singular_values ** 2
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    probs = energy / total
    probs = probs[probs > 1e-12]
    entropy = -float(np.sum(probs * np.log(probs)))
    er = float(np.exp(entropy))
    return er, er / 4.0


def caption_group_metrics(features_5xD, word_counts, token_counts, truncated_flags):
    # Input embeddings are L2-normalized.
    sim = np.clip(features_5xD @ features_5xD.T, -1.0, 1.0)
    upper = sim[np.triu_indices(5, k=1)]
    mean_sim = float(upper.mean())
    cisd = float(1.0 - mean_sim)
    nn_redundancy = float(np.max(sim - np.eye(5) * 2.0, axis=1).mean())
    er, er_norm = effective_rank(features_5xD - features_5xD.mean(axis=0, keepdims=True))

    return {
        "cisd": cisd,
        "mean_pairwise_similarity": mean_sim,
        "nn_redundancy": nn_redundancy,
        "effective_rank": er,
        "effective_rank_norm": er_norm,
        "mean_word_count": float(np.mean(word_counts)),
        "mean_clip_token_count": float(np.mean(token_counts)),
        "truncated_caption_count": int(np.sum(truncated_flags)),
    }, sim


def resolve_image_path(image_id, image_dir, image_pattern, image_manifest=None):
    if image_manifest is not None:
        key = str(image_id)
        if key not in image_manifest:
            raise KeyError(f"image_id={image_id} not found in image manifest.")
        p = Path(image_manifest[key])
        if not p.is_absolute():
            p = Path(image_dir) / p
        return p

    filename = image_pattern.format(image_id=int(image_id))
    return Path(image_dir) / filename


def load_image_manifest(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # COCO instances/captions format: {"images": [{"id": ..., "file_name": ...}, ...]}
    if isinstance(obj, dict) and isinstance(obj.get("images"), list):
        return {str(x["id"]): x["file_name"] for x in obj["images"]}
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    raise ValueError("Image manifest must be a dict or a COCO JSON object with an 'images' list.")


def encode_images(model, preprocess, image_ids, image_dir, image_pattern,
                  image_manifest, device, batch_size):
    image_features = []
    for start in tqdm(range(0, len(image_ids), batch_size), desc="CLIP image encoding"):
        batch_ids = image_ids[start:start + batch_size]
        tensors = []
        for image_id in batch_ids:
            path = resolve_image_path(image_id, image_dir, image_pattern, image_manifest)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Image not found for image_id={image_id}: {path}\n"
                    "Check --image-dir, --image-pattern, or --image-manifest."
                )
            with Image.open(path) as im:
                tensors.append(preprocess(im.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch).float()
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        image_features.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(image_features, axis=0)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(per_image_rows):
    metric_names = [
        "cisd", "mean_pairwise_similarity", "nn_redundancy",
        "effective_rank", "effective_rank_norm",
        "mean_word_count", "mean_clip_token_count", "truncated_caption_count",
        "grounding_mean", "grounding_min",
    ]
    rows = []
    for method in METHOD_ORDER:
        subset = [r for r in per_image_rows if r["caption_type"] == method]
        row = {"caption_type": method, "n_images": len(subset)}
        for metric in metric_names:
            values = np.asarray(
                [r[metric] for r in subset if r.get(metric) is not None],
                dtype=np.float64
            )
            if values.size:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                row[f"{metric}_median"] = float(np.median(values))
            else:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
                row[f"{metric}_median"] = ""
        rows.append(row)
    return rows


def paired_bootstrap_ci(deltas, iterations, seed, chunk_size=128):
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(iterations, dtype=np.float64)
    cursor = 0
    while cursor < iterations:
        current = min(chunk_size, iterations - cursor)
        indices = rng.integers(0, n, size=(current, n))
        boot_means[cursor:cursor + current] = deltas[indices].mean(axis=1)
        cursor += current
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    adjusted = np.ones(n, dtype=np.float64)
    order = np.argsort(p_values)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted.tolist()


def paired_statistics(per_image_rows, iterations, seed):
    # Reassemble by image_id so all comparisons are genuinely paired.
    lookup = {}
    for row in per_image_rows:
        lookup[(str(row["image_id"]), row["caption_type"])] = row

    image_ids = sorted({str(r["image_id"]) for r in per_image_rows})
    metrics = ["cisd", "nn_redundancy", "effective_rank_norm"]
    has_grounding = any(r.get("grounding_mean") is not None for r in per_image_rows)
    if has_grounding:
        metrics += ["grounding_mean", "grounding_min"]

    output = []
    for metric_idx, metric in enumerate(metrics):
        metric_rows = []
        pvals = []
        for comp_idx, baseline in enumerate(["raw", "rich", "rich_weaken"]):
            paired = [
                (lookup[(iid, "rich_diversity")][metric],
                 lookup[(iid, baseline)][metric])
                for iid in image_ids
                if (iid, "rich_diversity") in lookup
                and (iid, baseline) in lookup
                and lookup[(iid, "rich_diversity")].get(metric) is not None
                and lookup[(iid, baseline)].get(metric) is not None
            ]
            if not paired:
                continue
            target = np.asarray([x[0] for x in paired], dtype=np.float64)
            ref = np.asarray([x[1] for x in paired], dtype=np.float64)
            delta = target - ref
            ci_low, ci_high = paired_bootstrap_ci(
                delta, iterations, seed + metric_idx * 101 + comp_idx
            )
            if wilcoxon is not None:
                try:
                    p_value = float(wilcoxon(target, ref, zero_method="wilcox",
                                             alternative="two-sided", method="auto").pvalue)
                except Exception:
                    p_value = 1.0
            else:
                p_value = float("nan")
            row = {
                "metric": metric,
                "comparison": f"rich_diversity - {baseline}",
                "n_paired_images": int(len(delta)),
                "mean_paired_difference": float(delta.mean()),
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "wilcoxon_p": p_value,
                "holm_adjusted_p": "",
            }
            metric_rows.append(row)
            pvals.append(p_value)

        valid = [i for i, p in enumerate(pvals) if np.isfinite(p)]
        if valid:
            adjusted = holm_adjust([pvals[i] for i in valid])
            for i, adj in zip(valid, adjusted):
                metric_rows[i]["holm_adjusted_p"] = float(adj)
        output.extend(metric_rows)
    return output


def length_adjusted_fixed_effects(per_image_rows):
    """
    Within-image OLS:
        CISD = image fixed effect + method indicators + mean CLIP BPE length.
    Coefficients for methods are adjusted differences vs Raw.
    This is a diagnostic, not a causal claim.
    """
    lookup = {}
    image_ids = sorted({str(r["image_id"]) for r in per_image_rows})
    for r in per_image_rows:
        lookup[(str(r["image_id"]), r["caption_type"])] = r

    y_rows, x_rows = [], []
    for iid in image_ids:
        if not all((iid, m) in lookup for m in METHOD_ORDER):
            continue
        group = [lookup[(iid, m)] for m in METHOD_ORDER]
        y = np.asarray([g["cisd"] for g in group], dtype=np.float64)
        lengths = np.asarray([g["mean_clip_token_count"] for g in group], dtype=np.float64)
        # Reference category is Raw. Dummy order: Rich, Rich-Weaken, Rich-Diversity.
        method_dummies = np.asarray([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        X = np.column_stack([method_dummies, lengths])
        y_rows.append(y - y.mean())
        x_rows.append(X - X.mean(axis=0, keepdims=True))

    if not y_rows:
        return []
    y = np.concatenate(y_rows)
    X = np.concatenate(x_rows, axis=0)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    names = ["rich_vs_raw_adjusted", "rich_weaken_vs_raw_adjusted",
             "rich_diversity_vs_raw_adjusted", "clip_token_length_slope"]
    return [
        {
            "term": name,
            "coefficient": float(value),
            "interpretation": (
                "Adjusted CISD difference vs Raw, controlling for mean CLIP token length"
                if name != "clip_token_length_slope"
                else "CISD change per additional mean CLIP content token"
            ),
        }
        for name, value in zip(names, beta)
    ]


def position_similarity_matrix(embeddings):
    # embeddings shape: N x 5 x D for Rich-Diversity
    sums = np.zeros((5, 5), dtype=np.float64)
    for item in embeddings:
        sums += item @ item.T
    return sums / len(embeddings)


def make_plots(per_image_rows, position_matrix, output_dir):
    if plt is None:
        print("WARNING: matplotlib is unavailable; skipping plots.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("cisd", "CLIP Intra-set Semantic Diversity (CISD)", "cisd_boxplot.png"),
        ("nn_redundancy", "Nearest-neighbor Redundancy", "nn_redundancy_boxplot.png"),
        ("effective_rank_norm", "Normalized Effective Rank", "effective_rank_boxplot.png"),
    ]
    for metric, title, filename in metric_specs:
        values = [
            [r[metric] for r in per_image_rows if r["caption_type"] == method]
            for method in METHOD_ORDER
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(values, labels=METHOD_ORDER, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(position_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    labels = ["C1 Main", "C2 Attr", "C3 Local", "C4 Rel", "C5 Scene"]
    ax.set_xticks(range(5), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(5), labels=labels)
    ax.set_title("Rich-Diversity Caption Position Similarity")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{position_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "rich_diversity_position_similarity.png", dpi=180)
    plt.close(fig)

    # Group-level CISD vs grounding means, only if image grounding was run.
    grounded = [r for r in per_image_rows if r.get("grounding_mean") is not None]
    if grounded:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in METHOD_ORDER:
            subset = [r for r in grounded if r["caption_type"] == method]
            x = np.mean([r["grounding_mean"] for r in subset])
            y = np.mean([r["cisd"] for r in subset])
            ax.scatter(x, y, label=method, s=65)
            ax.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Mean CLIP Image-Text Similarity")
        ax.set_ylabel("Mean CISD")
        ax.set_title("Semantic Diversity vs Image-Text Consistency")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "diversity_vs_grounding.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic diversity of Talk2DINO caption variants with frozen CLIP."
    )
    parser.add_argument("--captions-json", required=True, help="5k_rich_diversity JSON path.")
    parser.add_argument("--output-dir", default="experiments/caption_semantic_diversity/outputs")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-checkpoint", default=None,
                        help="Optional local OpenAI CLIP .pt/.pth or TorchScript checkpoint.")
    parser.add_argument("--clip-cache-dir", default=str(Path.home() / ".cache" / "clip"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-dir", default=None,
                        help="Optional COCO image directory. If omitted, grounding is skipped.")
    parser.add_argument("--image-pattern", default="COCO_train2014_{image_id:012d}.jpg",
                        help="Python format pattern; available field: image_id.")
    parser.add_argument("--image-manifest", default=None,
                        help="Optional COCO JSON or {image_id: relative/absolute path} mapping.")
    args = parser.parse_args()

    if args.batch_size < 1 or args.bootstrap_iterations < 100:
        parser.error("--batch-size must be positive and --bootstrap-iterations >= 100.")

    json_path = Path(args.captions_json).expanduser().resolve()
    if not json_path.is_file():
        parser.error(f"Caption JSON not found: {json_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is False.")

    print(f"[1/6] Loading captions: {json_path}")
    records = load_caption_records(json_path)
    image_ids = [r["image_id"] for r in records]
    print(f"      Validated {len(records)} images, 4 methods x 5 captions/image.")

    print(f"[2/6] Loading frozen CLIP {args.clip_model} on {device}")
    model, preprocess, weight_source = load_clip_model(
        args.clip_model, device, args.clip_checkpoint, args.clip_cache_dir
    )
    model.eval()
    model.requires_grad_(False)

    flat_texts = []
    flat_meta = []
    for row in records:
        for method in METHOD_ORDER:
            for position, caption in enumerate(row[method], start=1):
                flat_texts.append(caption)
                flat_meta.append((row["image_id"], method, position))

    print(f"[3/6] Encoding {len(flat_texts)} captions with CLIP.")
    text_features, token_counts, truncated_flags = encode_texts(
        model, flat_texts, device, args.batch_size
    )
    if text_features.shape != (len(flat_texts), 512):
        print(f"WARNING: Expected ViT-B/16 text features [N,512], got {text_features.shape}.")

    n = len(records)
    embeddings = text_features.reshape(n, len(METHOD_ORDER), 5, -1)
    tokens_4d = token_counts.reshape(n, len(METHOD_ORDER), 5)
    trunc_4d = truncated_flags.reshape(n, len(METHOD_ORDER), 5)

    per_image_rows = []
    per_caption_rows = []
    position_matrices = []
    rich_diversity_embeddings = embeddings[:, METHOD_ORDER.index("rich_diversity"), :, :]

    image_features = None
    if args.image_dir:
        print(f"[4/6] Encoding {n} COCO images for optional grounding.")
        manifest = load_image_manifest(args.image_manifest)
        image_features = encode_images(
            model, preprocess, image_ids, args.image_dir, args.image_pattern,
            manifest, device, args.batch_size
        )
    else:
        print("[4/6] No --image-dir supplied; image grounding will be skipped.")

    for i, row in enumerate(tqdm(records, desc="Computing per-image metrics")):
        for method_idx, method in enumerate(METHOD_ORDER):
            captions = row[method]
            features = embeddings[i, method_idx]
            words = np.asarray([len(WORD_RE.findall(c)) for c in captions], dtype=np.int32)
            toks = tokens_4d[i, method_idx]
            trunc = trunc_4d[i, method_idx]
            metrics, sim = caption_group_metrics(features, words, toks, trunc)
            grounding = None
            if image_features is not None:
                scores = features @ image_features[i]
                metrics["grounding_mean"] = float(scores.mean())
                metrics["grounding_min"] = float(scores.min())
                grounding = scores
            else:
                metrics["grounding_mean"] = None
                metrics["grounding_min"] = None

            per_image_rows.append({
                "image_id": row["image_id"],
                "caption_type": method,
                **metrics,
            })

            for pos, caption in enumerate(captions):
                caption_row = {
                    "image_id": row["image_id"],
                    "caption_type": method,
                    "caption_position": pos + 1,
                    "caption": caption,
                    "word_count": int(words[pos]),
                    "clip_token_count": int(toks[pos]),
                    "clip_truncated": bool(trunc[pos]),
                }
                if grounding is not None:
                    caption_row["image_text_cosine"] = float(grounding[pos])
                per_caption_rows.append(caption_row)

    position_matrix = position_similarity_matrix(rich_diversity_embeddings)

    print("[5/6] Statistical analysis (paired bootstrap + Wilcoxon + Holm).")
    summary_rows = summarize_metrics(per_image_rows)
    stats_rows = paired_statistics(per_image_rows, args.bootstrap_iterations, args.seed)
    length_rows = length_adjusted_fixed_effects(per_image_rows)

    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "per_caption_metrics.csv", per_caption_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    write_csv(output_dir / "paired_statistics.csv", stats_rows)
    write_csv(output_dir / "length_adjusted_ols.csv", length_rows)

    pos_rows = []
    pos_labels = ["main_objects_actions", "attributes_appearance",
                  "local_details", "relationships", "background_scene"]
    for i, label_i in enumerate(pos_labels):
        row = {"caption_position": label_i}
        for j, label_j in enumerate(pos_labels):
            row[label_j] = float(position_matrix[i, j])
        pos_rows.append(row)
    write_csv(output_dir / "rich_diversity_position_similarity.csv", pos_rows)

    np.save(output_dir / "rich_diversity_position_similarity.npy", position_matrix)
    make_plots(per_image_rows, position_matrix, output_dir / "figures")

    config = {
        "captions_json": str(json_path),
        "captions_json_sha256": sha256_file(json_path),
        "num_images": len(records),
        "captions_per_method_per_image": 5,
        "methods": METHOD_KEYS,
        "clip_model": args.clip_model,
        "clip_weight_source": weight_source,
        "device": device,
        "batch_size": args.batch_size,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "image_grounding_enabled": image_features is not None,
        "image_dir": str(Path(args.image_dir).expanduser().resolve()) if args.image_dir else None,
        "image_pattern": args.image_pattern if args.image_dir else None,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric_definitions": {
            "cisd": "1 - mean cosine similarity over the 10 unordered caption pairs; higher means more separated.",
            "nn_redundancy": "Mean over captions of the maximum cosine similarity to another caption; lower means less local duplication.",
            "effective_rank_norm": "exp(entropy of squared singular-value spectrum of centered 5x512 embeddings) / 4.",
            "grounding_mean": "Mean cosine similarity between the image embedding and the five caption embeddings.",
            "grounding_min": "Minimum image-caption cosine similarity within the five captions.",
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("[6/6] Done.")
    print(f"Outputs: {output_dir}")
    print("Primary file: summary_metrics.csv")
    print("Per-image paired data: per_image_metrics.csv")
    print("Statistical comparisons: paired_statistics.csv")
    print("Plots: figures/")
    if wilcoxon is None:
        print("WARNING: SciPy not found. Wilcoxon p-values are NaN; install scipy to enable tests.")


if __name__ == "__main__":
    main()
