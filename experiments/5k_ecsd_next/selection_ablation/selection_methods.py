#!/usr/bin/env python3
"""Deterministic B1-B4 greedy selection over one shared ECSD v2 candidate pool."""
import math
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

from candidate_quality.semantic_role_classifier import (
    classify_semantic_roles, content_tokens, normalize_text, quality_flags,
)

METHODS = {
    "B1": "Coverage Only",
    "B2": "CLIP Diversity Only",
    "B3": "Coverage + Diversity",
    "B4": "Coverage + Diversity + Role Novelty",
}

def build_fact_weights(raw_captions):
    frequency = Counter()
    for caption in raw_captions:
        frequency.update(content_tokens(caption))
    n = max(1, len(raw_captions))
    weights = {
        token: 1.0 + math.log((n + 1.0) / (count + 1.0))
        for token, count in frequency.items()
    }
    return frequency, weights

def weighted_jaccard(a, b, weights):
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return 0.0
    numerator = sum(weights.get(x, 1.0) for x in a & b)
    denominator = sum(weights.get(x, 1.0) for x in union)
    return numerator / denominator if denominator else 0.0

def _vector_array(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)

def cosine(a, b):
    a, b = _vector_array(a), _vector_array(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0

def _candidate_categories(candidate):
    cats = candidate.get("categories")
    if cats is None:
        cats = classify_semantic_roles(candidate.get("text", ""))["categories"]
    return {x for x in cats if x != "Unknown"}

def _candidate_facts(candidate):
    return set(candidate.get("facts") or content_tokens(candidate.get("text", "")))

def _valid_candidates(candidates, filter_config=None):
    """One common validity filter for all four ablation arms."""
    result = []
    seen_text = set()
    for item in sorted(candidates, key=lambda x: str(x.get("candidate_id", ""))):
        text = str(item.get("text", "")).strip()
        key = normalize_text(text)
        if not key or key in seen_text:
            continue
        # Re-evaluate every candidate under the same configured filter for all B1-B4 arms.
        flags = quality_flags(text, item.get("candidate_type", "candidate"), filter_config)
        if flags:
            continue
        if not _candidate_facts(item):
            continue
        seen_text.add(key)
        result.append(item)
    return result

def _coverage(candidate, covered, weights, total_weight):
    new_facts = _candidate_facts(candidate) - covered
    if not new_facts or total_weight <= 0:
        return 0.0
    return sum(weights.get(t, 1.0) for t in new_facts) / total_weight

def _diversity(candidate, selected, embedding_by_text):
    if not selected:
        return 0.0
    emb = embedding_by_text[candidate["text"]]
    distances = []
    for item in selected:
        sim = max(-1.0, min(1.0, cosine(emb, embedding_by_text[item["text"]])))
        distances.append((1.0 - sim) / 2.0)
    return sum(distances) / len(distances)

def _role_novelty(candidate, covered_categories):
    categories = _candidate_categories(candidate)
    if not categories:
        return 0.0
    return len(categories - covered_categories) / len(categories)

def choose_anchor(candidates, raw_captions, filter_config=None):
    """Same deterministic anchor for B1-B4: max weighted fact mass among valid originals."""
    valid = _valid_candidates(candidates, filter_config)
    originals = [x for x in valid if x.get("candidate_type") == "original"]
    if not originals:
        originals = valid
    if not originals:
        raise ValueError("No valid candidates available for anchor selection.")
    _, weights = build_fact_weights(raw_captions)
    total = sum(weights.values()) or 1.0
    return sorted(
        originals,
        key=lambda x: (
            -sum(weights.get(t, 1.0) for t in _candidate_facts(x)) / total,
            str(x.get("candidate_id", "")),
        ),
    )[0]

def select_five(candidates, raw_captions, embedding_by_text, method, score_weights=None, filter_config=None):
    if method not in METHODS:
        raise ValueError("Unknown method {}. Expected one of {}.".format(method, sorted(METHODS)))
    valid = _valid_candidates(candidates, filter_config)
    if not valid:
        raise ValueError("Candidate pool is empty after the shared validity filter.")
    _, weights = build_fact_weights(raw_captions)
    total_weight = sum(weights.values()) or 1.0
    anchor = choose_anchor(valid, raw_captions, filter_config)
    selected = [anchor]
    selected_keys = {normalize_text(anchor["text"])}
    covered_facts = set(_candidate_facts(anchor))
    covered_categories = set(_candidate_categories(anchor))
    trace = [{
        "step": 1, "candidate_id": anchor["candidate_id"], "text": anchor["text"],
        "candidate_type": anchor.get("candidate_type"), "score": None,
        "score_components": {"anchor": "max weighted fact mass among valid originals"},
        "source_caption_ids": anchor.get("source_caption_ids", []),
        "source_spans": anchor.get("source_spans", []),
        "categories": sorted(_candidate_categories(anchor)),
        "facts": sorted(_candidate_facts(anchor)),
    }]

    while len(selected) < 5:
        best = None
        best_score = -float("inf")
        best_components = None
        for candidate in valid:
            key = normalize_text(candidate["text"])
            if key in selected_keys:
                continue
            coverage = _coverage(candidate, covered_facts, weights, total_weight)
            diversity = _diversity(candidate, selected, embedding_by_text)
            role_novelty = _role_novelty(candidate, covered_categories)
            defaults = {
                "B1": {"coverage": 1.0, "diversity": 0.0, "role_novelty": 0.0},
                "B2": {"coverage": 0.0, "diversity": 1.0, "role_novelty": 0.0},
                "B3": {"coverage": 0.5, "diversity": 0.5, "role_novelty": 0.0},
                "B4": {"coverage": 0.4, "diversity": 0.4, "role_novelty": 0.2},
            }
            weights_for_method = (score_weights or {}).get(method, defaults[method])
            score = (
                float(weights_for_method.get("coverage", 0.0)) * coverage
                + float(weights_for_method.get("diversity", 0.0)) * diversity
                + float(weights_for_method.get("role_novelty", 0.0)) * role_novelty
            )
            components = {
                "delta_coverage": coverage,
                "clip_diversity": diversity,
                "role_novelty": role_novelty,
                "objective": method,
                "objective_weights": weights_for_method,
            }
            # Candidate list is sorted by stable ID; strict > preserves deterministic tie-break.
            if score > best_score + 1e-12:
                best, best_score, best_components = candidate, score, components

        if best is None:
            break
        selected.append(best)
        selected_keys.add(normalize_text(best["text"]))
        covered_facts.update(_candidate_facts(best))
        covered_categories.update(_candidate_categories(best))
        trace.append({
            "step": len(selected), "candidate_id": best["candidate_id"],
            "text": best["text"], "candidate_type": best.get("candidate_type"),
            "score": float(best_score), "score_components": best_components,
            "source_caption_ids": best.get("source_caption_ids", []),
            "source_spans": best.get("source_spans", []),
            "categories": sorted(_candidate_categories(best)),
            "facts": sorted(_candidate_facts(best)),
        })

    # If fewer than five unique valid candidates remain, use source originals as explicit fallback.
    all_originals = sorted(
        [x for x in candidates if x.get("candidate_type") == "original"],
        key=lambda x: str(x.get("candidate_id", "")),
    )
    fallback_index = 0
    while len(selected) < 5 and all_originals:
        fallback = all_originals[fallback_index % len(all_originals)]
        fallback_index += 1
        selected.append(fallback)
        trace.append({
            "step": len(selected), "candidate_id": fallback.get("candidate_id"),
            "text": fallback["text"], "candidate_type": "original",
            "score": None, "score_components": {"fallback": True},
            "source_caption_ids": fallback.get("source_caption_ids", []),
            "source_spans": fallback.get("source_spans", []),
            "categories": sorted(_candidate_categories(fallback)),
            "facts": sorted(_candidate_facts(fallback)),
        })
    while len(selected) < 5:
        fallback_text = raw_captions[(len(selected)) % len(raw_captions)]
        selected.append({
            "candidate_id": "raw_fallback_{}".format(len(selected)),
            "text": fallback_text, "candidate_type": "original",
            "facts": sorted(content_tokens(fallback_text)),
            "categories": classify_semantic_roles(fallback_text)["categories"],
            "source_caption_ids": [], "source_spans": [],
        })
        trace.append({
            "step": len(selected), "candidate_id": selected[-1]["candidate_id"],
            "text": fallback_text, "candidate_type": "original", "score": None,
            "score_components": {"fallback": "raw_caption_contract"},
            "source_caption_ids": [], "source_spans": [],
        })

    return [x["text"] for x in selected[:5]], trace
