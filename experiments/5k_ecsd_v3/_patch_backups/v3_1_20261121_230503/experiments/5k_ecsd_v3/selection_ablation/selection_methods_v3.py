#!/usr/bin/env python3
"""Deterministic B1-B4 selection over one deduplicated ECSD v3 pool."""
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
from candidate_quality.semantic_role_classifier import (
    classify_semantic_roles, content_tokens, quality_flags,
)
from utils.text_utils import canonical_surface_text

METHODS = {
    "B1": "Coverage Only",
    "B2": "CLIP Diversity Only",
    "B3": "Coverage + Diversity",
    "B4": "Coverage + Diversity + Role Novelty",
}
TYPE_PRIORITY = {"original": 0, "clause": 1, "noun_phrase": 2}
DEFAULT_WEIGHTS = {
    "B1": {"coverage": 1.0, "diversity": 0.0, "role_novelty": 0.0},
    "B2": {"coverage": 0.0, "diversity": 1.0, "role_novelty": 0.0},
    "B3": {"coverage": 0.5, "diversity": 0.5, "role_novelty": 0.0},
    "B4": {"coverage": 0.4, "diversity": 0.4, "role_novelty": 0.2},
}


def build_fact_weights(raw_captions):
    frequency = Counter()
    for caption in raw_captions:
        frequency.update(content_tokens(caption))
    n = max(1, len(raw_captions))
    weights = {token: 1.0 + math.log((n + 1.0) / (count + 1.0)) for token, count in frequency.items()}
    return frequency, weights


def weighted_jaccard(a, b, weights):
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return 0.0
    numerator = sum(weights.get(x, 1.0) for x in a & b)
    denominator = sum(weights.get(x, 1.0) for x in union)
    return numerator / denominator if denominator else 0.0


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def candidate_facts(candidate):
    return set(candidate.get("facts") or content_tokens(candidate.get("text", "")))


def candidate_categories(candidate):
    cats = candidate.get("categories")
    if cats is None:
        cats = classify_semantic_roles(candidate.get("text", ""))["categories"]
    return {x for x in cats if x != "Unknown"}


def valid_candidates(candidates, filter_config=None):
    """Validate unique v3 IDs/texts and apply one shared filter in stable order."""
    ids = [x.get("candidate_id") for x in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate pool contains duplicate candidate_id values")
    texts = [canonical_surface_text(x.get("text", "")) for x in candidates]
    if len(texts) != len(set(texts)):
        raise ValueError("Candidate pool contains duplicate canonical surface texts")
    ordered = sorted(
        candidates,
        key=lambda x: (TYPE_PRIORITY.get(x.get("candidate_type"), 99), str(x.get("candidate_id", ""))),
    )
    result = []
    for item in ordered:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        flags = quality_flags(text, item.get("candidate_type", "candidate"), filter_config)
        if flags or not candidate_facts(item):
            continue
        result.append(item)
    return result


def choose_anchor(candidates, raw_captions, filter_config=None):
    """Deterministic max weighted-fact-mass valid Original; ID breaks exact ties."""
    valid = valid_candidates(candidates, filter_config)
    originals = [x for x in valid if x.get("candidate_type") == "original"]
    pool = originals if originals else valid
    if not pool:
        raise ValueError("No valid candidates for anchor")
    _, weights = build_fact_weights(raw_captions)
    denominator = sum(weights.values()) or 1.0
    ranked = sorted(
        pool,
        key=lambda x: (
            -sum(weights.get(t, 1.0) for t in candidate_facts(x)) / denominator,
            str(x["candidate_id"]),
        ),
    )
    return ranked[0]


def select_five(candidates, raw_captions, embedding_by_text, method,
                score_weights=None, filter_config=None, anchor_candidate_id=None):
    if method not in METHODS:
        raise ValueError("Unknown method: {}".format(method))
    valid = valid_candidates(candidates, filter_config)
    if not valid:
        raise ValueError("No valid candidates after common filter")
    by_id = {x["candidate_id"]: x for x in valid}
    if anchor_candidate_id is None:
        anchor = choose_anchor(valid, raw_captions, filter_config)
    else:
        if anchor_candidate_id not in by_id:
            raise ValueError("Shared anchor {} is not in the filtered candidate pool".format(anchor_candidate_id))
        anchor = by_id[anchor_candidate_id]
    if anchor["text"] not in embedding_by_text:
        raise KeyError("Missing CLIP embedding for anchor text")

    weights = dict(DEFAULT_WEIGHTS[method])
    if score_weights and method in score_weights:
        weights.update(score_weights[method])
    _, fact_weights = build_fact_weights(raw_captions)
    total_weight = sum(fact_weights.values()) or 1.0

    selected = [anchor]
    selected_ids = {anchor["candidate_id"]}
    selected_texts = {canonical_surface_text(anchor["text"])}
    covered_facts = set(candidate_facts(anchor))
    covered_roles = set(candidate_categories(anchor))
    trace = [{
        "step": 1,
        "candidate_id": anchor["candidate_id"],
        "text": anchor["text"],
        "candidate_type": anchor["candidate_type"],
        "score": None,
        "score_components": {"anchor": "shared deterministic max weighted-fact-mass Original"},
        "source_caption_ids": anchor.get("source_caption_ids", []),
        "source_spans": anchor.get("source_spans", []),
        "categories": sorted(candidate_categories(anchor)),
        "facts": sorted(candidate_facts(anchor)),
    }]

    while len(selected) < 5:
        best = None
        best_score = -float("inf")
        best_components = None
        for candidate in valid:  # stable candidate order is the tie-break
            candidate_id = candidate["candidate_id"]
            surface = canonical_surface_text(candidate["text"])
            if candidate_id in selected_ids or surface in selected_texts:
                continue
            new_facts = candidate_facts(candidate) - covered_facts
            coverage = (
                sum(fact_weights.get(t, 1.0) for t in new_facts) / total_weight
                if new_facts else 0.0
            )
            vector = np.asarray(embedding_by_text[candidate["text"]], dtype=np.float32)
            distances = [
                (1.0 - max(-1.0, min(1.0, cosine(vector, embedding_by_text[x["text"]])))) / 2.0
                for x in selected
            ]
            diversity = sum(distances) / len(distances) if distances else 0.0
            roles = candidate_categories(candidate)
            role_novelty = len(roles - covered_roles) / len(roles) if roles else 0.0
            score = (
                float(weights.get("coverage", 0.0)) * coverage
                + float(weights.get("diversity", 0.0)) * diversity
                + float(weights.get("role_novelty", 0.0)) * role_novelty
            )
            components = {
                "delta_coverage": coverage,
                "clip_diversity": diversity,
                "role_novelty": role_novelty,
                "objective": method,
                "objective_weights": weights,
            }
            if score > best_score + 1e-12:
                best, best_score, best_components = candidate, score, components
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best["candidate_id"])
        selected_texts.add(canonical_surface_text(best["text"]))
        covered_facts.update(candidate_facts(best))
        covered_roles.update(candidate_categories(best))
        trace.append({
            "step": len(selected),
            "candidate_id": best["candidate_id"],
            "text": best["text"],
            "candidate_type": best["candidate_type"],
            "score": float(best_score),
            "score_components": best_components,
            "source_caption_ids": best.get("source_caption_ids", []),
            "source_spans": best.get("source_spans", []),
            "categories": sorted(candidate_categories(best)),
            "facts": sorted(candidate_facts(best)),
        })

    # If filter removes too many items, fill from unselected Original evidence deterministically.
    if len(selected) < 5:
        originals = sorted(
            [x for x in candidates if x.get("candidate_type") == "original"],
            key=lambda x: str(x.get("candidate_id", "")),
        )
        for item in originals:
            surface = canonical_surface_text(item["text"])
            if item["candidate_id"] in selected_ids or surface in selected_texts:
                continue
            selected.append(item)
            selected_ids.add(item["candidate_id"])
            selected_texts.add(surface)
            trace.append({
                "step": len(selected),
                "candidate_id": item["candidate_id"],
                "text": item["text"],
                "candidate_type": "original",
                "score": None,
                "score_components": {"fallback": "unfiltered_original_contract"},
                "source_caption_ids": item.get("source_caption_ids", []),
                "source_spans": item.get("source_spans", []),
                "categories": sorted(candidate_categories(item)),
                "facts": sorted(candidate_facts(item)),
            })
            if len(selected) == 5:
                break
    if len(selected) != 5:
        raise RuntimeError(
            "Cannot select five distinct candidates; got {}. Check duplicate Raw Captions / filters.".format(len(selected))
        )
    return [x["text"] for x in selected], trace
