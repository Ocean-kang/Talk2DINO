# ECSD_V3_1_FALLBACK_PATCH
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
    # Deterministic fallback order: valid Original > any Original > valid candidate > any candidate.
    all_candidates = list(candidates)
    if not all_candidates:
        raise ValueError("No candidates available for anchor")
    valid = valid_candidates(all_candidates, filter_config)
    valid_originals = [x for x in valid if x.get("candidate_type") == "original"]
    all_originals = [x for x in all_candidates if x.get("candidate_type") == "original"]
    if valid_originals:
        pool, rule = valid_originals, "valid_original"
    elif all_originals:
        pool, rule = all_originals, "relaxed_original"
    elif valid:
        pool, rule = valid, "valid_non_original"
    else:
        pool, rule = all_candidates, "relaxed_any_candidate"
    _, fact_weights = build_fact_weights(raw_captions)
    denominator = sum(fact_weights.values()) or 1.0
    ranked = sorted(
        pool,
        key=lambda x: (
            -sum(fact_weights.get(t, 1.0) for t in candidate_facts(x)) / denominator,
            TYPE_PRIORITY.get(x.get("candidate_type"), 99),
            str(x.get("candidate_id", "")),
        ),
    )
    chosen = dict(ranked[0])
    chosen["_anchor_rule"] = rule
    return chosen


def select_five(candidates, raw_captions, embedding_by_text, method,
                score_weights=None, filter_config=None, anchor_candidate_id=None):
    # Normal valid -> relaxed distinct -> explicit duplicate slots.
    if method not in METHODS:
        raise ValueError("Unknown method: {}".format(method))
    all_candidates = list(candidates)
    if not all_candidates:
        raise ValueError("No candidates available for image")

    ids = [x.get("candidate_id") for x in all_candidates]
    surfaces = [canonical_surface_text(x.get("text", "")) for x in all_candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate pool contains duplicate candidate_id values")
    if len(surfaces) != len(set(surfaces)):
        raise ValueError("Candidate pool contains duplicate canonical surface texts")

    all_by_id = {x["candidate_id"]: x for x in all_candidates}
    normal = valid_candidates(all_candidates, filter_config)
    normal_ids = {x["candidate_id"] for x in normal}

    if anchor_candidate_id is None:
        anchor = choose_anchor(all_candidates, raw_captions, filter_config)
    else:
        if anchor_candidate_id not in all_by_id:
            raise ValueError("Shared anchor {} is absent from candidate pool".format(anchor_candidate_id))
        anchor = all_by_id[anchor_candidate_id]
    if anchor["text"] not in embedding_by_text:
        raise KeyError("Missing CLIP embedding for anchor text: " + anchor["text"])

    objective_weights = dict(DEFAULT_WEIGHTS[method])
    if score_weights and method in score_weights:
        objective_weights.update(score_weights[method])
    _, fact_weights = build_fact_weights(raw_captions)
    total_weight = sum(fact_weights.values()) or 1.0

    selected = []
    selected_ids = set()
    selected_surfaces = set()
    covered_facts = set()
    covered_roles = set()
    trace = []

    def append_slot(candidate, stage, fallback_type="none", reason=None,
                    duplicate_of=None, score=None, components=None):
        slot = len(selected) + 1
        flags = quality_flags(candidate["text"], candidate.get("candidate_type", "candidate"), filter_config)
        trace.append({
            "step": slot,
            "slot_id": "slot_{:02d}".format(slot),
            "candidate_id": candidate["candidate_id"],
            "text": candidate["text"],
            "candidate_type": candidate.get("candidate_type", "candidate"),
            "score": score,
            "score_components": components or {},
            "source_caption_ids": candidate.get("source_caption_ids", []),
            "source_spans": candidate.get("source_spans", []),
            "categories": sorted(candidate_categories(candidate)),
            "facts": sorted(candidate_facts(candidate)),
            "quality_flags": flags,
            "selection_stage": stage,
            "fallback_type": fallback_type,
            "fallback_reason": reason,
            "duplicate_of_candidate_id": duplicate_of,
        })
        selected.append(candidate)
        selected_ids.add(candidate["candidate_id"])
        selected_surfaces.add(canonical_surface_text(candidate["text"]))
        covered_facts.update(candidate_facts(candidate))
        covered_roles.update(candidate_categories(candidate))

    def objective(candidate):
        new_facts = candidate_facts(candidate) - covered_facts
        coverage = sum(fact_weights.get(t, 1.0) for t in new_facts) / total_weight if new_facts else 0.0
        vector = np.asarray(embedding_by_text[candidate["text"]], dtype=np.float32)
        distances = [
            (1.0 - max(-1.0, min(1.0, cosine(vector, embedding_by_text[x["text"]])))) / 2.0
            for x in selected
        ]
        diversity = sum(distances) / len(distances) if distances else 0.0
        roles = candidate_categories(candidate)
        role_novelty = len(roles - covered_roles) / len(roles) if roles else 0.0
        score = (
            float(objective_weights.get("coverage", 0.0)) * coverage
            + float(objective_weights.get("diversity", 0.0)) * diversity
            + float(objective_weights.get("role_novelty", 0.0)) * role_novelty
        )
        return score, {
            "delta_coverage": coverage,
            "clip_diversity": diversity,
            "role_novelty": role_novelty,
            "objective": method,
            "objective_weights": objective_weights,
        }

    def greedy_fill(pool, stage, fallback_type="none", reason=None):
        while len(selected) < 5:
            remaining = [
                x for x in pool
                if x["candidate_id"] not in selected_ids
                and canonical_surface_text(x["text"]) not in selected_surfaces
            ]
            if not remaining:
                break
            best, best_score, best_components = None, -float("inf"), None
            for candidate in remaining:
                if candidate["text"] not in embedding_by_text:
                    raise KeyError("Missing CLIP embedding for candidate: " + candidate["text"])
                score, components = objective(candidate)
                if score > best_score + 1e-12:
                    best, best_score, best_components = candidate, score, components
            append_slot(best, stage, fallback_type, reason,
                        score=float(best_score), components=best_components)

    anchor_is_normal = anchor["candidate_id"] in normal_ids
    append_slot(
        anchor, "anchor",
        fallback_type="none" if anchor_is_normal else "relaxed_quality_filter",
        reason=None if anchor_is_normal else "shared_anchor_failed_common_quality_filter",
        components={
            "anchor_rule": anchor.get("_anchor_rule", "shared_precomputed_anchor"),
            "anchor_policy": "one precomputed candidate_id shared by B1-B4",
        },
    )

    # Stage 1: candidates passing the common filter.
    normal_pool = sorted(
        normal,
        key=lambda x: (TYPE_PRIORITY.get(x.get("candidate_type"), 99), str(x["candidate_id"])),
    )
    greedy_fill(normal_pool, "normal")

    # Stage 2: rejected distinct candidates, strict type priority Original > Clause > NP.
    if len(selected) < 5:
        rejected = [x for x in all_candidates if x["candidate_id"] not in normal_ids]
        for kind in ("original", "clause", "noun_phrase"):
            group = sorted(
                [x for x in rejected if x.get("candidate_type") == kind],
                key=lambda x: str(x["candidate_id"]),
            )
            greedy_fill(group, "relaxed", "relaxed_quality_filter",
                        "candidate_failed_common_quality_filter")
            if len(selected) == 5:
                break
        if len(selected) < 5:
            known = {"original", "clause", "noun_phrase"}
            other = sorted(
                [x for x in rejected if x.get("candidate_type") not in known],
                key=lambda x: (str(x.get("candidate_type")), str(x["candidate_id"])),
            )
            greedy_fill(other, "relaxed", "relaxed_quality_filter",
                        "candidate_failed_common_quality_filter")

    # Stage 3: only when fewer than five distinct texts exist in the entire pool.
    if len(selected) < 5:
        originals = [x for x in all_candidates if x.get("candidate_type") == "original"]
        representative_pool = originals or selected or all_candidates
        ranked = sorted(
            representative_pool,
            key=lambda x: (
                -sum(fact_weights.get(t, 1.0) for t in candidate_facts(x)),
                TYPE_PRIORITY.get(x.get("candidate_type"), 99),
                str(x.get("candidate_id", "")),
            ),
        )
        representative = ranked[0]
        while len(selected) < 5:
            append_slot(
                representative, "duplicate_fallback", "duplicate_slot",
                "insufficient_distinct_candidates",
                duplicate_of=representative["candidate_id"],
                components={"fallback": "explicit_duplicate_slot", "objective": method},
            )

    if len(selected) != 5 or len(trace) != 5:
        raise AssertionError("Expected five caption slots, got {}".format(len(selected)))
    if len({x["slot_id"] for x in trace}) != 5:
        raise AssertionError("slot_id values must be unique")
    return [x["text"] for x in selected], trace
