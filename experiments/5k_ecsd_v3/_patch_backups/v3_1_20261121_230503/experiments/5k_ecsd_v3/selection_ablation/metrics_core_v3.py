"""Pure metric functions for ECSD v3; no CLIP model loading in this module."""
from itertools import combinations
import numpy as np

from candidate_quality.semantic_role_classifier import classify_semantic_roles, content_tokens, quality_flags
from selection_ablation.selection_methods_v3 import build_fact_weights, weighted_jaccard
from utils.text_utils import canonical_surface_text

PAIRS = list(combinations(range(5), 2))
ROLES = ["Entity", "Attribute", "Action", "Relation", "Context"]


def compute_caption_set_metrics(captions, raw_captions, embedding_by_text, candidate_types=None, filter_config=None):
    if len(captions) != 5 or not all(isinstance(x, str) and x.strip() for x in captions):
        raise ValueError("Each image must have exactly five non-empty captions")
    vectors = [np.asarray(embedding_by_text[x], dtype=np.float32).reshape(-1) for x in captions]
    similarities = []
    for i, j in PAIRS:
        denom = float(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
        sim = float(np.dot(vectors[i], vectors[j]) / denom) if denom else 0.0
        similarities.append(max(-1.0, min(1.0, sim)))
    mean_similarity = float(np.mean(similarities))
    cisd = 1.0 - mean_similarity

    fact_freq, fact_weights = build_fact_weights(raw_captions)
    source_facts = set(fact_freq)
    selected_facts = [content_tokens(x) for x in captions]
    covered = set().union(*selected_facts)
    denominator = sum(fact_weights.get(x, 1.0) for x in source_facts) or 1.0
    coverage = sum(fact_weights.get(x, 1.0) for x in (covered & source_facts)) / denominator

    categories = set()
    validity = []
    candidate_types = list(candidate_types or ["original"] * 5)
    if len(candidate_types) != 5:
        raise ValueError("candidate_types must have five entries")
    for text, candidate_type in zip(captions, candidate_types):
        categories.update(x for x in classify_semantic_roles(text)["categories"] if x != "Unknown")
        validity.append(not quality_flags(text, candidate_type, filter_config))
    category_counts = {role: 0 for role in ROLES}
    for text in captions:
        labels = classify_semantic_roles(text)["categories"]
        for role in ROLES:
            category_counts[role] += int(role in labels)
    category_coverage = len(categories) / float(len(ROLES))

    redundancy = float(np.mean([
        weighted_jaccard(selected_facts[i], selected_facts[j], fact_weights)
        for i, j in PAIRS
    ]))
    raw_surfaces = {canonical_surface_text(x) for x in raw_captions}
    exact_match_count = sum(canonical_surface_text(x) in raw_surfaces for x in captions)
    type_counts = {name: 0 for name in ("original", "clause", "noun_phrase", "fallback")}
    for kind in (candidate_types or []):
        type_counts[kind if kind in type_counts else "fallback"] += 1

    return {
        "cisd": cisd,
        "mean_pairwise_similarity": mean_similarity,
        "weighted_fact_coverage": float(coverage),
        "role_coverage": float(category_coverage),
        "role_count": len(categories),
        "redundancy_weighted_jaccard": redundancy,
        "raw_exact_match_rate": exact_match_count / 5.0,
        "raw_exact_match_count": exact_match_count,
        "candidate_validity_proxy": float(np.mean(validity)),
        "unique_semantic_fact_count": len(covered),
        "candidate_type_original_count": type_counts["original"],
        "candidate_type_clause_count": type_counts["clause"],
        "candidate_type_noun_phrase_count": type_counts["noun_phrase"],
        "candidate_type_fallback_count": type_counts["fallback"],
        **{"role_{}_caption_count".format(k.lower()): v for k, v in category_counts.items()},
    }
