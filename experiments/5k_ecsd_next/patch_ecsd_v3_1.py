#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ECSD v3.1 patcher. Run from Talk2DINO root: python patch_ecsd_v3_1.py

import argparse
import ast
import shutil
from datetime import datetime
from pathlib import Path

REL = Path("experiments/5k_ecsd_v3")
MARKER = "ECSD_V3_1_FALLBACK_PATCH"


def stop(message):
    raise SystemExit("[STOP] " + message)


def read(path):
    return path.read_text(encoding="utf-8")


def backup_write(path, content, backup_root):
    old = read(path)
    if old == content:
        print("[SKIP] unchanged:", path)
        return
    repo_root = backup_root.parents[3]
    backup = backup_root / path.relative_to(repo_root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    tmp = path.with_suffix(path.suffix + ".v31.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    print("[PATCH]", path, "backup:", backup)


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        stop("{}: expected one source block, found {}; no fuzzy edit performed".format(label, count))
    return text.replace(old, new, 1)


SELECTION = r'''
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
'''


def patch_selection(path):
    text = read(path)
    if MARKER in text:
        print("[SKIP] selection module already patched")
        return text
    start = text.find("def choose_anchor(")
    if start < 0:
        stop("choose_anchor not found")
    text = text[:start] + SELECTION.strip() + "\n"
    text = "# " + MARKER + "\n" + text
    ast.parse(text, filename=str(path))
    return text




def patch_generator(path):
    text = read(path)
    if "anchor = choose_anchor(candidates, raw, filter_config)" not in text:
        text = replace_once(
            text,
            "anchor = choose_anchor(valid, raw, filter_config)",
            "anchor = choose_anchor(candidates, raw, filter_config)",
            "generator anchor fallback",
        )
    if '"fallback_summary": fallback_summary' not in text:
        old = '''    metadata = {
        "version": "ECSD-v3",
'''
        new = '''    fallback_summary = {
        method: {
            "duplicate_slot_count": sum(
                1 for row in trace_rows[method] for item in row["selected"]
                if item.get("fallback_type") == "duplicate_slot"
            ),
            "relaxed_quality_filter_count": sum(
                1 for row in trace_rows[method] for item in row["selected"]
                if item.get("fallback_type") == "relaxed_quality_filter"
            ),
        }
        for method in methods
    }
    metadata = {
        "version": "ECSD-v3.1",
        "fallback_summary": fallback_summary,
'''
        text = replace_once(text, old, new, "generator fallback metadata")
    ast.parse(text, filename=str(path))
    return text


def patch_validator(path):
    text = read(path)
    if 'slot_ids = [x.get("slot_id"' in text:
        print("[SKIP] validator already patched")
        return text
    old = '''            selected_ids = [x.get("candidate_id") for x in selected]
            if len(selected_ids) != len(set(selected_ids)):
                raise ValueError("{} image_id={} repeats candidate IDs".format(method, image_id))
            candidate_map = {x["candidate_id"]: x for x in pool_map[image_id]["candidates"]}
            for item in selected:
                cid = item.get("candidate_id")
                if cid not in candidate_map:
                    raise ValueError("{} image_id={} trace references unknown candidate_id {}".format(method, image_id, cid))
                if item.get("text") != candidate_map[cid].get("text"):
                    raise ValueError("{} image_id={} trace text differs from candidate pool for {}".format(method, image_id, cid))
'''
    new = '''            slot_ids = [x.get("slot_id", "slot_{:02d}".format(int(x.get("step", i + 1)))) for i, x in enumerate(selected)]
            if len(slot_ids) != 5 or len(set(slot_ids)) != 5:
                raise ValueError("{} image_id={} must have five unique slot_id values".format(method, image_id))
            candidate_map = {x["candidate_id"]: x for x in pool_map[image_id]["candidates"]}
            seen_ids = set()
            for item in selected:
                cid = item.get("candidate_id")
                if cid not in candidate_map:
                    raise ValueError("{} image_id={} trace references unknown candidate_id {}".format(method, image_id, cid))
                if item.get("text") != candidate_map[cid].get("text"):
                    raise ValueError("{} image_id={} trace text differs from candidate pool for {}".format(method, image_id, cid))
                fallback_type = item.get("fallback_type", "none")
                if cid in seen_ids:
                    if fallback_type != "duplicate_slot" or item.get("duplicate_of_candidate_id") != cid:
                        raise ValueError("{} image_id={} repeated candidate_id {} without explicit duplicate_slot trace".format(method, image_id, cid))
                elif fallback_type == "duplicate_slot":
                    raise ValueError("{} image_id={} marks first occurrence of {} as duplicate_slot".format(method, image_id, cid))
                seen_ids.add(cid)
'''
    text = replace_once(text, old, new, "validator selected-ID rules")
    old = '''        "shared_anchor_mismatches": anchor_mismatch_count,
        "v3_alias_rows_validated": alias_count,
'''
    new = '''        "shared_anchor_mismatches": anchor_mismatch_count,
        "duplicate_fallback_slots": sum(
            1 for method in methods for row in trace_maps[method].values()
            for item in row.get("selected", []) if item.get("fallback_type") == "duplicate_slot"
        ),
        "relaxed_fallback_slots": sum(
            1 for method in methods for row in trace_maps[method].values()
            for item in row.get("selected", []) if item.get("fallback_type") == "relaxed_quality_filter"
        ),
        "v3_alias_rows_validated": alias_count,
'''
    text = replace_once(text, old, new, "validator fallback report")
    text = text.replace(
        '"five unique selected candidate IDs per image",',
        '"five unique slot IDs per image; repeated candidate IDs require explicit duplicate_slot trace",',
    )
    ast.parse(text, filename=str(path))
    return text



def patch_exporter(path):
    text = read(path)
    if "selected_slot_multiplicity" in text:
        print("[SKIP] manual audit exporter already patched")
        return text
    old = '''        selected_ids = [x.get("candidate_id") for x in selected_items]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("B4 trace repeats candidate IDs for image_id={}".format(image_id))
'''
    new = '''        slot_ids = [x.get("slot_id", "slot_{:02d}".format(int(x.get("step", i + 1)))) for i, x in enumerate(selected_items)]
        if len(slot_ids) != 5 or len(set(slot_ids)) != 5:
            raise ValueError("B4 trace must contain five unique slots for image_id={}".format(image_id))
        selected_ids = [x.get("candidate_id") for x in selected_items]
        seen_ids = set()
        for item in selected_items:
            cid = item.get("candidate_id")
            fallback_type = item.get("fallback_type", "none")
            if cid in seen_ids:
                if fallback_type != "duplicate_slot" or item.get("duplicate_of_candidate_id") != cid:
                    raise ValueError("Repeated B4 candidate_id lacks explicit duplicate_slot for image_id={}".format(image_id))
            elif fallback_type == "duplicate_slot":
                raise ValueError("First B4 occurrence cannot be marked duplicate_slot for image_id={}".format(image_id))
            seen_ids.add(cid)
'''
    text = replace_once(text, old, new, "manual audit duplicate-slot validation")
    text = replace_once(
        text,
        '''        selected_set = set(selected_ids)
        selected_texts = [x["text"] for x in selected_items]
        candidate_view = []
''',
        '''        selected_set = set(selected_ids)
        selected_texts = [x["text"] for x in selected_items]
        selected_multiplicity = {cid: selected_ids.count(cid) for cid in selected_set}
        candidate_view = []
''',
        "manual audit multiplicity map",
    )
    text = replace_once(
        text,
        '''                "selection_status": "Selected" if is_selected else "Not selected",
                **{field: "" for field in LABEL_FIELDS},
''',
        '''                "selection_status": "Selected" if is_selected else "Not selected",
                "selected_slot_multiplicity": selected_multiplicity.get(candidate["candidate_id"], 0),
                **{field: "" for field in LABEL_FIELDS},
''',
        "manual audit CSV multiplicity",
    )
    text = replace_once(
        text,
        '''                "selected": is_selected,
                "quality_flags": candidate.get("quality_flags", []),
''',
        '''                "selected": is_selected,
                "selected_slot_multiplicity": selected_multiplicity.get(candidate["candidate_id"], 0),
                "quality_flags": candidate.get("quality_flags", []),
''',
        "manual audit viewer multiplicity",
    )
    text = replace_once(
        text,
        '''        if sum(x["selection_status"] == "Selected" for x in output if x["image_id"] == image_id) != 5:
            raise AssertionError("Selected-by-ID count must be exactly five for image_id={}".format(image_id))
        viewer.append({
            "image_id": image_id,
            "raw_captions": raw,
            "selected_captions": selected_texts,
            "selected_candidate_ids": selected_ids,
            "candidates": candidate_view,
        })
''',
        '''        if sum(x["selection_status"] == "Selected" for x in output if x["image_id"] == image_id) != len(selected_set):
            raise AssertionError("Selected candidate rows must equal unique selected IDs for image_id={}".format(image_id))
        viewer.append({
            "image_id": image_id,
            "raw_captions": raw,
            "selected_captions": selected_texts,
            "selected_candidate_ids": selected_ids,
            "selected_slot_count": len(selected_items),
            "duplicate_slot_count": sum(x.get("fallback_type") == "duplicate_slot" for x in selected_items),
            "candidates": candidate_view,
        })
''',
        "manual audit selected-row count",
    )
    text = replace_once(
        text,
        '''        "source_spans", "predicted_categories", "classification_rules", "selection_status",
''',
        '''        "source_spans", "predicted_categories", "classification_rules", "selection_status",
        "selected_slot_multiplicity",
''',
        "manual audit CSV columns",
    )
    ast.parse(text, filename=str(path))
    return text


def patch_metrics_core(path):
    text = read(path)
    if "fallback_duplicate_rate" in text:
        print("[SKIP] metrics core already patched")
        return text
    text = replace_once(
        text,
        "def compute_caption_set_metrics(captions, raw_captions, embedding_by_text, candidate_types=None, filter_config=None):",
        "def compute_caption_set_metrics(captions, raw_captions, embedding_by_text, candidate_types=None, filter_config=None, fallback_types=None):",
        "metrics signature",
    )
    old = '''    type_counts = {name: 0 for name in ("original", "clause", "noun_phrase", "fallback")}
    for kind in (candidate_types or []):
        type_counts[kind if kind in type_counts else "fallback"] += 1

    return {
'''
    new = '''    type_counts = {name: 0 for name in ("original", "clause", "noun_phrase", "fallback")}
    for kind in (candidate_types or []):
        type_counts[kind if kind in type_counts else "fallback"] += 1
    fallback_types = list(fallback_types or ["none"] * 5)
    if len(fallback_types) != 5:
        raise ValueError("fallback_types must have five entries")
    duplicate_fallback_count = sum(x == "duplicate_slot" for x in fallback_types)
    relaxed_fallback_count = sum(x == "relaxed_quality_filter" for x in fallback_types)

    return {
'''
    text = replace_once(text, old, new, "metrics fallback counters")
    text = replace_once(
        text,
        '''        "candidate_validity_proxy": float(np.mean(validity)),
        "unique_semantic_fact_count": len(covered),
''',
        '''        "candidate_validity_proxy": float(np.mean(validity)),
        "fallback_duplicate_count": duplicate_fallback_count,
        "fallback_duplicate_rate": duplicate_fallback_count / 5.0,
        "fallback_relaxed_count": relaxed_fallback_count,
        "fallback_relaxed_rate": relaxed_fallback_count / 5.0,
        "unique_semantic_fact_count": len(covered),
''',
        "metrics fallback output",
    )
    ast.parse(text, filename=str(path))
    return text


def patch_evaluator(path):
    text = read(path)
    if "fallback_maps" in text:
        print("[SKIP] evaluator already patched")
        return text
    text = replace_once(text, "from collections import defaultdict",
                        "from collections import Counter, defaultdict",
                        "evaluator Counter import")
    text = replace_once(text, "    trace_maps = {}\n    type_maps = {}\n",
                        "    trace_maps = {}\n    type_maps = {}\n    fallback_maps = {}\n",
                        "evaluator maps")
    old = '''        type_maps[method] = {
            int(x["image_id"]): [z.get("candidate_type", "fallback") for z in x["selected"]]
            for x in traces
        }
'''
    new = '''        type_maps[method] = {
            int(x["image_id"]): [z.get("candidate_type", "fallback") for z in x["selected"]]
            for x in traces
        }
        fallback_maps[method] = {
            int(x["image_id"]): [z.get("fallback_type", "none") for z in x["selected"]]
            for x in traces
        }
'''
    text = replace_once(text, old, new, "evaluator fallback map")
    text = replace_once(
        text,
        '                filter_config=config.get("candidate_filter", {}),\n            )',
        '                filter_config=config.get("candidate_filter", {}),\n'
        '                fallback_types=fallback_maps.get(arm, {}).get(image_id, ["none"] * 5),\n'
        '            )',
        "evaluator metric call",
    )
    old = '''                left_ids = {x["candidate_id"] for x in trace_maps[left][image_id]["selected"]}
                right_ids = {x["candidate_id"] for x in trace_maps[right][image_id]["selected"]}
                shared_total += len(left_ids & right_ids)
                exact += int(left_ids == right_ids)
'''
    new = '''                left_ids = Counter(x["candidate_id"] for x in trace_maps[left][image_id]["selected"])
                right_ids = Counter(x["candidate_id"] for x in trace_maps[right][image_id]["selected"])
                shared_total += sum((left_ids & right_ids).values())
                exact += int(left_ids == right_ids)
'''
    text = replace_once(text, old, new, "evaluator multiset comparison")
    text = replace_once(
        text,
        '"version": "ECSD-v3",',
        '"version": "ECSD-v3.1",\n        "fallback_metrics": "duplicate_slot and relaxed_quality_filter counts/rates come from final traces; CISD includes all five slots including duplicates",',
        "evaluator metadata version",
    )
    ast.parse(text, filename=str(path))
    return text


AUDIT_SCRIPT = r'''#!/usr/bin/env python3
# CPU-only candidate shortage audit.
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_utils import load_config
from selection_ablation.selection_methods_v3 import valid_candidates
from candidate_quality.semantic_role_classifier import quality_flags
from utils.text_utils import canonical_surface_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--candidate-pool", default="outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl")
    parser.add_argument("--config", default="experiments/5k_ecsd_v3/configs/default.json")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_v3")
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    source_map = {int(x["image_id"]): x for x in source}
    pool = []
    with Path(args.candidate_pool).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    pool.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSONL line {}: {}".format(line_no, exc))
    pool_map = {int(x["image_id"]): x for x in pool}
    if len(source_map) != len(source) or set(source_map) != set(pool_map):
        raise ValueError("Source and candidate-pool image IDs are not a one-to-one match")
    config = load_config(args.config)
    filter_config = config.get("candidate_filter", {})
    rows = []
    for image_id in sorted(source_map):
        raw = source_map[image_id].get("raw_captions", [])
        candidates = pool_map[image_id].get("candidates", [])
        if raw != pool_map[image_id].get("raw_captions"):
            raise ValueError("Raw Caption mismatch for image_id={}".format(image_id))
        valid = valid_candidates(candidates, filter_config)
        valid_ids = {x["candidate_id"] for x in valid}
        surfaces = {canonical_surface_text(x.get("text", "")) for x in candidates}
        raw_surfaces = {canonical_surface_text(x) for x in raw}
        reasons = Counter()
        for item in candidates:
            if item["candidate_id"] not in valid_ids:
                flags = quality_flags(item.get("text", ""), item.get("candidate_type", "candidate"), filter_config)
                reasons.update(flags or ["other_filter"])
        relaxed = len(candidates) - len(valid_ids)
        rows.append({
            "image_id": image_id,
            "raw_caption_count": len(raw),
            "raw_unique_count": len(raw_surfaces),
            "candidate_pool_count": len(candidates),
            "candidate_unique_count": len(surfaces),
            "valid_candidate_count": len(valid_ids),
            "relaxed_candidate_count": relaxed,
            "invalid_candidate_count": relaxed,
            "invalid_reasons": json.dumps(dict(reasons), ensure_ascii=False, sort_keys=True),
            "fallback_required": int(len(surfaces) < 5),
        })
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "selection_shortage_audit_v3.json"
    csv_path = out / "selection_shortage_summary_v3.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print("[DONE] images={} valid<5={} distinct<5={}".format(
        len(rows), sum(x["valid_candidate_count"] < 5 for x in rows),
        sum(x["fallback_required"] for x in rows)))
    print("[DONE] JSON:", json_path)
    print("[DONE] CSV:", csv_path)


if __name__ == "__main__":
    main()
'''


TEST_SCRIPT = r'''import sys
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from candidate_quality.semantic_role_classifier import content_tokens, classify_semantic_roles
from candidate_quality.build_candidate_pool_v3 import build_candidates
from candidate_quality.export_manual_audit_v3 import build_audit_rows
from selection_ablation.selection_methods_v3 import METHODS, choose_anchor, select_five
from selection_ablation.metrics_core_v3 import compute_caption_set_metrics
from selection_ablation.validate_generation_v3 import validate_generation, METHOD_FILES

RAW = [
    "A man rides a bicycle near a tree.",
    "A dog runs across green grass.",
    "A woman holds a blue umbrella.",
    "A plate of food sits on a wooden table.",
    "A child throws a ball in a park.",
]
FILTER = {"max_words": 36, "min_content_tokens": {
    "original": 2, "clause": 2, "noun_phrase": 1, "candidate": 2
}}
WEIGHTS = {
    "B1": {"coverage": 1.0, "diversity": 0.0, "role_novelty": 0.0},
    "B2": {"coverage": 0.0, "diversity": 1.0, "role_novelty": 0.0},
    "B3": {"coverage": 0.5, "diversity": 0.5, "role_novelty": 0.0},
    "B4": {"coverage": 0.4, "diversity": 0.4, "role_novelty": 0.2},
}


def candidate(i, text, kind="original"):
    return {
        "candidate_id": "c{}".format(i), "text": text, "candidate_type": kind,
        "facts": sorted(content_tokens(text)),
        "categories": classify_semantic_roles(text)["categories"],
        "source_caption_ids": [min(i, 4)], "source_spans": [],
    }


def make_embeddings(candidates):
    result = {}
    for x in candidates:
        v = np.array([len(x["text"]) % 11 + 1, len(x["text"].split()) + 1,
                      sum(map(ord, x["text"])) % 97 + 1], dtype=np.float32)
        result[x["text"]] = v / np.linalg.norm(v)
    return result


class FallbackSelectionTests(unittest.TestCase):
    def test_five_normal(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW)]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 5)
        self.assertTrue(all(x["fallback_type"] == "none" for x in trace))

    def test_relaxed_candidate_fills_fifth(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4] + ["and a dog"])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B1", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 5)
        self.assertEqual(trace[-1]["fallback_type"], "relaxed_quality_filter")

    def test_four_distinct_use_duplicate_slot(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B2", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["slot_id"] for x in trace}), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 4)
        self.assertEqual(trace[-1]["fallback_type"], "duplicate_slot")

    def test_one_candidate_fills_five_slots(self):
        cs = [candidate(0, RAW[0])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        out, trace = select_five(cs, RAW, emb, "B4", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(len({x["candidate_id"] for x in trace}), 1)
        self.assertEqual(len({x["slot_id"] for x in trace}), 5)
        self.assertEqual(sum(x["fallback_type"] == "duplicate_slot" for x in trace), 4)

    def test_all_raw_duplicates_do_not_crash(self):
        cs = [candidate(0, RAW[0])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, [RAW[0]] * 5, FILTER)
        out, trace = select_five(cs, [RAW[0]] * 5, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"])
        self.assertEqual(len(out), 5)
        self.assertEqual(sum(x["fallback_type"] == "duplicate_slot" for x in trace), 4)

    def test_b1_b4_shared_precomputed_anchor(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        anchors = []
        for method in METHODS:
            _, trace = select_five(cs, RAW, emb, method, WEIGHTS, FILTER, anchor["candidate_id"])
            anchors.append(trace[0]["candidate_id"])
        self.assertEqual(len(set(anchors)), 1)

    def test_selection_is_deterministic(self):
        cs = [candidate(i, x) for i, x in enumerate(RAW[:4])]
        emb = make_embeddings(cs)
        anchor = choose_anchor(cs, RAW, FILTER)
        self.assertEqual(
            select_five(cs, RAW, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"]),
            select_five(cs, RAW, emb, "B3", WEIGHTS, FILTER, anchor["candidate_id"]),
        )

    def test_cisd_keeps_duplicate_slots(self):
        repeated = [RAW[0]] * 5
        emb = make_embeddings([candidate(0, RAW[0])])
        metrics = compute_caption_set_metrics(
            repeated, RAW, emb, ["original"] * 5, FILTER,
            fallback_types=["none", "duplicate_slot", "duplicate_slot", "duplicate_slot", "duplicate_slot"],
        )
        self.assertAlmostEqual(metrics["cisd"], 0.0, places=6)
        self.assertEqual(metrics["fallback_duplicate_count"], 4)
        self.assertAlmostEqual(metrics["fallback_duplicate_rate"], 0.8)


class ValidatorDuplicateSlotTests(unittest.TestCase):
    def test_trace_and_json_accept_explicit_duplicate_slots(self):
        raw = [RAW[0]] * 5
        config = {
            "candidate_construction": {"min_clause_content_tokens": 2, "max_np_tokens": 8,
                                       "min_np_content_tokens": 1, "max_candidates_per_image": 100},
            "candidate_filter": FILTER,
        }
        candidates, _ = build_candidates(raw, 77, config)
        chosen = next(x for x in candidates if x["candidate_type"] == "original")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_path, pool_path = root / "source.json", root / "pool.jsonl"
            cap_dir, trace_dir = root / "captions", root / "traces"
            cap_dir.mkdir()
            trace_dir.mkdir()
            source_row = {"image_id": 77, "raw_captions": raw,
                          "rich_captions": raw, "rich_diversity_captions": raw}
            source_path.write_text(json.dumps([source_row]), encoding="utf-8")
            pool_path.write_text(json.dumps({"image_id": 77, "raw_captions": raw,
                                             "candidates": candidates}) + "\n", encoding="utf-8")
            for method, (filename, field) in METHOD_FILES.items():
                selected = []
                for slot in range(1, 6):
                    selected.append({
                        "step": slot, "slot_id": "slot_{:02d}".format(slot),
                        "candidate_id": chosen["candidate_id"], "text": chosen["text"],
                        "candidate_type": "original",
                        "fallback_type": "none" if slot == 1 else "duplicate_slot",
                        "duplicate_of_candidate_id": None if slot == 1 else chosen["candidate_id"],
                    })
                row = dict(source_row)
                row[field] = [chosen["text"]] * 5
                (cap_dir / filename).write_text(json.dumps([row]), encoding="utf-8")
                trace = {"image_id": 77, "method": method,
                         "anchor_candidate_id": chosen["candidate_id"], "selected": selected}
                (trace_dir / (method + "_selection_trace.jsonl")).write_text(
                    json.dumps(trace) + "\n", encoding="utf-8")
            alias_row = dict(source_row)
            alias_row["ecsd_v3_captions"] = [chosen["text"]] * 5
            (cap_dir / "5k_ECSD_v3.json").write_text(json.dumps([alias_row]), encoding="utf-8")
            report = validate_generation(source_path, pool_path, cap_dir, trace_dir, expected_images=1)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["duplicate_fallback_slots"], 16)


    def test_manual_audit_accepts_duplicate_slots_and_reports_multiplicity(self):
        raw = [RAW[0]] * 5
        config = {
            "candidate_construction": {"min_clause_content_tokens": 2, "max_np_tokens": 8,
                                       "min_np_content_tokens": 1, "max_candidates_per_image": 100},
            "candidate_filter": FILTER,
        }
        candidates, _ = build_candidates(raw, 88, config)
        chosen = next(x for x in candidates if x["candidate_type"] == "original")
        source_rows = [{"image_id": 88, "raw_captions": raw}]
        pool_rows = [{"image_id": 88, "raw_captions": raw, "candidates": candidates}]
        selected = []
        for slot in range(1, 6):
            selected.append({
                "step": slot, "slot_id": "slot_{:02d}".format(slot),
                "candidate_id": chosen["candidate_id"], "text": chosen["text"],
                "candidate_type": "original",
                "fallback_type": "none" if slot == 1 else "duplicate_slot",
                "duplicate_of_candidate_id": None if slot == 1 else chosen["candidate_id"],
            })
        trace_rows = [{"image_id": 88, "method": "B4",
                       "anchor_candidate_id": chosen["candidate_id"], "selected": selected}]
        rows, viewer, _ = build_audit_rows(source_rows, pool_rows, trace_rows, 1, 1)
        chosen_row = next(x for x in rows if x["candidate_id"] == chosen["candidate_id"])
        self.assertEqual(chosen_row["selection_status"], "Selected")
        self.assertEqual(chosen_row["selected_slot_multiplicity"], 5)
        self.assertEqual(viewer[0]["selected_slot_count"], 5)
        self.assertEqual(viewer[0]["duplicate_slot_count"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

'''


def main():
    parser = argparse.ArgumentParser(description="Patch ECSD v3 to v3.1 fallback behavior")
    parser.add_argument("--root", default=".", help="Talk2DINO repository root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    pkg = root / REL
    files = {
        "selection": pkg / "selection_ablation/selection_methods_v3.py",
        "generator": pkg / "selection_ablation/generate_ablation_captions_v3.py",
        "validator": pkg / "selection_ablation/validate_generation_v3.py",
        "metrics": pkg / "selection_ablation/metrics_core_v3.py",
        "evaluator": pkg / "selection_ablation/evaluate_metrics_v3.py",
        "exporter": pkg / "candidate_quality/export_manual_audit_v3.py",
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        stop("ECSD v3 source files not found. Run from Talk2DINO root. Missing: " + ", ".join(missing))

    edits = [
        (files["selection"], patch_selection(files["selection"])),
        (files["generator"], patch_generator(files["generator"])),
        (files["validator"], patch_validator(files["validator"])),
        (files["metrics"], patch_metrics_core(files["metrics"])),
        (files["evaluator"], patch_evaluator(files["evaluator"])),
        (files["exporter"], patch_exporter(files["exporter"])),
    ]
    for path, content in edits:
        ast.parse(content, filename=str(path))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = pkg / "_patch_backups" / ("v3_1_" + stamp)
    for path, content in edits:
        backup_write(path, content, backup_root)

    audit = pkg / "selection_ablation/audit_candidate_shortage_v3.py"
    if not audit.exists():
        audit.write_text(AUDIT_SCRIPT, encoding="utf-8")
        print("[ADD]", audit)
    else:
        print("[KEEP] existing audit script:", audit)

    test_file = pkg / "tests/test_ecsd_v3_1_fallback.py"
    if not test_file.exists():
        test_file.write_text(TEST_SCRIPT, encoding="utf-8")
        print("[ADD]", test_file)
    else:
        print("[KEEP] existing tests:", test_file)

    readme = pkg / "README.md"
    readme_text = read(readme)
    if "ECSD v3.1 fallback behavior" not in readme_text:
        readme_text += r'''

## ECSD v3.1 fallback behavior

1. Normal selection uses candidates passing the common quality filter.
2. Relaxed selection uses rejected distinct candidates in deterministic priority Original > Clause > Noun Phrase. Their quality flags remain in Trace.
3. Explicit duplicate slots are used only when the entire pool has fewer than five distinct texts. Trace records `fallback_type=duplicate_slot`, `duplicate_of_candidate_id`, and a unique `slot_id`.

CISD is calculated over all five final slots, including duplicates. Duplicate slots are never removed to inflate diversity. The evaluator reports duplicate and relaxed fallback counts/rates.

CPU-only shortage audit:
```bash
python experiments/5k_ecsd_v3/selection_ablation/audit_candidate_shortage_v3.py
```

Validate and rerun Stage B:
```bash
bash experiments/5k_ecsd_v3/scripts/validate_package_v3.sh
OVERWRITE=1 bash experiments/5k_ecsd_v3/scripts/run_experiment_b_v3.sh
```

Modified source files are backed up under `experiments/5k_ecsd_v3/_patch_backups/`. v2 files, manual annotations, model weights, and training code are not modified.
'''
        readme_backup = backup_root / "experiments/5k_ecsd_v3/README.md"
        readme_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(readme, readme_backup)
        readme.write_text(readme_text, encoding="utf-8")

    (pkg / "VERSION").write_text("3.1\n", encoding="utf-8")
    print("\n[PATCH COMPLETE] ECSD v3.1 applied.")
    print("Backups:", backup_root)
    print("Next commands:")
    print("  python experiments/5k_ecsd_v3/selection_ablation/audit_candidate_shortage_v3.py")
    print("  bash experiments/5k_ecsd_v3/scripts/validate_package_v3.sh")
    print("  OVERWRITE=1 bash experiments/5k_ecsd_v3/scripts/run_experiment_b_v3.sh")


if __name__ == "__main__":
    main()
