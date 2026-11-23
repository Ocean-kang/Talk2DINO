#!/usr/bin/env python3
"""Build a traceable ECSD v2 candidate pool from Raw Captions only."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_utils import load_config

try:
    from .semantic_role_classifier import (
        ACTION_TERMS, ATTRIBUTE_TERMS, CONNECTOR_TERMS, ENTITY_TERMS,
        WORD_RE, classify_semantic_roles, content_tokens, quality_flags,
        normalize_text,
    )
except ImportError:  # direct script execution from the experiment folder
    from semantic_role_classifier import (
        ACTION_TERMS, ATTRIBUTE_TERMS, CONNECTOR_TERMS, ENTITY_TERMS,
        WORD_RE, classify_semantic_roles, content_tokens, quality_flags,
        normalize_text,
    )

DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "some"}
PREPOSITIONS = {
    "in", "on", "at", "to", "from", "for", "by", "as", "into", "onto",
    "with", "without", "over", "under", "above", "below", "near", "beside",
    "behind", "between", "around", "through", "across", "along", "inside",
    "outside", "of",
}
CLAUSE_MARKERS = {"and", "or", "but", "while", "which", "who", "that"}
AUXILIARIES = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had"}

def sentence_form(text):
    text = re.sub(r"\s+", " ", str(text).strip(" \t\r\n,;:.!?"))
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."

def _trim_span(text, start, end):
    while start < end and (text[start].isspace() or text[start] in ",;:.!?"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ",;:.!?"):
        end -= 1
    # Remove dangling conjunctions/prepositions while preserving exact source offsets.
    changed = True
    while changed and start < end:
        changed = False
        prefix = re.match(r"(?i)\s*([A-Za-z]+)\b", text[start:end])
        if prefix and prefix.group(1).lower() in CLAUSE_MARKERS | PREPOSITIONS:
            start += prefix.end()
            while start < end and text[start].isspace():
                start += 1
            changed = True
        suffix = re.search(r"(?i)\b([A-Za-z]+)\s*$", text[start:end])
        if suffix and suffix.group(1).lower() in CLAUSE_MARKERS | PREPOSITIONS:
            end = start + suffix.start()
            while end > start and text[end - 1].isspace():
                end -= 1
            changed = True
    return start, end, text[start:end].strip()

def _has_action(text):
    tokens = {x.lower() for x in WORD_RE.findall(text)}
    return bool(tokens & ACTION_TERMS)

def _add_candidate(out, caption_index, caption, start, end, candidate_type, filter_config=None):
    start, end, span_text = _trim_span(caption, start, end)
    if not span_text:
        return
    if candidate_type == "noun_phrase":
        text = sentence_form("A photo shows " + span_text[0].lower() + span_text[1:])
    else:
        text = sentence_form(span_text)
    facts = sorted(content_tokens(text))
    if not facts:
        return
    role_info = classify_semantic_roles(text)
    flags = quality_flags(text, candidate_type, filter_config)
    source_facts = content_tokens(span_text)
    unsupported = sorted(set(facts) - set(source_facts))
    candidate_id = "c{:02d}_{}_{}_{}".format(
        caption_index, candidate_type, start, end
    )
    out.append({
        "candidate_id": candidate_id,
        "text": text,
        "candidate_type": candidate_type,
        "source_caption_ids": [caption_index],
        "source_spans": [{
            "caption_index": caption_index,
            "start": int(start),
            "end": int(end),
            "text": span_text,
        }],
        "facts": facts,
        "categories": role_info["categories"],
        "category_rules": role_info["rules"],
        "quality_flags": flags,
        "heuristic_valid": not flags,
        "unsupported_content_tokens": unsupported,
        "role": role_info["categories"][0] if len(role_info["categories"]) == 1 else (
            "multi" if role_info["categories"] != ["Unknown"] else "Unknown"
        ),
    })

def _clause_segments(caption):
    """Punctuation splits plus conservative conjunction splits when both sides have verbs."""
    for match in re.finditer(r"[^,;:.!?]+", caption):
        raw = match.group(0)
        left_trim = len(raw) - len(raw.lstrip())
        start = match.start() + left_trim
        end = match.start() + len(raw.rstrip())
        segment = caption[start:end]
        if not segment:
            continue
        split_points = []
        for conj in re.finditer(r"\b(?:and|while|but)\b", segment, flags=re.I):
            left = segment[:conj.start()].strip()
            right = segment[conj.end():].strip()
            if left and right and _has_action(left) and _has_action(right):
                split_points.append((start + conj.start(), start + conj.end()))
        if not split_points:
            yield start, end
            continue
        cursor = start
        for a, b in split_points:
            if a > cursor:
                yield cursor, a
            cursor = b
        if cursor < end:
            yield cursor, end

def build_improved_candidates(raw_captions, config=None):
    """Return candidate objects with exact source spans and explicit quality flags."""
    candidates = []
    config = config or {}
    construction = config.get("candidate_construction", {})
    filter_config = config.get("candidate_filter", {})
    min_clause_tokens = int(construction.get("min_clause_content_tokens", 3))
    max_np_tokens = int(construction.get("max_np_tokens", 8))
    min_np_tokens = int(construction.get("min_np_content_tokens", 1))
    max_candidates = max(len(raw_captions), int(construction.get("max_candidates_per_image", 200)))
    captions = [str(x).strip() for x in raw_captions]

    # Preserve every original caption first, even when the optional candidate cap is reached.
    for caption_index, caption in enumerate(captions):
        if caption:
            _add_candidate(candidates, caption_index, caption, 0, len(caption), "original", filter_config)

    # Additional candidates are added only after all five original anchors are retained.
    for caption_index, caption in enumerate(captions):
        if not caption or len(candidates) >= max_candidates:
            break

        for start, end in _clause_segments(caption):
            if len(candidates) >= max_candidates:
                break
            fragment = caption[start:end].strip()
            first = WORD_RE.search(fragment)
            if not first or first.group(0).lower() in CLAUSE_MARKERS:
                continue
            if first.group(0).lower() in ACTION_TERMS:
                continue
            if len(content_tokens(fragment)) >= min_clause_tokens:
                _add_candidate(candidates, caption_index, caption, start, end, "clause", filter_config)

        tokens = list(WORD_RE.finditer(caption))
        for idx, token in enumerate(tokens):
            if len(candidates) >= max_candidates:
                break
            if token.group(0).lower() not in DETERMINERS:
                continue
            end_idx = idx + 1
            while end_idx < len(tokens) and (end_idx - idx) <= max_np_tokens:
                word = tokens[end_idx].group(0).lower()
                gap = caption[tokens[end_idx - 1].end():tokens[end_idx].start()]
                if re.search(r"[,;:.!?]", gap):
                    break
                if word in PREPOSITIONS | CLAUSE_MARKERS | ACTION_TERMS | AUXILIARIES | DETERMINERS:
                    break
                end_idx += 1
            if end_idx <= idx + 1:
                continue
            last = tokens[end_idx - 1]
            phrase = caption[token.start():last.end()]
            facts = content_tokens(phrase)
            if len(facts) < min_np_tokens:
                continue
            # Keep one-word noun heads only when the local entity inventory supports them.
            if len(facts) == 1 and not (facts & ENTITY_TERMS):
                continue
            if not facts or facts <= ATTRIBUTE_TERMS:
                continue
            _add_candidate(candidates, caption_index, caption, token.start(), last.end(), "noun_phrase", filter_config)

    # Candidate IDs are deterministic within an image. Duplicates are retained for diagnosis.
    return candidates[:max_candidates]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--output", default="outputs/5k_ecsd_next/candidates/ecsd_v2_candidates.jsonl")
    parser.add_argument("--summary", default="outputs/5k_ecsd_next/candidates/candidate_pool_summary.json")
    parser.add_argument("--expected-images", type=int, default=5000)
    parser.add_argument("--config", default="experiments/5k_ecsd_next/configs/default.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    out_path = Path(args.output)
    summary_path = Path(args.summary)
    if not source.is_file():
        raise FileNotFoundError("Raw caption JSON not found: {}".format(source))
    if out_path.exists() and not args.overwrite:
        print("[A] Existing candidate pool found. Validating for reuse...")

        source_rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
        source_map = {int(x["image_id"]): x["raw_captions"] for x in source_rows}

        if len(source_map) != args.expected_images:
            raise RuntimeError("Source image count mismatch.")

        pool_ids = set()
        total_candidates = 0

        with out_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                row = json.loads(line)
                image_id = int(row["image_id"])

                if image_id in pool_ids:
                    raise RuntimeError(f"Duplicate image_id: {image_id}")

                if image_id not in source_map:
                    raise RuntimeError(f"Unknown image_id: {image_id}")

                if row["raw_captions"] != source_map[image_id]:
                    raise RuntimeError(f"Raw captions mismatch: {image_id}")

                candidates = row.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    raise RuntimeError(f"Invalid candidates: {image_id}")

                for candidate in candidates:
                    if not isinstance(candidate, dict) or not str(candidate.get("text", "")).strip():
                        raise RuntimeError(f"Invalid candidate text: {image_id}")

                pool_ids.add(image_id)
                total_candidates += len(candidates)

        if pool_ids != set(source_map):
            raise RuntimeError(
                f"Candidate pool image IDs mismatch: {len(pool_ids)}/{len(source_map)}"
            )

        print("[PASS] Existing candidate pool is valid.")
        print(f"[INFO] Images: {len(pool_ids)}")
        print(f"[INFO] Candidates: {total_candidates}")
        print("[A] Reusing existing candidate pool. Skipping generation.")
        return
    config = load_config(args.config)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list.")
    if args.expected_images and len(rows) != args.expected_images:
        raise ValueError("Expected {} images, got {}.".format(args.expected_images, len(rows)))

    seen_ids = set()
    counts = Counter()
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if "image_id" not in row:
                raise ValueError("Record {} missing image_id.".format(index))
            image_id = int(row["image_id"])
            if image_id in seen_ids:
                raise ValueError("Duplicate image_id {}".format(image_id))
            seen_ids.add(image_id)
            raw = row.get("raw_captions")
            if not isinstance(raw, list) or len(raw) != 5 or not all(
                isinstance(x, str) and x.strip() for x in raw
            ):
                raise ValueError("image_id={} must have five non-empty raw_captions.".format(image_id))
            candidates = build_improved_candidates(raw, config)
            for item in candidates:
                counts[item["candidate_type"]] += 1
            total += len(candidates)
            handle.write(json.dumps({
                "image_id": image_id,
                "raw_captions": raw,
                "candidates": candidates,
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 500 == 0:
                print("[candidate-pool] {}/{}".format(index + 1, len(rows)), flush=True)

    summary = {
        "num_images": len(rows),
        "total_candidates": total,
        "mean_candidates_per_image": total / max(1, len(rows)),
        "candidate_type_counts": dict(counts),
        "candidate_pool": str(out_path),
        "evidence_source": "raw_captions only",
        "heuristic_labels_are_not_human_ground_truth": True,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[DONE] candidate pool:", out_path)
    print("[DONE] summary:", summary_path)

if __name__ == "__main__":
    main()
