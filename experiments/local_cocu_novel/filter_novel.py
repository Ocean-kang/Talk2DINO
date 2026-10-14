#!/usr/bin/env python3
"""Remove concepts described by a target image's original COCO captions.

Runs BEFORE the existing Local-CoCu head matching, avoiding wasted teacher
selection on already-described concepts. Uses lexical phrase boundaries, not a
learned semantic similarity cutoff; the decision is auditable and deterministic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Iterable

WORD = re.compile(r"[a-z0-9]+")
IRREGULAR = {
    "men": "man", "women": "woman", "people": "person",
    "children": "child", "mice": "mouse", "geese": "goose",
    "teeth": "tooth", "feet": "foot", "oxen": "ox",
}


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(WORD.findall(text.casefold()))


def canonical_token(word: str, exact_only: bool = False) -> str:
    """Conservative English singularization for common noun forms.

    Does NOT promise semantic equivalence; for precise lexical-only matching,
    use ``--exact-only``. Same rule is used again during dataset validation.
    """
    if exact_only:
        return word
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies") and word[-4] not in "aeiou":
        return word[:-3] + "y"  # puppies -> puppy
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zzes")):
        return word[:-2]  # classes -> class, dishes -> dish
    if len(word) > 4 and word.endswith("ses") and word[:-2] in ("bus", "gas"):
        return word[:-2]  # buses -> bus
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ous")):
        return word[:-1]  # dogs -> dog
    return word


def norm_tokens(text: str, exact_only: bool = False) -> tuple[str, ...]:
    return tuple(canonical_token(t, exact_only) for t in tokenize(text))


def in_raw_captions(concept: str, captions: Iterable[str], *, exact_only: bool = False) -> bool:
    """Exact contiguous phrase match within ONE caption, using word boundaries."""
    needle = norm_tokens(concept, exact_only)
    if not needle:
        return False
    n = len(needle)
    for caption in captions:
        tokens = norm_tokens(str(caption), exact_only)
        if any(tokens[i:i + n] == needle for i in range(len(tokens) - n + 1)):
            return True
    return False


def filter_one(candidates: Iterable[str], captions: Iterable[str], *, exact_only: bool = False):
    captions = list(captions)
    kept, removed, seen = [], [], set()
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        name = " ".join(tokenize(candidate))
        key = norm_tokens(name, exact_only)
        if not key or key in seen:
            continue
        seen.add(key)
        if in_raw_captions(name, captions, exact_only=exact_only):
            removed.append(name)
        else:
            kept.append(name)
    return kept, removed


def load_original_captions(path: Path, keep_first: int = 5):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "annotations" not in payload:
        raise ValueError(f"Not a COCO captions JSON: {path}")
    grouped = defaultdict(list)
    for ann in payload["annotations"]:
        grouped[str(int(ann["image_id"]))].append(str(ann["caption"]))
    return {k: v[:keep_first] for k, v in grouped.items()}


def run_filter(candidate_path: Path, raw_path: Path, output: Path, audit_path: Path,
               keep_first: int = 5, exact_only: bool = False):
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, dict):
        raise ValueError("Candidates must be a dictionary keyed by image ID")
    original = load_original_captions(raw_path, keep_first)
    filtered, audit = {}, {}
    stats = Counter()
    for raw_id, pool in candidates.items():
        image_id = str(int(raw_id))
        if image_id not in original or len(original[image_id]) < keep_first:
            raise ValueError(f"Image {image_id}: fewer than {keep_first} raw captions")
        if not isinstance(pool, list):
            raise ValueError(f"Image {image_id}: candidates must be a list")
        kept, removed = filter_one(pool, original[image_id], exact_only=exact_only)
        filtered[image_id] = kept
        audit[image_id] = {
            "input_count": len(pool), "removed_count": len(removed),
            "novel_count": len(kept), "removed": removed,
        }
        stats["images"] += 1
        stats["candidates_input"] += len(pool)
        stats["removed"] += len(removed)
        stats["novel"] += len(kept)
        stats["images_without_novel"] += int(not kept)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps({
        "settings": {"keep_first": keep_first, "exact_only": exact_only,
                     "definition": "contiguous normalized token phrase in original captions"},
        "summary": dict(stats), "images": audit,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[FILTER]", dict(stats))
    print("[OUTPUT]", output)
    print("[AUDIT]", audit_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--raw-captions", default="data/coco2014/annotations/captions_train2014.json")
    p.add_argument("--output", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--keep-first", type=int, default=5)
    p.add_argument("--exact-only", action="store_true", help="Disable conservative plural/irregular noun matching")
    a = p.parse_args()
    run_filter(Path(a.candidates), Path(a.raw_captions), Path(a.output),
               Path(a.audit), a.keep_first, a.exact_only)


if __name__ == "__main__":
    main()
