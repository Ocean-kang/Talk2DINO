#!/usr/bin/env python3
"""Generate the five-caption ECSD-naive set for every subset5K image."""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from clip_utils import encode_texts, load_clip_model
from ecsd_core import build_candidates, build_fact_weights, weighted_jaccard

REQUIRED_FIELDS = (
    "raw_captions",
    "rich_captions",
    "rich_weaken_captions",
    "rich_diversity_captions",
)


def load_records(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of image records.")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "image_id" not in row:
            raise ValueError("Record {} is missing image_id.".format(index))
        image_id = int(row["image_id"])
        if image_id in seen:
            raise ValueError("Duplicate image_id: {}".format(image_id))
        seen.add(image_id)
        for field in REQUIRED_FIELDS:
            captions = row.get(field)
            if not isinstance(captions, list) or len(captions) != 5:
                raise ValueError(
                    "image_id={} field={} must contain exactly five captions.".format(
                        image_id, field
                    )
                )
            if not all(isinstance(x, str) and x.strip() for x in captions):
                raise ValueError(
                    "image_id={} field={} has an empty or invalid caption.".format(
                        image_id, field
                    )
                )
    return rows


def select_five(candidates, source_captions, embedding_by_text):
    frequency, weights = build_fact_weights(source_captions)
    total_weight = max(sum(weights.values()), 1e-8)
    core_facts = {token for token, count in frequency.items() if count >= 2}
    if not core_facts:
        core_facts = set(weights)

    originals = [c for c in candidates if c["candidate_type"] == "original"]
    if not originals:
        raise RuntimeError("Candidate construction did not retain source captions.")

    def anchor_score(candidate):
        facts = set(candidate["facts"])
        core_coverage = sum(weights.get(t, 1.0) for t in facts & core_facts)
        core_total = max(sum(weights.get(t, 1.0) for t in core_facts), 1e-8)
        action_bonus = 0.08 if candidate["role"] == "action" else 0.0
        return core_coverage / core_total + action_bonus + 0.02 * candidate["quality"]

    first = max(originals, key=anchor_score)
    selected = [first]
    selected_texts = {first["text"]}
    covered = set(first["facts"])
    selected_roles = {first["role"]}

    score_trace = [{
        "candidate_id": first["candidate_id"],
        "text": first["text"],
        "candidate_type": first["candidate_type"],
        "role": first["role"],
        "score": float(anchor_score(first)),
        "score_components": {"anchor_core_coverage": float(anchor_score(first))},
        "source_caption_ids": first["source_caption_ids"],
        "source_spans": first["source_spans"],
    }]

    while len(selected) < 5:
        best = None
        best_score = -float("inf")
        best_components = None
        selected_embeddings = torch.stack(
            [embedding_by_text[item["text"]] for item in selected], dim=0
        )

        for candidate in candidates:
            if candidate["text"] in selected_texts:
                continue
            facts = set(candidate["facts"])
            new_facts = facts - covered
            delta_coverage = sum(weights.get(t, 1.0) for t in new_facts) / total_weight

            candidate_embedding = embedding_by_text[candidate["text"]]
            similarities = torch.mv(selected_embeddings, candidate_embedding)
            clip_distance = float(
                ((1.0 - similarities).clamp(0.0, 2.0) / 2.0).mean().item()
            )
            role_novelty = 1.0 if candidate["role"] not in selected_roles else 0.0
            redundancy = max(
                weighted_jaccard(candidate["facts"], item["facts"], weights)
                for item in selected
            )
            quality = float(candidate["quality"])
            score = (
                0.45 * delta_coverage
                + 0.30 * clip_distance
                + 0.15 * role_novelty
                + 0.10 * quality
                - 0.20 * redundancy
            )
            if score > best_score:
                best = candidate
                best_score = score
                best_components = {
                    "delta_coverage": float(delta_coverage),
                    "clip_distance": float(clip_distance),
                    "role_novelty": float(role_novelty),
                    "quality": float(quality),
                    "weighted_jaccard_redundancy": float(redundancy),
                }

        if best is None:
            break
        selected.append(best)
        selected_texts.add(best["text"])
        covered.update(best["facts"])
        selected_roles.add(best["role"])
        score_trace.append({
            "candidate_id": best["candidate_id"],
            "text": best["text"],
            "candidate_type": best["candidate_type"],
            "role": best["role"],
            "score": float(best_score),
            "score_components": best_components,
            "source_caption_ids": best["source_caption_ids"],
            "source_spans": best["source_spans"],
        })

    # If the source set itself contains fewer than five unique sentences and
    # no additional phrase candidates can be formed, preserve the exact
    # five-caption contract rather than inventing unsupported content.
    fallback_index = 0
    while len(selected) < 5:
        fallback = originals[fallback_index % len(originals)]
        fallback_index += 1
        selected.append(fallback)
        score_trace.append({
            "candidate_id": fallback["candidate_id"],
            "text": fallback["text"],
            "candidate_type": fallback["candidate_type"],
            "role": fallback["role"],
            "score": None,
            "score_components": {"fallback": True},
            "source_caption_ids": fallback["source_caption_ids"],
            "source_spans": fallback["source_spans"],
        })

    return [item["text"] for item in selected[:5]], score_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--output", default="data/coco2014_subset_50k/5k_ECSD.json")
    parser.add_argument("--trace", default="outputs/5k_ecsd/ecsd_selection_trace.jsonl")
    parser.add_argument("--summary", default="outputs/5k_ecsd/generation_summary.json")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-cache", default="~/.cache/clip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError("Input captions JSON not found: {}".format(input_path))
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            "{} already exists. Pass --overwrite to regenerate it.".format(output_path)
        )

    rows = load_records(input_path)
    per_image_candidates = []
    unique_texts = {}
    for row_index, row in enumerate(rows):
        candidates = build_candidates(row["raw_captions"])
        per_image_candidates.append(candidates)
        for candidate in candidates:
            unique_texts.setdefault(candidate["text"], None)
        if (row_index + 1) % 500 == 0:
            print("[ECSD] constructed candidates for {}/{} images".format(
                row_index + 1, len(rows)
            ), flush=True)

    texts = list(unique_texts)
    print("[ECSD] unique candidate texts: {}".format(len(texts)), flush=True)
    clip_module, model, device = load_clip_model(
        args.clip_model, args.device, args.clip_cache
    )
    embeddings = encode_texts(
        texts, clip_module, model, device, batch_size=args.batch_size
    )
    embedding_by_text = {
        text: embeddings[index] for index, text in enumerate(texts)
    }

    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    type_counts = Counter()
    role_counts = Counter()
    candidate_count_total = 0
    with trace_path.open("w", encoding="utf-8") as trace_file:
        for index, (row, candidates) in enumerate(zip(rows, per_image_candidates)):
            candidate_count_total += len(candidates)
            selected, trace = select_five(
                candidates, row["raw_captions"], embedding_by_text
            )
            row["ecsd_captions"] = selected
            for item in trace:
                type_counts[item["candidate_type"]] += 1
                role_counts[item["role"]] += 1
            trace_file.write(json.dumps({
                "image_id": int(row["image_id"]),
                "candidate_count": len(candidates),
                "selected": trace,
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 500 == 0:
                print("[ECSD] selected captions for {}/{} images".format(
                    index + 1, len(rows)
                ), flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "ok",
        "algorithm": "ECSD-naive-v1",
        "num_images": len(rows),
        "captions_per_image": 5,
        "candidate_count_total": candidate_count_total,
        "candidate_count_mean": candidate_count_total / max(len(rows), 1),
        "selected_candidate_types": dict(type_counts),
        "selected_roles": dict(role_counts),
        "clip_model": args.clip_model,
        "device": device,
        "input": str(input_path),
        "output": str(output_path),
        "trace": str(trace_path),
        "constraints": {
            "llm": False,
            "cocu": False,
            "f6": False,
            "bootstrap": False,
            "image_encoder_for_caption_generation": False,
            "evidence_source": "raw_captions only",
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[DONE] ECSD JSON: {}".format(output_path))
    print("[DONE] Selection trace: {}".format(trace_path))
    print("[DONE] Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
