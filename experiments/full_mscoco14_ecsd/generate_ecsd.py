#!/usr/bin/env python3
"""Generate full-scale ECSD captions with the exact 5k_ecsd selection algorithm.

The implementation is image-chunked so candidate embeddings are not held for the
entire COCO corpus. A JSONL trace is the resume journal and is flushed after each
image chunk.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ECSD_5K = ROOT / "experiments" / "5k_ecsd"
if str(ECSD_5K) not in sys.path:
    sys.path.insert(0, str(ECSD_5K))

from clip_utils import encode_texts, load_clip_model  # noqa: E402
from ecsd_core import build_candidates, build_fact_weights, content_tokens, weighted_jaccard  # noqa: E402


def raw_digest(captions):
    payload = json.dumps(captions, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list")
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "image_id" not in row:
            raise ValueError("Record {} missing image_id".format(i))
        image_id = int(row["image_id"])
        if image_id in seen:
            raise ValueError("Duplicate image_id {}".format(image_id))
        seen.add(image_id)
        captions = row.get("raw_captions")
        if not isinstance(captions, list) or len(captions) != 5:
            raise ValueError("image_id={} must have exactly five raw_captions".format(image_id))
        if not all(isinstance(x, str) and x.strip() for x in captions):
            raise ValueError("image_id={} contains empty/non-string caption".format(image_id))
    return rows


def select_five(candidates, source_captions, embedding_by_text):
    """Selection logic copied without algorithmic changes from 5k_ecsd."""
    frequency, weights = build_fact_weights(source_captions)
    total_weight = max(sum(weights.values()), 1e-8)
    core_facts = {token for token, count in frequency.items() if count >= 2}
    if not core_facts:
        core_facts = set(weights)

    originals = [c for c in candidates if c["candidate_type"] == "original"]
    if not originals:
        raise RuntimeError("Candidate construction did not retain source captions")

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
    trace = [{
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
        best, best_score, best_components = None, -float("inf"), None
        selected_embeddings = torch.stack([embedding_by_text[x["text"]] for x in selected])
        for candidate in candidates:
            if candidate["text"] in selected_texts:
                continue
            facts = set(candidate["facts"])
            new_facts = facts - covered
            delta_coverage = sum(weights.get(t, 1.0) for t in new_facts) / total_weight
            candidate_embedding = embedding_by_text[candidate["text"]]
            similarities = torch.mv(selected_embeddings, candidate_embedding)
            clip_distance = float(((1.0 - similarities).clamp(0.0, 2.0) / 2.0).mean().item())
            role_novelty = 1.0 if candidate["role"] not in selected_roles else 0.0
            redundancy = max(
                weighted_jaccard(candidate["facts"], item["facts"], weights)
                for item in selected
            )
            quality = float(candidate["quality"])
            score = (
                0.45 * delta_coverage + 0.30 * clip_distance
                + 0.15 * role_novelty + 0.10 * quality - 0.20 * redundancy
            )
            if score > best_score:
                best, best_score = candidate, score
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
        trace.append({
            "candidate_id": best["candidate_id"],
            "text": best["text"],
            "candidate_type": best["candidate_type"],
            "role": best["role"],
            "score": float(best_score),
            "score_components": best_components,
            "source_caption_ids": best["source_caption_ids"],
            "source_spans": best["source_spans"],
        })

    fallback_index = 0
    while len(selected) < 5:
        fallback = originals[fallback_index % len(originals)]
        fallback_index += 1
        selected.append(fallback)
        trace.append({
            "candidate_id": fallback["candidate_id"],
            "text": fallback["text"],
            "candidate_type": fallback["candidate_type"],
            "role": fallback["role"],
            "score": None,
            "score_components": {"fallback": True},
            "source_caption_ids": fallback["source_caption_ids"],
            "source_spans": fallback["source_spans"],
        })
    return [x["text"] for x in selected[:5]], trace


def read_resume_trace(path, input_by_id):
    """Load compact resume state and truncate a damaged final JSONL record."""
    completed = {}
    candidate_total = 0
    type_counts, role_counts = Counter(), Counter()
    path = Path(path)
    if not path.exists():
        return completed, candidate_total, type_counts, role_counts

    last_good_offset = 0
    with path.open("r+", encoding="utf-8") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                last_good_offset = handle.tell()
                break
            if not line.strip():
                last_good_offset = handle.tell()
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Only an interrupted final write is recoverable. Corruption in
                # the middle of the journal must stop the run for safety.
                remainder = handle.read()
                if remainder.strip():
                    raise ValueError("Corrupt non-final line in resume trace {}".format(path))
                print("[resume] truncating incomplete final trace record")
                handle.seek(line_start)
                handle.truncate()
                last_good_offset = line_start
                break

            image_id = int(record["image_id"])
            if image_id not in input_by_id:
                raise ValueError("Trace contains image_id={} absent from current input".format(image_id))
            if record.get("raw_sha256") != raw_digest(input_by_id[image_id]["raw_captions"]):
                raise ValueError("Raw captions changed for image_id={}; refusing stale resume".format(image_id))
            captions = record.get("ecsd_captions", [])
            if len(captions) != 5 or not all(isinstance(x, str) and x.strip() for x in captions):
                raise ValueError("Invalid resumed ECSD record for image_id={}".format(image_id))
            if image_id in completed:
                raise ValueError("Duplicate image_id={} in resume trace".format(image_id))
            completed[image_id] = captions
            candidate_total += int(record.get("candidate_count", 0))
            for item in record.get("selected", []):
                type_counts[item.get("candidate_type", "unknown")] += 1
                role_counts[item.get("role", "unknown")] += 1
            last_good_offset = handle.tell()
    return completed, candidate_total, type_counts, role_counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSON records with image_id and raw_captions")
    p.add_argument("--output", required=True)
    p.add_argument("--trace", required=True, help="JSONL resume journal and selected-candidate trace")
    p.add_argument("--summary", required=True)
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--clip-cache", default="~/.cache/clip")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256, help="CLIP text batch size")
    p.add_argument("--image-batch-size", type=int, default=256, help="Images processed per resume chunk")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-images", type=int, default=0, help="Optional smoke-test limit; 0 means all")
    args = p.parse_args()

    input_path, output_path, trace_path = map(Path, (args.input, args.output, args.trace))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    for path in (output_path, trace_path):
        if path.exists() and args.overwrite:
            path.unlink()
        elif path.exists() and not args.resume:
            raise FileExistsError("{} exists; pass --resume or --overwrite".format(path))

    if args.batch_size <= 0 or args.image_batch_size <= 0:
        raise ValueError("--batch-size and --image-batch-size must be positive")
    rows = load_records(input_path)
    if args.max_images:
        rows = rows[:args.max_images]
    input_by_id = {int(row["image_id"]): row for row in rows}
    if args.resume:
        completed, candidate_total, type_counts, role_counts = read_resume_trace(trace_path, input_by_id)
    else:
        completed, candidate_total, type_counts, role_counts = {}, 0, Counter(), Counter()
    pending = [row for row in rows if int(row["image_id"]) not in completed]
    print("[ECSD] images={} completed={} pending={}".format(len(rows), len(completed), len(pending)))

    if pending:
        clip_module, model, device = load_clip_model(args.clip_model, args.device, args.clip_cache)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as trace_file:
            for start in range(0, len(pending), args.image_batch_size):
                chunk = pending[start:start + args.image_batch_size]
                candidates_by_image = []
                unique_texts = {}
                for row in chunk:
                    candidates = build_candidates(row["raw_captions"])
                    candidates_by_image.append((row, candidates))
                    for candidate in candidates:
                        unique_texts.setdefault(candidate["text"], None)
                texts = list(unique_texts)
                embeddings = encode_texts(
                    texts, clip_module, model, device, batch_size=args.batch_size
                )
                embedding_by_text = {text: embeddings[i] for i, text in enumerate(texts)}
                for row, candidates in candidates_by_image:
                    selected, selection_trace = select_five(
                        candidates, row["raw_captions"], embedding_by_text
                    )
                    if len(selected) != 5:
                        raise RuntimeError("ECSD did not return exactly five captions for image {}".format(row["image_id"]))
                    source_facts = set().union(*(content_tokens(x) for x in row["raw_captions"]))
                    unsupported = [x for x in selected if not content_tokens(x).issubset(source_facts)]
                    if unsupported:
                        raise RuntimeError("ECSD selected unsupported lexical facts for image {}: {}".format(row["image_id"], unsupported))
                    record = {
                        "image_id": int(row["image_id"]),
                        "raw_sha256": raw_digest(row["raw_captions"]),
                        "candidate_count": len(candidates),
                        "ecsd_captions": selected,
                        "selected": selection_trace,
                    }
                    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed[record["image_id"]] = selected
                    candidate_total += len(candidates)
                    for item in selection_trace:
                        type_counts[item.get("candidate_type", "unknown")] += 1
                        role_counts[item.get("role", "unknown")] += 1
                trace_file.flush()
                import os
                os.fsync(trace_file.fileno())
                print("[ECSD] completed {}/{} images".format(len(completed), len(rows)), flush=True)

    output_rows = []
    for row in rows:
        selected = completed[int(row["image_id"])]
        output_rows.append({
            "image_id": int(row["image_id"]),
            "raw_captions": row["raw_captions"],
            "ecsd_captions": selected,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, ensure_ascii=False), encoding="utf-8")
    summary = {
        "status": "ok",
        "algorithm": "ECSD-naive-v1 (reused from experiments/5k_ecsd)",
        "num_images": len(output_rows),
        "captions_per_image": 5,
        "candidate_count_total": candidate_total,
        "candidate_count_mean": candidate_total / max(1, len(output_rows)),
        "selected_candidate_types": dict(type_counts),
        "selected_roles": dict(role_counts),
        "clip_model": args.clip_model,
        "clip_device": device if pending else "reused-trace-no-encoding",
        "input": str(input_path),
        "output": str(output_path),
        "trace": str(trace_path),
        "resumable": True,
        "constraints": {
            "llm": False,
            "image_encoder_for_caption_generation": False,
            "evidence_source": "the five original COCO captions only",
            "selection_weights_unchanged": True,
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[DONE] {}".format(output_path))
    print("[DONE] {}".format(summary_path))


if __name__ == "__main__":
    main()
