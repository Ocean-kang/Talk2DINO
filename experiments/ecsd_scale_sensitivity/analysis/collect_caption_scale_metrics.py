#!/usr/bin/env python3
"""Aggregate full-train per-image CISD and ECSD trace diagnostics over nested scales."""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ECSD_CORE = ROOT / "experiments" / "5k_ecsd"
if str(ECSD_CORE) not in sys.path:
    sys.path.insert(0, str(ECSD_CORE))
from ecsd_core import content_tokens


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_per_image(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result[int(row["image_id"])] = row
    return result


def mean(values):
    return statistics.mean(values) if values else ""


def std(values):
    return statistics.pstdev(values) if values else ""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--ecsd", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--per-image-cisd", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--long-tail-max-image-frequency", type=int, default=5)
    a = p.parse_args()

    raw_rows = read_json(a.raw)
    ecsd_rows = read_json(a.ecsd)
    raw_map = {int(x["image_id"]): x["raw_captions"] for x in raw_rows}
    ecsd_map = {int(x["image_id"]): x["ecsd_captions"] for x in ecsd_rows}
    if set(raw_map) != set(ecsd_map):
        raise ValueError("Raw and ECSD image IDs differ")

    cisd_map = read_per_image(a.per_image_cisd)
    trace_map = {}
    with Path(a.trace).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                trace_map[int(row["image_id"])] = row

    manifest = read_json(a.manifest)
    rows = []
    for scale, info in manifest["sizes"].items():
        ids = [int(x) for x in info["image_ids"]]
        raw_doc_freq, ecsd_doc_freq = Counter(), Counter()
        raw_cisd, ecsd_cisd = [], []
        raw_sim, ecsd_sim, raw_facts_n, ecsd_facts_n, source_coverage = [], [], [], [], []
        candidate_counts, valid_candidate_counts = [], []
        candidate_role_coverage, selected_role_coverage, fallback_slots = [], [], []
        selected_type_counts = Counter()
        candidate_type_counts = Counter()

        for image_id in ids:
            raw_captions = raw_map[image_id]
            ecsd_captions = ecsd_map[image_id]
            raw_facts = set().union(*(content_tokens(x) for x in raw_captions))
            ecsd_facts = set().union(*(content_tokens(x) for x in ecsd_captions))
            raw_doc_freq.update(raw_facts)
            ecsd_doc_freq.update(ecsd_facts)

            if image_id in cisd_map:
                metric = cisd_map[image_id]
                raw_cisd.append(float(metric["raw_cisd"]))
                ecsd_cisd.append(float(metric["ecsd_cisd"]))
                raw_sim.append(float(metric["raw_mean_similarity"]))
                ecsd_sim.append(float(metric["ecsd_mean_similarity"]))
                raw_facts_n.append(float(metric["raw_unique_facts"]))
                ecsd_facts_n.append(float(metric["ecsd_unique_facts"]))
                source_coverage.append(float(metric["ecsd_source_fact_coverage"]))

            trace = trace_map.get(image_id)
            if trace is None:
                raise ValueError("Missing B4 trace for image_id={}".format(image_id))
            candidate_counts.append(int(trace.get("candidate_count", 0)))
            valid_candidate_counts.append(int(trace.get("valid_candidate_count", 0)))
            candidate_role_coverage.append(float(trace.get("candidate_role_coverage", 0.0)))
            selected_role_coverage.append(float(trace.get("selected_role_coverage", 0.0)))
            fallback_slots.append(int(trace.get("fallback_slot_count", 0)))
            candidate_type_counts.update(trace.get("candidate_type_counts", {}))
            for item in trace.get("selected", []):
                selected_type_counts[item.get("candidate_type", "unknown")] += 1

        raw_vocab = set(raw_doc_freq)
        ecsd_vocab = set(ecsd_doc_freq)
        raw_long_tail = {
            token for token, count in raw_doc_freq.items()
            if count <= a.long_tail_max_image_frequency
        }
        retained_long_tail = raw_long_tail & ecsd_vocab
        row = {
            "scale": scale,
            "num_images": len(ids),
            "raw_mean_cisd": mean(raw_cisd),
            "raw_std_cisd": std(raw_cisd),
            "ecsd_mean_cisd": mean(ecsd_cisd),
            "ecsd_std_cisd": std(ecsd_cisd),
            "raw_mean_pairwise_similarity": mean(raw_sim),
            "ecsd_mean_pairwise_similarity": mean(ecsd_sim),
            "raw_mean_unique_facts_per_image": mean(raw_facts_n),
            "ecsd_mean_unique_facts_per_image": mean(ecsd_facts_n),
            "ecsd_source_fact_coverage": mean(source_coverage),
            "raw_corpus_concept_count": len(raw_vocab),
            "ecsd_corpus_concept_count": len(ecsd_vocab),
            "raw_concept_retention": len(raw_vocab & ecsd_vocab) / max(1, len(raw_vocab)),
            "raw_long_tail_concept_count": len(raw_long_tail),
            "ecsd_long_tail_retention": len(retained_long_tail) / max(1, len(raw_long_tail)),
            "mean_candidate_count": mean(candidate_counts),
            "mean_valid_candidate_count": mean(valid_candidate_counts),
            "mean_candidate_role_coverage": mean(candidate_role_coverage),
            "mean_selected_role_coverage": mean(selected_role_coverage),
            "fallback_slots_per_image": mean(fallback_slots),
            "fallback_slot_rate": sum(fallback_slots) / max(1, len(ids) * 5),
            "candidate_original_count": candidate_type_counts.get("original", 0),
            "candidate_clause_count": candidate_type_counts.get("clause", 0),
            "candidate_noun_phrase_count": candidate_type_counts.get("noun_phrase", 0),
            "selected_original_count": selected_type_counts.get("original", 0),
            "selected_clause_count": selected_type_counts.get("clause", 0),
            "selected_noun_phrase_count": selected_type_counts.get("noun_phrase", 0),
            "long_tail_threshold_image_frequency": a.long_tail_max_image_frequency,
        }
        rows.append(row)
        print("[caption-metrics] {}: {} images".format(scale, len(ids)))

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("[DONE] {}".format(out))


if __name__ == "__main__":
    main()
