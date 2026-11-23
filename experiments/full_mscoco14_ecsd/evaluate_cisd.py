#!/usr/bin/env python3
"""Chunked per-image CISD and lexical source-fact coverage for Raw vs ECSD."""
import argparse
import csv
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ECSD_5K = ROOT / "experiments" / "5k_ecsd"
if str(ECSD_5K) not in sys.path:
    sys.path.insert(0, str(ECSD_5K))
from clip_utils import encode_texts, load_clip_model  # noqa: E402
from ecsd_core import content_tokens  # noqa: E402

PAIRS = list(combinations(range(5), 2))


def q(values, fraction):
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * fraction
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] * (1 - (pos - lo)) + values[hi] * (pos - lo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--ecsd", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clip-model", default="ViT-B/16")
    p.add_argument("--clip-cache", default="~/.cache/clip")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--image-batch-size", type=int, default=512)
    args = p.parse_args()

    raw_rows = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    ecsd_rows = json.loads(Path(args.ecsd).read_text(encoding="utf-8"))
    raw_map = {int(x["image_id"]): x["raw_captions"] for x in raw_rows}
    ecsd_map = {int(x["image_id"]): x["ecsd_captions"] for x in ecsd_rows}
    if set(raw_map) != set(ecsd_map):
        raise ValueError("Raw/ECSD image IDs differ")
    for image_id in raw_map:
        if len(raw_map[image_id]) != 5 or len(ecsd_map[image_id]) != 5:
            raise ValueError("image_id={} must have five captions per arm".format(image_id))

    clip_module, model, device = load_clip_model(args.clip_model, args.device, args.clip_cache)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_image_path = out / "per_image_cisd.csv"
    summary_values = {arm: {"cisd": [], "sim": [], "unique_facts": []} for arm in ("raw", "ecsd")}
    coverage_values = []
    fields = ["image_id", "raw_cisd", "ecsd_cisd", "raw_mean_similarity", "ecsd_mean_similarity",
              "raw_unique_facts", "ecsd_unique_facts", "ecsd_source_fact_coverage"]

    with per_image_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        ids = sorted(raw_map)
        for start in range(0, len(ids), args.image_batch_size):
            chunk_ids = ids[start:start + args.image_batch_size]
            texts = []
            for image_id in chunk_ids:
                texts.extend(raw_map[image_id])
                texts.extend(ecsd_map[image_id])
            unique_texts = list(dict.fromkeys(texts))
            embeddings = encode_texts(unique_texts, clip_module, model, device, batch_size=args.batch_size)
            emb_by_text = {text: embeddings[i] for i, text in enumerate(unique_texts)}
            batch_rows = []
            for image_id in chunk_ids:
                metrics = {}
                for arm, captions in (("raw", raw_map[image_id]), ("ecsd", ecsd_map[image_id])):
                    embs = [emb_by_text[x] for x in captions]
                    sims = [float((embs[i] * embs[j]).sum().item()) for i, j in PAIRS]
                    mean_sim = sum(sims) / len(sims)
                    cisd = 1.0 - mean_sim
                    facts = set().union(*(content_tokens(x) for x in captions))
                    metrics[arm] = (cisd, mean_sim, len(facts))
                    summary_values[arm]["cisd"].append(cisd)
                    summary_values[arm]["sim"].append(mean_sim)
                    summary_values[arm]["unique_facts"].append(len(facts))
                raw_facts = set().union(*(content_tokens(x) for x in raw_map[image_id]))
                ecsd_facts = set().union(*(content_tokens(x) for x in ecsd_map[image_id]))
                coverage = len(raw_facts & ecsd_facts) / max(1, len(raw_facts))
                coverage_values.append(coverage)
                batch_rows.append({
                    "image_id": image_id,
                    "raw_cisd": metrics["raw"][0],
                    "ecsd_cisd": metrics["ecsd"][0],
                    "raw_mean_similarity": metrics["raw"][1],
                    "ecsd_mean_similarity": metrics["ecsd"][1],
                    "raw_unique_facts": metrics["raw"][2],
                    "ecsd_unique_facts": metrics["ecsd"][2],
                    "ecsd_source_fact_coverage": coverage,
                })
            writer.writerows(batch_rows)
            print("[CISD] {}/{} images".format(min(start + len(chunk_ids), len(ids)), len(ids)), flush=True)

    summary_rows = []
    for arm in ("raw", "ecsd"):
        cisd = summary_values[arm]["cisd"]
        sims = summary_values[arm]["sim"]
        facts = summary_values[arm]["unique_facts"]
        summary_rows.append({
            "arm": arm,
            "num_images": len(cisd),
            "mean_cisd": statistics.mean(cisd),
            "std_cisd_population": statistics.pstdev(cisd),
            "median_cisd": statistics.median(cisd),
            "q25_cisd": q(cisd, 0.25),
            "q75_cisd": q(cisd, 0.75),
            "mean_pairwise_similarity": statistics.mean(sims),
            "mean_unique_content_facts": statistics.mean(facts),
        })
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = {
        "metric": "CISD = 1 - mean cosine similarity over ten caption pairs",
        "encoder": args.clip_model,
        "l2_normalized": True,
        "num_images": len(raw_map),
        "captions_per_image": 5,
        "mean_ecsd_source_fact_coverage": statistics.mean(coverage_values),
        "device": device,
        "per_image_csv": str(per_image_path),
        "summary_csv": str(out / "summary.csv"),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[DONE] CISD summary -> {}".format(out / "summary.csv"))
    print("[DONE] Mean ECSD source-fact coverage: {:.4f}".format(metadata["mean_ecsd_source_fact_coverage"]))


if __name__ == "__main__":
    main()
