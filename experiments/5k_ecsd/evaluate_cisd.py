#!/usr/bin/env python3
"""Compute per-image CLIP Intra-set Semantic Diversity for all caption arms."""
import argparse
import csv
import json
import statistics
from itertools import combinations
from pathlib import Path

import torch

from clip_utils import encode_texts, load_clip_model

ARMS = {
    "raw": "raw_captions",
    "rich": "rich_captions",
    "rich_weaken": "rich_weaken_captions",
    "rich_diversity": "rich_diversity_captions",
    "ecsd": "ecsd_captions",
}
PAIRS = list(combinations(range(5), 2))
PAIR_NAMES = ["s{}{}".format(i + 1, j + 1) for i, j in PAIRS]


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", default="data/coco2014_subset_50k/5k_ECSD.json")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd/cisd")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-cache", default="~/.cache/clip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expected-images", type=int, default=5000)
    args = parser.parse_args()

    rows = json.loads(Path(args.captions).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Caption JSON must be a list.")
    if args.expected_images and len(rows) != args.expected_images:
        raise ValueError(
            "Expected {} image records, found {}.".format(args.expected_images, len(rows))
        )

    image_ids = [int(row["image_id"]) for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image_id in caption JSON.")

    flat_texts = []
    for row in rows:
        for arm, field in ARMS.items():
            captions = row.get(field)
            if not isinstance(captions, list) or len(captions) != 5:
                raise ValueError(
                    "image_id={} field={} must contain exactly five captions.".format(
                        row["image_id"], field
                    )
                )
            if not all(isinstance(text, str) and text.strip() for text in captions):
                raise ValueError("Empty/non-string caption in image_id={} field={}".format(
                    row["image_id"], field
                ))
            flat_texts.extend(captions)

    unique_texts = list(dict.fromkeys(flat_texts))
    clip_module, model, device = load_clip_model(
        args.clip_model, args.device, args.clip_cache
    )
    embeddings = encode_texts(
        unique_texts, clip_module, model, device, batch_size=args.batch_size
    )
    embedding_by_text = {
        text: embeddings[index] for index, text in enumerate(unique_texts)
    }

    per_image_rows = []
    values_by_arm = {arm: [] for arm in ARMS}
    sims_by_arm = {arm: [] for arm in ARMS}

    for row_index, row in enumerate(rows):
        output_row = {"image_id": int(row["image_id"])}
        for arm, field in ARMS.items():
            embs = torch.stack([embedding_by_text[text] for text in row[field]], dim=0)
            sims = [float(torch.dot(embs[i], embs[j]).item()) for i, j in PAIRS]
            mean_similarity = sum(sims) / len(sims)
            cisd = 1.0 - mean_similarity

            output_row[arm + "_mean_pairwise_similarity"] = mean_similarity
            output_row[arm + "_cisd"] = cisd
            for pair_name, similarity in zip(PAIR_NAMES, sims):
                output_row[arm + "_" + pair_name] = similarity

            values_by_arm[arm].append(cisd)
            sims_by_arm[arm].extend(sims)

        per_image_rows.append(output_row)
        if (row_index + 1) % 500 == 0:
            print("[CISD] evaluated {}/{} images".format(
                row_index + 1, len(rows)
            ), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_path = output_dir / "per_image_cisd.csv"
    fields = ["image_id"]
    for arm in ARMS:
        fields.extend([arm + "_mean_pairwise_similarity", arm + "_cisd"])
        fields.extend(arm + "_" + pair_name for pair_name in PAIR_NAMES)
    with per_image_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_rows = []
    for arm in ARMS:
        values = values_by_arm[arm]
        similarities = sims_by_arm[arm]
        summary_rows.append({
            "arm": arm,
            "num_images": len(values),
            "mean_cisd": statistics.mean(values),
            "std_cisd_population": statistics.pstdev(values),
            "median_cisd": statistics.median(values),
            "q25_cisd": quantile(values, 0.25),
            "q75_cisd": quantile(values, 0.75),
            "mean_pairwise_similarity": statistics.mean(similarities),
            "min_cisd": min(values),
            "max_cisd": max(values),
        })

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "metric": "CISD",
        "definition": "1 - mean cosine similarity over the 10 unordered pairs",
        "encoder": args.clip_model,
        "l2_normalized": True,
        "num_images": len(rows),
        "captions_per_image": 5,
        "pair_count_per_image": 10,
        "arms": ARMS,
        "device": device,
        "per_image_csv": str(per_image_path),
        "summary_csv": str(summary_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("[DONE] Per-image CISD: {}".format(per_image_path))
    print("[DONE] CISD summary: {}".format(summary_path))
    for row in summary_rows:
        print(
            "{arm:14s} CISD={mean_cisd:.6f} ± {std_cisd_population:.6f} "
            "mean_sim={mean_pairwise_similarity:.6f}".format(**row)
        )


if __name__ == "__main__":
    main()
