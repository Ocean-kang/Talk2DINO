#!/usr/bin/env python3
"""Generate B1-B4 caption sets using one shared candidate pool and CLIP ViT-B/16."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEXT_ROOT = HERE.parent
REPO_ROOT = NEXT_ROOT.parents[1]
sys.path.insert(0, str(NEXT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "5k_ecsd"))

from config_utils import load_config
from selection_methods import METHODS, select_five
from clip_utils import load_clip_model, encode_texts

OUTPUTS = {
    "B1": ("5k_ECSD_B1_Coverage.json", "ecsd_b1_captions"),
    "B2": ("5k_ECSD_B2_Diversity.json", "ecsd_b2_captions"),
    "B3": ("5k_ECSD_B3_Coverage_Diversity.json", "ecsd_b3_captions"),
    "B4": ("5k_ECSD_B4_Coverage_Diversity_Role.json", "ecsd_b4_captions"),
}

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def load_base(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input caption JSON must be a list.")
    ids = [int(x["image_id"]) for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs in source captions.")
    return rows


def write_json(path, rows, overwrite=False):
    """Safely write or reuse an existing ablation caption JSON."""
    path = Path(path)
    overwrite = overwrite or os.environ.get("REBUILD_ABLATION") == "1"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                "Existing output is invalid JSON: {}".format(path)
            ) from exc

        if existing == rows:
            print("[REUSE] Existing output matches generated captions: {}".format(path))
            return

        raise FileExistsError(
            "{} exists but differs from generated captions. "
            "Original file preserved. Use --overwrite only if intentional.".format(path)
        )

    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
            f.write("\\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print("[DONE] Saved captions:", path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--config", default="experiments/5k_ecsd_next/configs/default.json")
    parser.add_argument("--candidate-pool", default="outputs/5k_ecsd_next/candidates/ecsd_v2_candidates.jsonl")
    parser.add_argument("--output-dir", default="data/coco2014_subset_50k")
    parser.add_argument("--trace-dir", default="outputs/5k_ecsd_next/selection_traces")
    parser.add_argument("--methods", nargs="+", default=["B1", "B2", "B3", "B4"], choices=["B1", "B2", "B3", "B4"])
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-cache", default="~/.cache/clip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--write-v2", action="store_true")
    parser.add_argument("--v2-only", action="store_true",
                        help="When used with B4, write only 5k_ECSD_v2.json, not the B4 ablation file.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    selection_config = config.get("selection", {})
    filter_config = config.get("candidate_filter", {})
    base_rows = load_base(args.input)
    pool_rows = read_jsonl(args.candidate_pool)
    base_map = {int(x["image_id"]): x for x in base_rows}
    pool_map = {int(x["image_id"]): x for x in pool_rows}
    if set(base_map) != set(pool_map):
        raise ValueError("Candidate pool image IDs do not exactly match source caption IDs.")
    for image_id in base_map:
        if base_map[image_id].get("raw_captions") != pool_map[image_id].get("raw_captions"):
            raise ValueError("Candidate pool Raw Captions differ from source for image_id={}.".format(image_id))

    unique_texts = list(dict.fromkeys(
        candidate["text"]
        for row in pool_rows for candidate in row["candidates"]
        if str(candidate.get("text", "")).strip()
    ))
    print("[CLIP] Encoding {} unique candidate texts.".format(len(unique_texts)), flush=True)
    clip_module, model, device = load_clip_model(
        args.clip_model, args.device, args.clip_cache
    )
    vectors = encode_texts(
        unique_texts, clip_module, model, device, batch_size=args.batch_size
    )
    embedding_by_text = {
        text: vectors[i] for i, text in enumerate(unique_texts)
    }

    output_dir = Path(args.output_dir)
    trace_dir = Path(args.trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        selected_rows = []
        trace_path = trace_dir / (method + "_selection_trace.jsonl")
        with trace_path.open("w", encoding="utf-8") as trace_handle:
            for index, pool_row in enumerate(pool_rows):
                image_id = int(pool_row["image_id"])
                selected, trace = select_five(
                    pool_row["candidates"],
                    pool_row["raw_captions"],
                    embedding_by_text,
                    method,
                    score_weights=selection_config,
                    filter_config=filter_config,
                )
                if len(selected) != 5:
                    raise RuntimeError("image_id={} method={} selected {} captions.".format(
                        image_id, method, len(selected)))
                row = dict(base_map[image_id])
                row[OUTPUTS[method][1]] = selected
                selected_rows.append(row)
                trace_handle.write(json.dumps({
                    "image_id": image_id,
                    "method": method,
                    "anchor_candidate_id": trace[0]["candidate_id"],
                    "selected": trace,
                }, ensure_ascii=False) + "\n")
                if (index + 1) % 500 == 0:
                    print("[{}] {}/{}".format(method, index + 1, len(pool_rows)), flush=True)

        filename, _ = OUTPUTS[method]
        if not (method == "B4" and args.v2_only):
            write_json(output_dir / filename, selected_rows, overwrite=args.overwrite)
            print("[DONE] {} -> {}".format(method, output_dir / filename))

        if method == "B4" and args.write_v2:
            v2_rows = []
            for row in selected_rows:
                copy = dict(row)
                copy["ecsd_v2_captions"] = copy[OUTPUTS["B4"][1]]
                v2_rows.append(copy)
            v2_path = output_dir / "5k_ECSD_v2.json"
            if v2_path.exists() and not args.overwrite:
                existing = load_base(v2_path)
                existing_map = {int(x["image_id"]): x.get("ecsd_v2_captions") for x in existing}
                new_map = {int(x["image_id"]): x["ecsd_v2_captions"] for x in v2_rows}
                if existing_map != new_map:
                    raise FileExistsError(
                        "{} exists but differs from current B4 output. Use --overwrite intentionally.".format(v2_path)
                    )
                print("[REUSE] Existing ECSD v2 matches B4 exactly:", v2_path)
            else:
                write_json(v2_path, v2_rows, overwrite=args.overwrite)
                print("[DONE] ECSD v2 ->", v2_path)

    metadata = {
        "methods": args.methods,
        "method_definitions": METHODS,
        "anchor_rule": "same maximum weighted-fact-mass original candidate for every method",
        "candidate_pool": str(args.candidate_pool),
        "clip_model": args.clip_model,
        "device": device,
        "evidence_source": "Raw Captions only",
        "captions_per_image": 5,
        "score_weights": selection_config,
        "candidate_filter": filter_config,
    }
    meta_path = trace_dir / "generation_metadata.json"
    # Metadata is a run manifest, not caption data; refresh it after expanding B4-only Stage A to B1-B4.
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
