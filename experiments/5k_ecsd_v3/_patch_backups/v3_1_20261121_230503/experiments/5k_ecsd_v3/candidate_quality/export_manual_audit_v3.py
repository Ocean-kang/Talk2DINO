#!/usr/bin/env python3
"""Export v3 manual audit; Selected is determined only by exact candidate_id membership."""
import argparse
import csv
import json
import random
from pathlib import Path
import sys

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))
from utils.safe_io import atomic_write_text

LABEL_FIELDS = [
    "validity_label", "groundedness_label", "completeness_label",
    "category_correct_label", "redundancy_label", "annotator_notes",
]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_audit_rows(source_rows, pool_rows, trace_rows, sample_size=100, seed=42):
    source_map = {int(x["image_id"]): x for x in source_rows}
    pool_map = {int(x["image_id"]): x for x in pool_rows}
    trace_map = {int(x["image_id"]): x for x in trace_rows}
    if (len(source_map) != len(source_rows) or len(pool_map) != len(pool_rows)
            or len(trace_map) != len(trace_rows)):
        raise ValueError("Duplicate image IDs in source, candidate pool, or B4 trace")
    if set(source_map) != set(pool_map) or set(source_map) != set(trace_map):
        raise ValueError("Source, candidate pool, and B4 trace image IDs must match exactly")
    ids = sorted(source_map)
    rng = random.Random(seed)
    sampled = sorted(rng.sample(ids, min(sample_size, len(ids))))
    output = []
    viewer = []
    for image_id in sampled:
        raw = source_map[image_id]["raw_captions"]
        if not isinstance(raw, list) or len(raw) != 5 or not all(str(x).strip() for x in raw):
            raise ValueError("image_id={} must have five non-empty Raw Captions".format(image_id))
        if trace_map[image_id].get("method") != "B4":
            raise ValueError("Expected B4 trace for image_id={}".format(image_id))
        if trace_map[image_id].get("anchor_candidate_id") != trace_map[image_id].get("selected", [{}])[0].get("candidate_id"):
            raise ValueError("Anchor field differs from first selected trace item for image_id={}".format(image_id))
        if pool_map[image_id].get("raw_captions") != raw:
            raise ValueError("Raw Caption mismatch for image_id={}".format(image_id))
        selected_items = trace_map[image_id].get("selected", [])
        if len(selected_items) != 5:
            raise ValueError("B4 trace must contain exactly five selected items for image_id={}".format(image_id))
        selected_ids = [x.get("candidate_id") for x in selected_items]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("B4 trace repeats candidate IDs for image_id={}".format(image_id))
        candidate_ids = [x.get("candidate_id") for x in pool_map[image_id]["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Duplicate candidate IDs in pool for image_id={}".format(image_id))
        candidate_map = {x["candidate_id"]: x for x in pool_map[image_id]["candidates"]}
        for item in selected_items:
            if item["candidate_id"] not in candidate_map:
                raise ValueError("Unknown selected candidate ID {} for image_id={}".format(item["candidate_id"], image_id))
            if item.get("text") != candidate_map[item["candidate_id"]].get("text"):
                raise ValueError("Trace text does not match candidate ID for image_id={}".format(image_id))
        selected_set = set(selected_ids)
        selected_texts = [x["text"] for x in selected_items]
        candidate_view = []
        for candidate in pool_map[image_id]["candidates"]:
            is_selected = candidate["candidate_id"] in selected_set
            row = {
                "image_id": image_id,
                **{"raw_caption_{}".format(i + 1): raw[i] for i in range(5)},
                "candidate_id": candidate["candidate_id"],
                "candidate_type": candidate["candidate_type"],
                "candidate_text": candidate["text"],
                "candidate_origins": json.dumps(candidate.get("candidate_origins", []), ensure_ascii=False),
                "source_caption_ids": json.dumps(candidate.get("source_caption_ids", [])),
                "source_spans": json.dumps(candidate.get("source_spans", []), ensure_ascii=False),
                "predicted_categories": "|".join(candidate.get("categories", [])),
                "classification_rules": json.dumps(candidate.get("category_rules", {}), ensure_ascii=False),
                "selection_status": "Selected" if is_selected else "Not selected",
                **{field: "" for field in LABEL_FIELDS},
            }
            output.append(row)
            candidate_view.append({
                "candidate_id": candidate["candidate_id"],
                "candidate_type": candidate["candidate_type"],
                "candidate_origins": candidate.get("candidate_origins", []),
                "text": candidate["text"],
                "source_spans": candidate.get("source_spans", []),
                "categories": candidate.get("categories", []),
                "rules": candidate.get("category_rules", {}),
                "selected": is_selected,
                "quality_flags": candidate.get("quality_flags", []),
            })
        if sum(x["selection_status"] == "Selected" for x in output if x["image_id"] == image_id) != 5:
            raise AssertionError("Selected-by-ID count must be exactly five for image_id={}".format(image_id))
        viewer.append({
            "image_id": image_id,
            "raw_captions": raw,
            "selected_captions": selected_texts,
            "selected_candidate_ids": selected_ids,
            "candidates": candidate_view,
        })
    return output, viewer, sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--candidate-pool", default="outputs/5k_ecsd_v3/candidates/ecsd_v3_candidates.jsonl")
    parser.add_argument("--trace", default="outputs/5k_ecsd_v3/selection_traces/B4_selection_trace.jsonl")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_v3/manual_audit")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_rows = read_json(args.source)
    pool_rows = read_jsonl(args.candidate_pool)
    trace_rows = read_jsonl(args.trace)
    rows, viewer_data, sampled = build_audit_rows(source_rows, pool_rows, trace_rows, args.sample_size, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manual_annotation_v3.csv"
    html_path = out_dir / "manual_audit_viewer_v3.html"
    guidelines_path = out_dir / "ANNOTATION_GUIDELINES_v3.md"
    if not args.overwrite:
        for path in (csv_path, html_path, guidelines_path):
            if path.exists():
                raise FileExistsError("{} exists; manual labels are protected. Use --overwrite only after backing up or migrating labels.".format(path))

    columns = [
        "image_id", "raw_caption_1", "raw_caption_2", "raw_caption_3", "raw_caption_4", "raw_caption_5",
        "candidate_id", "candidate_type", "candidate_origins", "candidate_text", "source_caption_ids",
        "source_spans", "predicted_categories", "classification_rules", "selection_status",
    ] + LABEL_FIELDS
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(csv_path, buffer.getvalue(), overwrite=args.overwrite)

    payload = json.dumps(viewer_data, ensure_ascii=False).replace("</", "<\\/")
    viewer = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECSD v3 Manual Audit</title>
<style>body{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#20242a}header{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #ddd;border-radius:10px;z-index:2}.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:16px 0}.raw{background:#f5f7fa;padding:8px;border-radius:6px;margin:6px 0}.candidate{border-top:1px solid #eee;padding:10px 4px}.selected{border-left:4px solid #21865b;padding-left:10px}.small{color:#626b75;font-size:12px}.tag{display:inline-block;background:#edf0f3;border-radius:4px;padding:2px 6px;margin-right:4px}input{padding:7px;width:100%;max-width:340px}</style></head><body>
<header><b>ECSD v3 Candidate Audit</b> · Selected is exact candidate_id match · annotate in CSV<div><input id="search" placeholder="Filter image or candidate"></div></header><div id="root"></div>
<script>const DATA=__DATA__;const root=document.getElementById('root');function el(t,c,s){const x=document.createElement(t);if(c)x.className=c;if(s!==undefined)x.textContent=s;return x;}function render(q=''){root.innerHTML='';DATA.forEach(im=>{if(q&&!JSON.stringify(im).toLowerCase().includes(q.toLowerCase()))return;const card=el('section','card');card.appendChild(el('h2','',`Image ${im.image_id}`));im.raw_captions.forEach((c,i)=>card.appendChild(el('div','raw',`Raw ${i+1}: ${c}`)));card.appendChild(el('h3','','Selected five'));im.selected_captions.forEach((c,i)=>card.appendChild(el('div','raw',`${i+1}. ${c}`)));card.appendChild(el('h3','',`Candidate pool (${im.candidates.length})`));im.candidates.forEach(c=>{const box=el('div',c.selected?'candidate selected':'candidate');box.appendChild(el('b','',`${c.candidate_type} · ${c.candidate_id}`));box.appendChild(el('div','',c.text));box.appendChild(el('div','small',`Selected: ${c.selected} | Roles: ${c.categories.join(', ')||'Unknown'} | Flags: ${c.quality_flags.join(', ')||'none'}`));box.appendChild(el('div','small',`Source: ${JSON.stringify(c.source_spans)}`));card.appendChild(box);});root.appendChild(card);});}document.getElementById('search').addEventListener('input',e=>render(e.target.value));render();</script></body></html>'''.replace("__DATA__", payload)
    atomic_write_text(html_path, viewer, overwrite=args.overwrite)
    guidelines = """# ECSD v3 Manual Annotation Guidelines\n\n- `validity_label`: Valid / Partially Valid / Invalid.\n- `groundedness_label`: Grounded / Unsupported, judged only against the five Raw Captions.\n- `completeness_label`: Complete / Incomplete.\n- `category_correct_label`: Correct / Incorrect. Multiple roles may be appropriate.\n- `redundancy_label`: Redundant / Complementary relative to the other four selected captions.\n- `annotator_notes`: explain ambiguous or erroneous cases.\n\nThe `selection_status` column is generated by exact `candidate_id` membership in the B4 trace. Do not edit it manually. Automatic lexical labels are not ground truth.\n"""
    atomic_write_text(guidelines_path, guidelines, overwrite=args.overwrite)
    print("[DONE] images={} candidate_rows={} selected_rows={}".format(len(sampled), len(rows), sum(x["selection_status"] == "Selected" for x in rows)))
    print("[DONE] CSV:", csv_path)
    print("[DONE] HTML:", html_path)


if __name__ == "__main__":
    main()
