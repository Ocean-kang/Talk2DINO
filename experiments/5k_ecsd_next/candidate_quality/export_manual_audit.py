#!/usr/bin/env python3
"""Export a deterministic 100-image annotation CSV and self-contained HTML viewer."""
import argparse
import csv
import json
import random
from pathlib import Path

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/coco2014_subset_50k/5k_rich_diversity.json")
    parser.add_argument("--candidate-pool", default="outputs/5k_ecsd_next/candidates/ecsd_v2_candidates.jsonl")
    parser.add_argument("--selected-json", default="data/coco2014_subset_50k/5k_ECSD_v2.json")
    parser.add_argument("--selected-field", default="ecsd_v2_captions")
    parser.add_argument("--output-dir", default="outputs/5k_ecsd_next/manual_audit")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_rows = json.loads(Path(args.source).read_text(encoding="utf-8"))
    selected_rows = json.loads(Path(args.selected_json).read_text(encoding="utf-8"))
    selected_map = {int(x["image_id"]): x[args.selected_field] for x in selected_rows}
    pool = read_jsonl(args.candidate_pool)
    pool_map = {int(x["image_id"]): x for x in pool}
    source_map = {int(x["image_id"]): x for x in source_rows}
    ids = sorted(set(pool_map) & set(selected_map) & set(source_map))
    if not ids:
        raise ValueError("No common image IDs between source, candidate pool, and selected JSON.")
    rng = random.Random(args.seed)
    sample_ids = sorted(rng.sample(ids, min(args.sample_size, len(ids))))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "manual_annotation.csv"
    viewer_path = out / "manual_audit_viewer.html"
    if (csv_path.exists() or viewer_path.exists()) and not args.overwrite:
        raise FileExistsError(
            "Manual audit files already exist. They may contain human labels; "
            "pass --overwrite only if you intentionally want to replace them."
        )
    columns = [
        "image_id", "raw_caption_1", "raw_caption_2", "raw_caption_3",
        "raw_caption_4", "raw_caption_5", "candidate_id", "candidate_type",
        "candidate_text", "source_caption_ids", "source_spans",
        "predicted_categories", "classification_rules", "selection_status",
        "validity_label", "groundedness_label", "completeness_label",
        "category_correct_label", "redundancy_label", "annotator_notes",
    ]
    viewer_data = []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for image_id in sample_ids:
            raw = source_map[image_id]["raw_captions"]
            selected = set(selected_map[image_id])
            candidates = pool_map[image_id]["candidates"]
            image_view = {
                "image_id": image_id,
                "raw_captions": raw,
                "selected_captions": selected_map[image_id],
                "candidates": [],
            }
            for candidate in candidates:
                spans = candidate.get("source_spans", [])
                selected_flag = candidate.get("text") in selected
                writer.writerow({
                    "image_id": image_id,
                    **{"raw_caption_{}".format(i + 1): raw[i] for i in range(5)},
                    "candidate_id": candidate.get("candidate_id", ""),
                    "candidate_type": candidate.get("candidate_type", ""),
                    "candidate_text": candidate.get("text", ""),
                    "source_caption_ids": json.dumps(candidate.get("source_caption_ids", [])),
                    "source_spans": json.dumps(spans, ensure_ascii=False),
                    "predicted_categories": "|".join(candidate.get("categories", [])),
                    "classification_rules": json.dumps(candidate.get("category_rules", {}), ensure_ascii=False),
                    "selection_status": "Selected" if selected_flag else "Not selected",
                    "validity_label": "",
                    "groundedness_label": "",
                    "completeness_label": "",
                    "category_correct_label": "",
                    "redundancy_label": "",
                    "annotator_notes": "",
                })
                image_view["candidates"].append({
                    "candidate_id": candidate.get("candidate_id", ""),
                    "candidate_type": candidate.get("candidate_type", ""),
                    "text": candidate.get("text", ""),
                    "spans": spans,
                    "categories": candidate.get("categories", []),
                    "rules": candidate.get("category_rules", {}),
                    "selected": selected_flag,
                    "flags": candidate.get("quality_flags", []),
                })
            viewer_data.append(image_view)

    payload = json.dumps(viewer_data, ensure_ascii=False).replace("</", "<\\/")
    viewer = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECSD Candidate Manual Audit</title>
<style>
body{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#20242a}
header{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #ddd;border-radius:10px;z-index:2}
.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:16px 0}
.raw{background:#f5f7fa;padding:8px;border-radius:6px;margin:6px 0}
.candidate{border-top:1px solid #eee;padding:10px 4px}
.selected{border-left:4px solid #21865b;padding-left:10px}
.small{color:#626b75;font-size:12px}.tag{display:inline-block;background:#edf0f3;border-radius:4px;padding:2px 6px;margin-right:4px}
input{padding:7px;width:100%;max-width:340px}
</style></head><body>
<header><b>ECSD Candidate Manual Audit</b> · deterministic sample · use the CSV for annotation labels
<div><input id="search" placeholder="Filter by image ID or candidate text"></div></header>
<div id="root"></div>
<script>
const DATA = __DATA__;
const root=document.getElementById('root');
function el(tag,cls,txt){const x=document.createElement(tag);if(cls)x.className=cls;if(txt!==undefined)x.textContent=txt;return x;}
function render(filter=''){
 root.innerHTML='';
 DATA.forEach(im=>{
  const all=JSON.stringify(im).toLowerCase();
  if(filter && !all.includes(filter.toLowerCase()))return;
  const card=el('section','card');card.appendChild(el('h2','',`Image ${im.image_id}`));
  im.raw_captions.forEach((c,i)=>card.appendChild(el('div','raw',`Raw ${i+1}: ${c}`)));
  card.appendChild(el('h3','','Selected five'));
  im.selected_captions.forEach((c,i)=>card.appendChild(el('div','raw',`${i+1}. ${c}`)));
  card.appendChild(el('h3','',`Candidate pool (${im.candidates.length})`));
  im.candidates.forEach(c=>{
   const box=el('div',c.selected?'candidate selected':'candidate');
   box.appendChild(el('b','',`${c.candidate_type} · ${c.candidate_id}`));
   box.appendChild(el('div','',c.text));
   box.appendChild(el('div','small',`Selected: ${c.selected} | Categories: ${c.categories.join(', ')||'Unknown'} | Flags: ${c.flags.join(', ')||'none'}`));
   box.appendChild(el('div','small',`Source spans: ${JSON.stringify(c.spans)}`));
   box.appendChild(el('div','small',`Rules: ${JSON.stringify(c.rules)}`));
   card.appendChild(box);
  });
  root.appendChild(card);
 });
}
document.getElementById('search').addEventListener('input',e=>render(e.target.value));
render();
</script></body></html>""".replace("__DATA__", payload)
    viewer_path.write_text(viewer, encoding="utf-8")
    guidelines = """# Manual Annotation Guidelines

Annotate each candidate independently, then judge its relation to the other selected/candidate descriptions.

- `validity_label`: `Valid`, `Partially Valid`, or `Invalid`. Judge grammatical completeness and whether the phrase can stand as a useful description.
- `groundedness_label`: `Grounded` or `Unsupported`. Only mark Unsupported when the candidate asserts content not present in the source Raw Captions. This is caption-evidence grounding, not image verification.
- `completeness_label`: `Complete` or `Incomplete`. Mark incomplete for dangling phrases, missing heads, or fragments that do not express a recoverable semantic unit.
- `category_correct_label`: `Correct` or `Incorrect`. Compare the predicted lexical category labels with the candidate's actual semantic role in context. Multiple labels may be correct.
- `redundancy_label`: `Redundant` or `Complementary`. Compare against the other selected captions for the same image.
- `annotator_notes`: record ambiguity or the specific error.

Do not treat the automatic lexical classifier as ground truth. The CSV contains blank annotation columns intentionally. Fill labels manually; no human labels are fabricated by the scripts.
"""
    (out / "ANNOTATION_GUIDELINES.md").write_text(guidelines, encoding="utf-8")
    print("[DONE] CSV:", csv_path)
    print("[DONE] HTML:", viewer_path)
    print("[DONE] Guidelines:", out / "ANNOTATION_GUIDELINES.md")

if __name__ == "__main__":
    main()
