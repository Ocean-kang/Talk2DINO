#!/usr/bin/env python3
"""Create completion-compatible manifest from freshly extracted CLIP/Local CoCu.

Preserves original 5+3 selection: global new_curated first, then curated;
local concept_head_scores sorted by score. NO hallucinated or padded concepts.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def ranked(items):
    result = []
    for index, obj in enumerate(items or []):
        if isinstance(obj, str):
            obj = {"concept": obj}
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("concept") or "").strip()
        if not name:
            continue
        score = obj.get("score")
        result.append((index, {"concept": name, "score": score,
                               "best_head": obj.get("best_head")}))
    # Original build_5plus3 ranking uses scores when available.
    if any(entry["score"] is not None for _, entry in result):
        result.sort(key=lambda r: (float("-inf") if r[1]["score"] is None
                                   else float(r[1]["score"]), -r[0]), reverse=True)
    seen = set()
    out = []
    for _, entry in result:
        k = entry["concept"].casefold()
        if k not in seen:
            seen.add(k)
            out.append(entry)
    return out


def select(source, row, limit):
    if source == 'clip':
        preferred = ranked(row.get('new_curated_concepts', []))
        backup = ranked(row.get('curated_concepts', []))
    else:
        preferred = ranked(row.get('concept_head_scores') or row.get('curated_concepts', []))
        backup = []
    seen = set()
    selected = []
    for item in preferred + backup:
        key = item['concept'].casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) == limit:
            break
    return selected


def build(source, source_json, base_train, output, limit=3):
    import torch
    payload = json.loads(Path(source_json).read_text(encoding='utf-8'))
    if source == 'clip':
        assert isinstance(payload, dict) and isinstance(payload.get('images'), list)
        src = {int(r['image_id']): r for r in payload['images']}
    else:
        assert isinstance(payload, dict)
        src = {int(r['image_id']): r for r in payload.values()
               if isinstance(r, dict) and 'image_id' in r}
    base = torch.load(base_train, map_location='cpu')
    grouped = defaultdict(list)
    for ann in base['annotations']:
        grouped[int(ann['image_id'])].append(ann)
    image_ids = [int(img['id']) for img in base['images']]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError('Duplicate image IDs in base PTH')
    records, insufficient = [], []
    for image_id in image_ids:
        original = grouped[image_id][:5]
        if len(original) != 5:
            raise RuntimeError(f'Image {image_id}: fewer than 5 raw captions')
        if image_id not in src:
            raise RuntimeError(f'Image {image_id}: absent from freshly extracted CoCu')
        selected = select(source, src[image_id], limit)
        if len(selected) != limit:
            insufficient.append((image_id, len(selected)))
        records.append({
            'image_id': image_id,
            'original_captions': [str(x['caption']) for x in original],
            'curated': [dict(concept=x['concept'], score=x['score'],
                             best_head=x['best_head'], rank=j+1)
                        for j, x in enumerate(selected)],
        })
    if insufficient:
        raise RuntimeError(
            f'{len(insufficient)} images have fewer than {limit} REAL curated concepts. '
            f'Examples: {insufficient[:15]}. Original 5+3 requires 3; '
            'check candidate extraction/gallery and do not fabricate concepts.'
        )
    dst = Path(output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({
        'meta': {'source': source, 'extraction_file': str(source_json),
                 'base_train': str(base_train), 'num_curated': limit,
                 'selection_policy': 'identical to bootstrap_5plus3'},
        'images': records
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] concept manifest: {dst} ({len(records)} images x {limit} concepts)')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',choices=['clip','local'],required=True)
    p.add_argument('--source-json',required=True)
    p.add_argument('--base-train',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--num-curated',type=int,default=3)
    a=p.parse_args()
    build(a.source,a.source_json,a.base_train,a.output,a.num_curated)

if __name__=='__main__':
    main()
