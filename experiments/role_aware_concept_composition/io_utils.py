"""Input adapters for F5, CoCu, Bootstrap and Talk2DINO PTH files."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

def read_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not rows:
            raise ValueError(f"No JSON/JSONL records found: {path}")
        return rows

def _to_rows(value):
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("curated", "concepts", "candidates", "selected_concepts", "matched", "selected"):
            if isinstance(value.get(key), list):
                return _to_rows(value[key])
        return []
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            out.append({"concept": item})
        elif isinstance(item, dict):
            concept = item.get("concept", item.get("name", item.get("text")))
            if concept:
                out.append(dict(item, concept=str(concept).strip()))
    return out

def _record_rows(record):
    if not isinstance(record, dict):
        return []
    for key in ("curated", "concepts", "candidates", "selected_concepts", "matched", "selected"):
        if key in record:
            rows = _to_rows(record[key])
            if rows:
                return rows
    variants = record.get("variants")
    if isinstance(variants, dict):
        collected = []
        for name in sorted(variants):
            entry = variants[name]
            if isinstance(entry, dict):
                collected.extend(_to_rows(entry.get("selected")))
        if collected:
            return collected
    return []

def load_source(path, source_name):
    payload = read_json_or_jsonl(path)
    result = defaultdict(list)

    def add_record(iid, rec):
        rows = _record_rows(rec)
        if rows:
            result[int(iid)].extend(rows)

    if isinstance(payload, list):
        for rec in payload:
            if isinstance(rec, dict) and "image_id" in rec:
                add_record(rec["image_id"], rec)
    elif isinstance(payload, dict):
        images = payload.get("images")
        if isinstance(images, list):
            for rec in images:
                if isinstance(rec, dict) and "image_id" in rec:
                    add_record(rec["image_id"], rec)
                elif isinstance(rec, dict) and "id" in rec:
                    add_record(rec["id"], rec)
        elif isinstance(images, dict):
            for iid, rec in images.items():
                if str(iid).isdigit():
                    add_record(iid, rec)
        else:
            for iid, rec in payload.items():
                if str(iid).isdigit():
                    add_record(iid, rec)
            if not result and isinstance(payload.get("records"), list):
                for rec in payload["records"]:
                    if isinstance(rec, dict) and "image_id" in rec:
                        add_record(rec["image_id"], rec)

    cleaned = {}
    for iid, rows in result.items():
        seen, keep = set(), []
        for row in rows:
            key = " ".join(str(row.get("concept", "")).lower().split())
            if key and key not in seen:
                seen.add(key)
                keep.append(row)
        cleaned[iid] = keep
    if not cleaned:
        raise ValueError(
            f"Could not find image_id -> concept rows in {path}. Supported schemas: "
            "{images:[{image_id,curated:[...]}]}, "
            "{images:{image_id:{selected_concepts:[...]}}}, image-ID keyed JSON, "
            "or F5 audit JSONL with candidates."
        )
    return cleaned

def load_gallery(path=None):
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("concept", payload.get("concepts", [])) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("Gallery must be a list or a dict with 'concept'/'concepts' list")
    out, seen = [], set()
    for item in values:
        phrase = item.get("concept") if isinstance(item, dict) else item
        if isinstance(phrase, str):
            key = " ".join(phrase.lower().split())
            if key and key not in seen:
                seen.add(key)
                out.append(phrase.strip())
    return out

def load_base_train(path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Run inside the existing Talk2DINO PyTorch environment.") from exc
    try:
        base = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        base = torch.load(path, map_location="cpu")
    if not isinstance(base, dict) or "images" not in base or "annotations" not in base:
        raise ValueError("base_train must be Talk2DINO train.pth with images and annotations")
    image_ids = [int(x["id"]) for x in base["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs in base train.pth")
    grouped = defaultdict(list)
    for ann in base["annotations"]:
        grouped[int(ann["image_id"])].append(ann)
    missing = [iid for iid in image_ids if len(grouped[iid]) < 5]
    if missing:
        raise ValueError(f"{len(missing)} images have fewer than five captions; first IDs: {missing[:10]}")
    for iid in image_ids:
        for ann in grouped[iid][:5]:
            if "caption" not in ann or "ann_feats" not in ann:
                raise ValueError(f"Image {iid}: first five annotations require caption and ann_feats")
            if int(torch.as_tensor(ann["ann_feats"]).numel()) != 512:
                raise ValueError(f"Image {iid}: ann_feats must be 512-D CLIP text feature")
    return base, image_ids, grouped

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
