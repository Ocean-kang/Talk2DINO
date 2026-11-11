"""Build all selected F5 datasets from the same COCO training population."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.local_cocu_talk2dino.common.algorithms import top_concepts
from experiments.local_cocu_talk2dino.common.candidates import from_neighbors
from experiments.local_cocu_talk2dino.common.data import gallery_words, novel, already_mentioned
from experiments.local_cocu_talk2dino.common.teacher import Teacher
from experiments.local_cocu_talk2dino_new_train.build_train import append_concepts, embed

from policies import VARIANTS, assign, select


def mine(iid, heads, captions, local_ids, cls_ids, gallery, teacher, args):
    raw = captions[iid]
    words = novel(from_neighbors(local_ids, captions, gallery, cap=300), raw)
    h = heads[iid].to(teacher.device)
    rows = top_concepts(words, h, teacher.encode(words), raw, k=30, per_head_k=50) if words else []
    peers = list(dict.fromkeys(int(n) for n in cls_ids[:30] if int(n) != iid and int(n) in heads))
    if peers and rows:
        negative_heads = torch.stack([heads[n] for n in peers])
        vectors = teacher.encode([r["concept"] for r in rows])
        scores = teacher.sim(negative_heads.flatten(0, 1), vectors)
        scores = scores.reshape(len(peers), 12, len(rows)).amax(dim=1).cpu().T.tolist()
    else:
        scores = [[] for _ in rows]
    for row, evidence in zip(rows, scores):
        target = row.pop("score")
        pairs = list(zip(peers, evidence))
        lower = sorted(pairs, key=lambda pair: pair[1])[:max(1, len(pairs) // 2)]
        filtered = [(n, s) for n, s in pairs if s <= target - args.negative_margin
                    and not already_mentioned(row["concept"], captions[n])]
        row.update(target_score=target, lower_half=lower, filtered=filtered)
    return h, rows


@torch.no_grad()
def build(args, work, variants):
    base = torch.load(args.base_train, map_location="cpu", weights_only=False)
    heads = {int(k): torch.as_tensor(v).float() for k, v in
             torch.load(args.heads, map_location="cpu", weights_only=False).items()}
    local, cls = [json.loads(Path(p).read_text(encoding="utf-8"))
                  for p in (args.local_neighbors, args.cls_neighbors)]
    gallery = gallery_words(args.gallery)
    images = base["images"]
    ids = [int(im["id"]) for im in images]
    records = defaultdict(list)
    for ann in base["annotations"]:
        records[int(ann["image_id"])].append(ann)
    if not ids or len(ids) != len(set(ids)) or set(records) != set(ids):
        raise ValueError("Training image/annotation ID coverage mismatch")
    if args.limit:
        ids = ids[:args.limit]
        images = images[:args.limit]
    captions = {i: [a["caption"] for a in group[:5]] for i, group in records.items()}
    # Require train-only retrieval caches. Unknown peers usually indicate a split mismatch.
    for iid in ids:
        if len(records[iid]) < 5:
            raise ValueError(f"Image {iid}: expected at least five captions, got {len(records[iid])}")
        records[iid] = records[iid][:5]  # Match the original F5 builder.
        if iid not in heads:
            raise ValueError(f"Image {iid}: missing from Head cache {args.heads}")
        if heads[iid].shape != (12, 768):
            raise ValueError(f"Image {iid}: expected Head shape (12, 768), got {tuple(heads[iid].shape)}")
        for cache in (local, cls):
            if str(iid) not in cache or any(int(n) not in captions for n in cache[str(iid)][:50]):
                raise ValueError(f"Missing train-only neighbor data for image {iid}")
    if not any(int(n) != i and int(n) in heads for i in ids for n in cls[str(i)][:30]):
        raise ValueError("CLS cache has no usable peer heads for this build")
    for im in images:
        if torch.as_tensor(im["disentangled_self_attn"]).shape != (12, 768):
            raise ValueError(f"Unexpected student feature shape for {im['id']}")
    teacher = Teacher(args.official, args.config, args.device, batch=args.batch)
    outputs = {v: [] for v in variants}
    edits = {v: [] for v in variants}
    counts = {v: Counter() for v in variants}
    clip_cache = {}
    with (work / "audit.jsonl").open("w", encoding="utf-8") as audit:
        for pos, iid in enumerate(ids):
            raw = captions[iid]
            local_ids = [int(n) for n in local[str(iid)][:50] if int(n) != iid]
            h, rows = mine(iid, heads, captions, local_ids, cls[str(iid)], gallery, teacher, args)
            raw_vec = torch.stack([torch.as_tensor(a["ann_feats"]).float().reshape(512)
                                   for a in records[iid]])
            projected = F.normalize(teacher.projector.project_clip_txt(raw_vec.to(teacher.device)).float(), dim=-1)
            head_caption = teacher.sim(h, projected).cpu()
            coverage = head_caption.amax(dim=1).tolist()
            new_words = sorted({r["concept"] for r in rows} - clip_cache.keys())
            if new_words:
                tokens = teacher.clip.tokenize([f"a photo of a {w}" for w in new_words], truncate=True)
                encoded = teacher.text.encode_text(tokens.to(teacher.device)).float().cpu()
                clip_cache.update(zip(new_words, F.normalize(encoded, dim=-1)))
            raw_unit = F.normalize(raw_vec, dim=-1)
            decisions = {}
            for v in variants:
                policy = VARIANTS[v]
                chosen = select(rows, coverage, policy, args.floor, args.margin, 3)
                mode = policy[1]
                costs = [(raw_unit @ clip_cache[r["concept"]]).tolist() if mode == "clip"
                         else head_caption[r["best_head"]].tolist() for r in chosen]
                slots = assign(costs, mode == "balanced")
                groups = [[] for _ in raw]
                decisions[v] = {"selected": [], "captions": []}
                for r, cost, slot in zip(chosen, costs, slots):
                    groups[slot].append(r["concept"])
                    decisions[v]["selected"].append(
                        {k: r[k] for k in ("concept", "score", "gate", "negative_ids", "negative_mean")}
                        | {"slot": slot, "costs": cost})
                counts[v][f"selected_{len(chosen)}"] += 1
                counts[v]["assignment_collisions"] += len(slots) - len(set(slots))
                for ann, concepts in zip(records[iid], groups):
                    updated = append_concepts(ann["caption"], concepts)
                    if concepts:
                        try:
                            teacher.clip.tokenize([updated])
                        except RuntimeError:
                            counts[v]["overflow_captions"] += 1
                            updated = ann["caption"]
                    if updated != ann["caption"]:
                        row = dict(ann, caption=updated, cocu_original_caption=ann["caption"])
                        outputs[v].append(row)
                        edits[v].append(row)
                        counts[v]["changed_captions"] += 1
                    else:
                        outputs[v].append(ann)
                    decisions[v]["captions"].append(updated)
            audit.write(json.dumps({"image_id": iid, "annotation_ids": [a["id"] for a in records[iid]],
                                    "raw_captions": raw, "coverage": coverage, "local_ids": local_ids,
                                    "cls_ids": cls[str(iid)][:30], "candidates": rows,
                                    "variants": decisions}, ensure_ascii=False, allow_nan=False) + "\n")
            if (pos + 1) % 1000 == 0:
                print(f"[BUILD] {pos + 1}/{len(ids)}", flush=True)

    for v, annotations in outputs.items():
        changed = edits[v]
        for start in range(0, len(changed), args.batch * 32):
            group = changed[start:start + args.batch * 32]
            feats = embed(teacher.text, teacher.clip, [a["caption"] for a in group], args.device, args.batch)
            for a, z in zip(group, feats):
                a["ann_feats"] = z
        target = work / v
        target.mkdir()
        torch.save(dict(base, images=images, annotations=annotations), target / "train.pth")
    return {v: dict(images=len(ids), annotations=5 * len(ids), **counts[v]) for v in variants}


