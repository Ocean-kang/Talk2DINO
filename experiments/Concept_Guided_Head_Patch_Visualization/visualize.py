"""Original-image overlays and offline HTML gallery."""
from __future__ import annotations

from html import escape
from pathlib import Path
import json
import re

import numpy as np
import torch
from PIL import Image

from experiments.Caption_Head_Visualization.visualize import (
    overlay_attention,
    make_panel,
    grid_cards,
)


MAP_FILES = {
    "p0_cls_to_patch": "p0_cls_to_patch.png",
    "p1_mean_incoming": "p1_mean_incoming.png",
    "p1_query_conditioned": "p1_query_conditioned.png",
    "p1_concept_propagation": "p1_concept_propagation.png",
    "p2_mean_feature_affinity": "p2_mean_feature_affinity.png",
    "p2_query_feature_affinity": "p2_query_feature_affinity.png",
}


def _shared_range(maps):
    arr = torch.as_tensor(maps).float().cpu().numpy()
    low, high = float(np.percentile(arr, 5)), float(np.percentile(arr, 99))
    if high - low < 1e-12:
        high = low + 1e-6
    return low, high


def save_image_visualizations(crop, relations, head_topk, caption_concepts, out_dir, alpha=0.65):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop.save(out_dir / "original_crop.png")

    map_families = {
        "p0_cls_to_patch": relations["p0_cls_to_patch"],
        "p1_mean_incoming": relations["p1_mean_incoming"],
        "p1_query_conditioned": relations["p1_query_conditioned"],
        "p1_concept_propagation": relations["p1_concept_propagation"],
        "p2_mean_feature_affinity": relations["p2_mean_feature_affinity"],
        "p2_query_feature_affinity": relations["p2_query_feature_affinity"],
    }
    ranges = {name: _shared_range(values) for name, values in map_families.items()}
    head_count = relations["p0_cls_to_patch"].shape[0]
    grid = relations["grid_size"]
    head_dirs = []
    for h in range(head_count):
        hdir = out_dir / "heads" / f"head_{h:02d}"
        hdir.mkdir(parents=True, exist_ok=True)
        top1 = head_topk[h][0] if head_topk[h] else {"concept": "(none)", "cosine": None}
        for name, maps in map_families.items():
            scalar = torch.as_tensor(maps[h]).reshape(grid, grid)
            overlay = overlay_attention(
                crop,
                scalar,
                alpha=alpha,
                global_range=ranges[name],
            )
            overlay.save(hdir / MAP_FILES[name])
        head_dirs.append({
            "head_index": h,
            "head_number": h + 1,
            "directory": f"heads/head_{h:02d}",
            "top1_concept": top1["concept"],
            "top1_cosine": top1["cosine"],
            "topk": head_topk[h],
            "maps": MAP_FILES,
        })

    # Compact P0 overview across all heads.
    p0_cards = []
    for h in range(head_count):
        image = overlay_attention(
            crop,
            torch.as_tensor(relations["p0_cls_to_patch"][h]).reshape(grid, grid),
            alpha=alpha,
            global_range=ranges["p0_cls_to_patch"],
        )
        top1 = head_topk[h][0] if head_topk[h] else {"concept": "(none)", "cosine": None}
        label = f"Head {h + 1} | {top1['concept']} | {top1['cosine']:.3f}" if top1["cosine"] is not None else f"Head {h + 1}"
        p0_cards.append(make_panel(image, label, "P0: original CLS-to-patch"))
    grid_cards(p0_cards, out_dir / "all_heads_p0.png", 4, title="All DINOv2 heads: P0 CLS-to-patch")

    # Caption concepts are visualized on their best-matching head using P1.
    cap_dir = out_dir / "caption_concepts"
    cap_dir.mkdir(exist_ok=True)
    for item in caption_concepts:
        filename = item.get("map_file")
        if not filename:
            continue
        item["map_path"] = f"caption_concepts/{filename}"

    return head_dirs


def write_html_gallery(output: str | Path, rows: list[dict]):
    output = Path(output)
    blocks = []
    for row in rows:
        iid = int(row["image_id"])
        base = f"images/{iid}"
        caption_tags = []
        for concept in row.get("caption_concepts", []):
            best = concept.get("best_head_number")
            score = concept.get("best_head_cosine")
            label = f"{concept['concept']} · H{best} · {score:.3f}" if best is not None else concept["concept"]
            caption_tags.append(f'<span class="tag">{escape(label)}</span>')
        head_cards = []
        for head in row.get("heads", []):
            hbase = f"{base}/{head['directory']}"
            topk = " · ".join(
                f"{escape(x['concept'])} ({x['cosine']:.3f})" for x in head.get("topk", [])
            )
            imgs = []
            for key, filename in MAP_FILES.items():
                title = {
                    "p0_cls_to_patch": "P0 · CLS → Patch",
                    "p1_mean_incoming": "P1 · Mean incoming",
                    "p1_query_conditioned": "P1 · Query-conditioned",
                    "p1_concept_propagation": "P1 · Concept propagation",
                    "p2_mean_feature_affinity": "P2 · Mean feature affinity",
                    "p2_query_feature_affinity": "P2 · Query feature affinity",
                }[key]
                imgs.append(
                    f'<figure><a href="{hbase}/{filename}" target="_blank">'
                    f'<img loading="lazy" src="{hbase}/{filename}" alt="{escape(title)}"></a>'
                    f'<figcaption>{escape(title)}</figcaption></figure>'
                )
            head_cards.append(
                f'<details class="head"><summary><b>Head {head["head_number"]}</b>'
                f' <span class="muted">Top-1: {escape(head["top1_concept"])} '
                f'({head["top1_cosine"]:.3f})</span></summary>'
                f'<p class="muted">Gallery Top-K: {topk}</p>'
                f'<div class="maps">{"".join(imgs)}</div></details>'
            )
        caption_links = []
        for concept in row.get("caption_concepts", []):
            if concept.get("map_path"):
                caption_links.append(
                    f'<a href="{base}/{concept["map_path"]}" target="_blank">'
                    f'{escape(concept["concept"])} · H{concept["best_head_number"]}</a>'
                )
        blocks.append(
            f'<article><header><h2>Image {iid}</h2>'
            f'<p>{escape(row.get("file_name", ""))}</p></header>'
            f'<img class="original" src="{base}/original_crop.png" alt="Original crop">'
            f'<p><b>Original captions</b><br>{"<br>".join(escape(x) for x in row.get("captions", []))}</p>'
            f'<p><b>Caption concepts</b><br>{"".join(caption_tags) or "<span class=muted>None matched in gallery</span>"}</p>'
            f'<p><b>Caption-concept P1 maps</b><br>{" · ".join(caption_links) or "<span class=muted>None</span>"}</p>'
            f'<p><a href="{base}/all_heads_p0.png">All-head P0 overview</a> · '
            f'<a href="{base}/diagnostics.json">Diagnostics JSON</a> · '
            f'<a href="{base}/relations.npz">Raw relation maps</a></p>'
            f'{"".join(head_cards)}</article>'
        )
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Concept-Guided Head-Patch Visualization</title>
<style>
:root{color-scheme:light}body{margin:0;background:#f3f5f8;color:#17202b;font:14px/1.55 system-ui,-apple-system,sans-serif}
main{max-width:1600px;margin:24px auto;padding:0 18px;display:grid;grid-template-columns:1fr;gap:22px}
article{background:#fff;border:1px solid #e1e6ed;border-radius:14px;padding:20px;box-shadow:0 3px 14px #17202b0b}
h1{font-size:26px;margin:24px 18px 4px}h2{font-size:20px;margin:0}header p,.muted{color:#687586}
.original{max-width:420px;width:100%;border-radius:8px;border:1px solid #e5e7eb}
.tag{display:inline-block;background:#edf4ff;border:1px solid #d8e7ff;border-radius:20px;padding:3px 9px;margin:3px 4px 3px 0}
a{color:#1559a6;text-decoration:none}a:hover{text-decoration:underline}
.head{border-top:1px solid #e7ebf0;padding:12px 0}.head summary{cursor:pointer}
.maps{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:12px}
figure{margin:0}.maps img{width:100%;border-radius:7px;border:1px solid #e2e7ed}
figcaption{font-size:12px;color:#5d6876;margin-top:3px}
@media(min-width:1100px){main{grid-template-columns:repeat(auto-fit,minmax(650px,1fr))}}
</style></head><body>
<h1>Concept-Guided Head-Patch Visualization</h1>
<p style="margin:0 18px;color:#687586">Frozen Talk2DINO · all heads · P0 / P1 / P2 · offline viewer</p>
<main>""" + "\n".join(blocks) + "</main></body></html>"
    (output / "index.html").write_text(html, encoding="utf-8")
