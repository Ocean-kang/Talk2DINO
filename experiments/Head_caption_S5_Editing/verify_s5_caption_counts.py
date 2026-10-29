#!/usr/bin/env python3
"""Validate same image IDs, annotation IDs, and per-image caption counts between two PTH files.

Usage (from Talk2DINO root):
  python /path/to/verify_s5_caption_counts.py \
      outputs/coco2014_b14/train.pth \
      outputs/Head_caption_S5_Editing/training/E3/train.pth
"""
from __future__ import annotations
from collections import Counter
import argparse
from pathlib import Path
import torch


def summarize(file):
    payload=torch.load(file,map_location='cpu',weights_only=False)
    images={int(x['id']) for x in payload['images']}
    if len(images)!=len(payload['images']):raise ValueError(f'{file}: duplicated image id(s)')
    ann_ids=[int(x['id']) for x in payload['annotations']]
    if len(ann_ids)!=len(set(ann_ids)):raise ValueError(f'{file}: duplicated annotation id(s)')
    grouped=Counter(int(x['image_id']) for x in payload['annotations'])
    missing=images-set(grouped)
    orphans=set(grouped)-images
    if missing or orphans:raise ValueError(f'{file}: empty images={list(missing)[:5]}, orphan captions={list(orphans)[:5]}')
    distr=Counter(grouped.values())
    print(f'{file}: images={len(images)}, captions={len(ann_ids)}, count distribution={dict(sorted(distr.items()))}')
    return images,set(ann_ids),grouped


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('edited',type=Path)
    args=parser.parse_args()
    a=summarize(args.source)
    b=summarize(args.edited)
    if a[0]!=b[0]:raise ValueError(f'Image ID set mismatch; lost={len(a[0]-b[0])}, extra={len(b[0]-a[0])}')
    if a[1]!=b[1]:raise ValueError(f'Annotation ID set mismatch; lost={len(a[1]-b[1])}, extra={len(b[1]-a[1])}')
    bad=[(iid,a[2][iid],b[2][iid]) for iid in a[0] if a[2][iid]!=b[2][iid]]
    if bad:raise ValueError(f'Per-image caption count mismatch: {bad[:10]}')
    print('[PASS] Image IDs, annotation IDs, and per-image caption counts preserved.')

if __name__=='__main__':main()
