#!/usr/bin/env python3
"""Pretrain structural checks; no implicit repair or synthetic captions."""
import argparse
from collections import Counter
import torch


def check(path, expected, original):
    data=torch.load(path,map_location='cpu')
    images=data['images']
    anns=data['annotations']
    ids={int(im['id']) for im in images}
    counts=Counter()
    for im in images:
        visual=im.get('disentangled_self_attn')
        if visual is None or getattr(visual,'ndim',0)!=2:
            raise RuntimeError(f"Image {im['id']} missing 2D disentangled_self_attn")
    for ann in anns:
        iid=int(ann['image_id'])
        if iid not in ids or 'ann_feats' not in ann:
            raise RuntimeError(f'Invalid annotation for image {iid}')
        if ann['ann_feats'].numel()!=512:
            raise RuntimeError(f'Expected CLIP text feature 512d: {ann.get("id")}')
        counts[iid]+=1
    if len(counts)!=len(ids) or set(counts.values())!={expected}:
        bad=[(iid,counts[iid]) for iid in ids if counts[iid]!=expected][:10]
        raise RuntimeError(f'Expected {expected} annotations per image; mismatches: {bad}')
    if {int(im['id']) for im in original['images']}!=ids:
        raise RuntimeError('Image-id set differs from original train.pth')
    print(f'[PASS] {path}: {len(ids)} images, {len(anns)} annotations, {expected}/image')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--base-train',required=True)
    p.add_argument('--expected',type=int,required=True)
    a=p.parse_args()
    check(a.dataset,a.expected,torch.load(a.base_train,map_location='cpu'))

if __name__=='__main__':
    main()
