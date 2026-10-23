#!/usr/bin/env python3
"""Produce a genuine Talk2DINO train.pth with exactly 5 annotations per image.

Unchanged CLIP features remain byte-identical in memory; modified captions are
encoded with the repository's frozen CLIP ViT-B/16, never with a Teacher MLP.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import torch
from .io import corpus,read_json,write_json


def plan(base,selection,allow_subset=False):
    ims=base['images'];groups=defaultdict(list)
    for ann in base['annotations']:groups[int(ann['image_id'])].append(ann)
    source=selection['images']
    ids={int(x['id']) for x in ims}
    selected=set(map(int,source))
    if not allow_subset and ids!=selected:
        raise ValueError(f'Discovery IDs differ from base image IDs: train={len(ids)} selection={len(selected)}. '
                         'Explicit --subset-train is required for a subset.')
    if not selected.issubset(ids):raise ValueError('Discovery contains image IDs absent from base train')
    imgs=[];anns=[];edits=[];counts=Counter()
    for im in ims:
        iid=int(im['id'])
        if iid not in selected:continue
        chosen=groups[iid][:5]
        if len(chosen)!=5:raise ValueError(f'Image {iid} does not have exactly five captions')
        records=source[str(iid)]['captions']
        if len(records)!=5:raise ValueError(f'Image {iid}: selected captions must be exactly 5')
        # Key by original annotation IDs, not incidental JSON ordering.
        decisions={int(r['annotation_id']):r for r in records}
        if len(decisions)!=5 or set(decisions)!={int(a['id']) for a in chosen}:
            raise ValueError(f'Image {iid}: caption annotation_id coverage mismatch')
        imgs.append(im)
        for ann in chosen:
            row=dict(ann); change=decisions[int(row['id'])]
            if str(row['caption'])!=change['raw_caption']:raise ValueError(f'Image {iid}: stale Raw Caption in selection')
            if change['augmented_caption']!=row['caption']:
                row['caption']=change['augmented_caption']
                row['head_match_original_caption']=change['raw_caption']
                row['head_match_added_concepts']=[x['concept'] for x in change['added']]
                edits.append(row)
            anns.append(row)
            counts['annotations']+=1
        counts['images']+=1
    counts['modified']=len(edits)
    return imgs,anns,edits,counts


def build(base_path,selection_path,out_path,manifest_path,device='cuda',batch_size=256,subset_train=False):
    # Avoid duplicate full source copy; only changed annotation rows are shallow copied.
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    selection=read_json(selection_path)
    if Path(selection['meta']['base_train']).resolve()!=Path(base_path).resolve():
        raise ValueError('Selection was mined against a different base train.pth')
    images,annotations,edited,counts=plan(base,selection,subset_train)
    if not images:raise ValueError('No training examples')
    if edited:
        import clip
        model,_=clip.load('ViT-B/16',device=device,jit=False)
        model.eval().requires_grad_(False)
        with torch.no_grad():
            for i in range(0,len(edited),batch_size):
                batch=edited[i:i+batch_size]
                tokens=clip.tokenize([x['caption'] for x in batch],truncate=True).to(device)
                values=model.encode_text(tokens).float().cpu()
                for row,feature in zip(batch,values):row['ann_feats']=feature
                if (i//batch_size)%100==0:print(f'[CLIP] {min(i+batch_size,len(edited))}/{len(edited)}',flush=True)
    obj=dict(base);obj['images']=images;obj['annotations']=annotations
    obj['head_match_meta']={'selection':str(selection_path),'5_captions_per_image':True,
                            'student_initialization':'fresh_original_train.py','counts':dict(counts)}
    out_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,out_path)
    write_json(manifest_path,{'selection':str(selection_path),'train_pth':str(out_path),'counts':dict(counts),
                              'image_ids':[int(x['id']) for x in images],
                              'changed_annotation_ids':[int(x['id']) for x in edited]})
    print('[BUILD]',out_path,dict(counts),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-train',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--subset-train',action='store_true')
    a=p.parse_args()
    build(a.base_train,a.selection,a.output,a.manifest,a.device,a.batch_size,a.subset_train)
if __name__=='__main__':main()
