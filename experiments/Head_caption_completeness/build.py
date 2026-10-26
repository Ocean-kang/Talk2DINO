"""Reencode accepted captions only; preserve Talk2DINO images and five annotation IDs."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import torch
from .core import ROOT,write_json


def plan(base,records,subset=False):
    ann_map={int(x['id']):x for x in base['annotations']}
    if len(ann_map)!=len(base['annotations']):raise ValueError('Duplicate annotation IDs in raw PTH')
    groups=defaultdict(list)
    for row in base['annotations']:groups[int(row['image_id'])].append(row)
    image_ids = {int(im['id']) for im in base['images']}
    orphan_ids = {int(a['image_id']) for a in base['annotations']} - image_ids
    if orphan_ids:
        raise ValueError(f'Annotations reference missing images: {list(orphan_ids)[:10]}')

    distribution = Counter(len(groups[i]) for i in image_ids)
    print('[CAPTION COUNT DISTRIBUTION]', dict(sorted(distribution.items())), flush=True)
    print('[BUILD] Preserving all original image and annotation records.', flush=True)
    edits=[];changed=[];idset=set();imgset=set()
    for r in records:
        aid=int(r['annotation_id']);iid=int(r['image_id'])
        if aid in idset:raise ValueError(f'Duplicate decision ID: {aid}')
        idset.add(aid);imgset.add(iid)
        if aid not in ann_map or int(ann_map[aid]['image_id'])!=iid:raise ValueError(f'Wrong annotation ID/image ID: {aid}/{iid}')
        raw=str(ann_map[aid]['caption'])
        if raw!=r['raw_caption']:raise ValueError(f'Raw caption changed for annotation={aid}')
        edited=r['caption'];op=r['operation']
        if (edited!=raw)!=(op in ('ADD','REMOVE')):raise ValueError(f'Invalid operation/text consistency for {aid}')
        if edited!=raw:changed.append((aid,edited,op,r.get('concept')))
    if subset:
        # An intentional diagnostic subset; never silently transform full training into a subset.
        im_out=[im for im in base['images'] if int(im['id']) in imgset]
        anns_out=[ann for ann in base['annotations'] if int(ann['image_id']) in imgset]
        if {int(x['id']) for x in anns_out}!=idset:raise ValueError('Subset not exactly five captions/image')
    else:
        if idset!=set(ann_map):raise ValueError(f'Selection does not cover all raw annotations: {len(idset)}/{len(ann_map)}')
        im_out=base['images'];anns_out=base['annotations']
    updated={aid:(text,op,concept) for aid,text,op,concept in changed}
    final=[];edits=[]
    for ann in anns_out:
        aid=int(ann['id'])
        if aid in updated:
            row=dict(ann);row['bsc_cr_raw_caption']=row['caption']
            row['caption'],row['bsc_cr_operation'],row['bsc_cr_concept']=updated[aid]
            edits.append(row)
        else:row=ann
        final.append(row)
    return im_out,final,edits


def build(base_train,selection,output,manifest,device='cuda',batch_size=256,subset=False):
    data=torch.load(base_train,map_location='cpu',weights_only=False)
    with open(selection, encoding='utf-8') as f:
        records = (json.loads(line) for line in f if line.strip())
        images, annotations, edited = plan(data, records, subset)
    if edited:
        import clip
        enc,_=clip.load('ViT-B/16',device=device,jit=False)
        enc.eval().requires_grad_(False)
        with torch.no_grad():
            for i in range(0,len(edited),batch_size):
                part=edited[i:i+batch_size]
                tokens=clip.tokenize([x['caption'] for x in part],truncate=False).to(device)
                values=enc.encode_text(tokens).float().cpu()
                for ann,vec in zip(part,values):ann['ann_feats']=vec
                if i%(batch_size*50)==0:print(f'[CLIP REENCODE] {min(i+batch_size,len(edited))}/{len(edited)}',flush=True)
    counts=Counter(row['bsc_cr_operation'] for row in edited)
    out=dict(data);out['images']=images;out['annotations']=annotations
    out['bsc_cr_meta']={'base_train':str(base_train.resolve()),'selection':str(selection.resolve()),
                        'edited':dict(counts),'raw_caption_count':len(data['annotations']),
                        'new_caption_count':len(annotations),'student':'fresh original train.py'}
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+'.tmp')
    torch.save(out,temp);temp.replace(output)
    write_json(manifest,{'train_pth':str(output.resolve()),'selection':str(selection.resolve()),
                         'images':len(images),'captions':len(annotations),'edited':dict(counts),
                         'modified_annotation_ids':[int(x['id']) for x in edited],
                         'same_five_annotations':all(n==5 for n in Counter(int(a['image_id']) for a in annotations).values())})
    print('[BUILD]',output,'images',len(images),'captions',len(annotations),'edits',dict(counts),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--base-train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--subset-train',action='store_true')
    a=p.parse_args(argv)
    build(a.base_train.resolve(),a.selection.resolve(),a.output.resolve(),a.manifest.resolve(),a.device,a.batch_size,a.subset_train)
if __name__=='__main__':main()
