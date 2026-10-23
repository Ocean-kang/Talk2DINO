"""Corpus and candidate adapters for the unmodified Talk2DINO PTH format."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import torch
from .core import clean_vocabulary

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/Head_Match'
WTS=ROOT/'weights/Head_Match'


def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)


def corpus(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if not all(k in obj for k in ('images','annotations')):raise ValueError('Expected Talk2DINO train.pth images/annotations')
    anns=defaultdict(list)
    for ann in obj['annotations']:
        anns[int(ann['image_id'])].append(ann)
    rows=[]
    for im in obj['images']:
        iid=int(im['id']); group=anns[iid][:5]
        if len(group)!=5:raise ValueError(f'Image {iid}: expected five captions, got {len(group)}')
        if any('caption' not in x or 'ann_feats' not in x for x in group):
            raise ValueError(f'Image {iid}: raw caption / ann_feats missing')
        if 'disentangled_self_attn' not in im:raise ValueError(f'Image {iid}: missing disentangled_self_attn')
        heads=torch.as_tensor(im['disentangled_self_attn'])
        if heads.ndim!=2 or heads.shape[1]!=768:raise ValueError(f'Image {iid}: head tensor shape {tuple(heads.shape)} incompatible with ViT-B/14')
        if any(torch.as_tensor(x['ann_feats']).numel()!=512 for x in group):
            raise ValueError(f'Image {iid}: expected 512-D CLIP ann_feats')
        rows.append((iid,heads,group))
    if not rows:raise ValueError('No images in training PTH')
    return obj,rows


def _from_entry(entry):
    if isinstance(entry,list):return [x.get('concept','') if isinstance(x,dict) else x for x in entry]
    if not isinstance(entry,dict):return []
    result=[]
    for field in ('candidate_concepts','curated_concepts','concepts','selected_concepts'):
        if field in entry:result+=_from_entry(entry[field])
    return result


def candidates_from_json(paths):
    combined=defaultdict(list)
    for path in paths:
        obj=read_json(path)
        if isinstance(obj,list):
            for item in obj:
                if isinstance(item,dict) and 'image_id' in item:
                    combined[int(item['image_id'])].extend(_from_entry(item))
        elif isinstance(obj,dict):
            for iid,v in obj.items():
                if str(iid).isdigit():combined[int(iid)].extend(_from_entry(v))
        else:raise ValueError(f'Unsupported concept JSON: {path}')
    return {key:clean_vocabulary(names) for key,names in combined.items()}


def gallery_from_json(path):
    obj=read_json(path)
    if isinstance(obj,dict):obj=obj.get('concept',list(obj))
    if not isinstance(obj,list):raise ValueError('Expected gallery list or {concept:list}')
    return clean_vocabulary([x.get('concept','') if isinstance(x,dict) else x for x in obj])
