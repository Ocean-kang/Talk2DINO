"""Frozen CLIP ViT-B/16 image features and blockwise nearest-neighbor retrieval.
No DINO retrieval substitution: BSC-CR CoCu-Style explicitly uses CLIP.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from .core import ROOT,write_json


def find_image(root,im):
    filename=Path(str(im['file_name']))
    candidates=[root/filename,root/filename.name,root/'train2014'/filename.name,root/'val2014'/filename.name,
                root/'coco2014'/'train2014'/filename.name,root/'coco2014'/'val2014'/filename.name]
    for p in candidates:
        if p.is_file():return p
    raise FileNotFoundError(f'No JPEG for image {im["id"]}: {filename}; searched under {root}')


class ImageRows(Dataset):
    def __init__(self,rows,root,preprocess):self.rows=rows;self.root=root;self.preprocess=preprocess
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        path=find_image(self.root,row)
        with Image.open(path) as pic:pic=pic.convert('RGB');return int(row['id']),self.preprocess(pic)


@torch.no_grad()
def extract(base, image_root, cache, device='cuda',batch_size=96,workers=8,limit=0):
    if cache.is_file():
        saved=torch.load(cache,map_location='cpu',weights_only=False)
        if saved.get('metadata',{}).get('base_train')==str(base.resolve()) and saved['metadata'].get('limit')==limit:
            print('[CLIP IMAGE CACHE]',cache,flush=True);return cache
        raise ValueError(f'Cache from a different input: {cache}; choose a new cache path')
    import clip
    dataset=torch.load(base,map_location='cpu',weights_only=False)
    rows=dataset['images'][:limit or None]
    model,preprocess=clip.load('ViT-B/16',device=device,jit=False)
    model.eval().requires_grad_(False)
    loader=DataLoader(ImageRows(rows,image_root,preprocess),batch_size=batch_size,shuffle=False,
                      num_workers=workers,pin_memory=device.startswith('cuda'))
    ids=[];feats=[]
    for b,(image_ids,imgs) in enumerate(loader):
        vectors=F.normalize(model.encode_image(imgs.to(device,non_blocking=True)).float(),dim=-1).cpu()
        ids+=list(map(int,image_ids.tolist()));feats.append(vectors)
        if b%100==0:print(f'[CLIP IMAGE] {len(ids)}/{len(rows)}',flush=True)
    if len(set(ids))!=len(ids):raise ValueError('Duplicate image IDs')
    cache.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'ids':ids,'features':torch.cat(feats),
                'metadata':{'model':'ViT-B/16','base_train':str(base.resolve()),'limit':limit}},cache)
    return cache


@torch.no_grad()
def neighbors(cache,output,k=10,batch=64,device='cuda'):
    data=torch.load(cache,map_location='cpu',weights_only=False)
    ids=data['ids']; feat=F.normalize(data['features'].float(),dim=-1)
    if k>=len(ids):raise ValueError('k must be smaller than image count')
    # Use optional FAISS for speed, otherwise exact bounded-memory matmul.
    try:
        import faiss,numpy as np
        index=faiss.IndexFlatIP(feat.shape[1]);index.add(feat.numpy().astype('float32'))
        scores,ii=index.search(feat.numpy().astype('float32'),k+1)
        out={str(iid):[int(ids[j]) for j in inds if ids[j]!=iid][:k] for iid,inds in zip(ids,ii)}
    except ImportError:
        out={};all_features=feat.to(device)
        for start in range(0,len(ids),batch):
            val=all_features[start:start+batch]@all_features.T
            ii=val.topk(k+1,dim=-1).indices.cpu().tolist()
            for iid,inds in zip(ids[start:start+batch],ii):
                out[str(iid)]=[int(ids[j]) for j in inds if ids[j]!=iid][:k]
            if start%(batch*100)==0:print(f'[CLIP NEIGHBORS] {start}/{len(ids)}',flush=True)
    write_json(output,{'metadata':{'model':'CLIP ViT-B/16','topk':k,'cache':str(cache.resolve())},'neighbors':out})
    return output


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--base-train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--image-root',type=Path,required=True,help='COCO train2014 JPG root or its parent')
    p.add_argument('--cache',type=Path,default=ROOT/'outputs/Head_caption_completeness/clip_images.pth')
    p.add_argument('--neighbors',type=Path,default=ROOT/'outputs/Head_caption_completeness/clip_neighbors.json')
    p.add_argument('--gpu-device',default='cuda')
    p.add_argument('--batch-size',type=int,default=96)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--topk',type=int,default=10)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(argv)
    extract(a.base_train.resolve(),a.image_root.resolve(),a.cache.resolve(),a.gpu_device,a.batch_size,a.workers,a.limit)
    neighbors(a.cache.resolve(),a.neighbors.resolve(),a.topk,device=a.gpu_device)
if __name__=='__main__':
    # entry via `python -m experiments.Head_caption_completeness.prepare`
    main()
