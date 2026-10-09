#!/usr/bin/env python3
"""Exact chunked cosine Top-K for Local-CoCu CLS retrieval; excludes self.

Unlike the old implementation, this does not allocate an N-by-N matrix.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def build(feature, output, topk=50, chunk_size=256, device='cuda'):
    data = torch.load(feature, map_location='cpu')
    ids = list(data.keys())
    if len(ids) < topk + 1:
        raise ValueError(f'Insufficient images: {len(ids)} for topk={topk}')
    values = torch.stack([torch.as_tensor(data[k]).float().reshape(-1) for k in ids])
    values = F.normalize(values, dim=-1)
    if not torch.isfinite(values).all():
        raise ValueError('Non-finite or invalid CLS embeddings')
    device = torch.device(device)
    gallery = values.to(device)
    result = {}
    for start in range(0, len(ids), chunk_size):
        end = min(start+chunk_size, len(ids))
        scores = gallery[start:end] @ gallery.T
        # Remove self match exactly as existing Local-CoCu retrieval.
        ridx = torch.arange(end-start, device=device)
        scores[ridx, torch.arange(start,end,device=device)] = -float('inf')
        neighbors = scores.topk(topk, dim=1).indices.cpu().tolist()
        for key, row in zip(ids[start:end], neighbors):
            result[str(key)] = [ids[j] for j in row]
        if (start // chunk_size) % 20 == 0:
            print(f'[retrieval] {end}/{len(ids)}',flush=True)
    dest = Path(output)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'neighbors.json').write_text(json.dumps(result),encoding='utf-8')
    print('[OK]',dest/'neighbors.json')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--feature',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--topk',type=int,default=50)
    p.add_argument('--chunk-size',type=int,default=256)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    build(a.feature,a.output,a.topk,a.chunk_size,a.device)

if __name__=='__main__':
    main()
