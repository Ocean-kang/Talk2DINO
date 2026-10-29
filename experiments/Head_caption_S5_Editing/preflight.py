"""Check required entrypoints and input artifacts without altering existing files."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import importlib
import torch
from experiments.Head_caption_completeness.core import ROOT, gallery

MODULES=['src.model','src.dataset','src.train_util',
         'experiments.Head_Match.common.teacher',
         'experiments.Head_caption_completeness_diagnosis.run',
         'experiments.Head_caption_completeness_diagnosis.core',
         'experiments.Head_caption_completeness_diagnosis.text_evidence',
         'experiments.Head_caption_completeness.pipeline']

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--deep',action='store_true',help='Load all cached .pth files, verify image IDs and 5-caption count')
    p.add_argument('--train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--val',type=Path,default=ROOT/'outputs/coco2014_b14/val.pth')
    p.add_argument('--clip-images',type=Path,default=ROOT/'outputs/Head_caption_completeness/clip_images.pth')
    p.add_argument('--gallery',type=Path,default=ROOT/'experiments/local_cocu/MSCOCO14_train_gallery.json')
    a=p.parse_args(argv)
    required={ 'original train.py':ROOT/'train.py', 'Teacher config':ROOT/'configs/vitb_mlp_infonce.yaml',
               'Teacher weights':ROOT/'weights/vitb_mlp_infonce.pth',
               'train.pth':a.train,'val.pth':a.val,'gallery.json':a.gallery,'CLIP image cache':a.clip_images }
    for label,path in required.items():
        status='OK' if path.is_file() else 'MISSING'
        print(f'[{status}] {label}: {path}')
    absent=[str(x) for x in required.values() if not x.is_file()]
    for mod in MODULES+['clip']:
        try:importlib.import_module(mod);print('[IMPORT OK]',mod)
        except Exception as ex:raise RuntimeError(f'Import failed for {mod}: {ex}') from ex
    if absent:raise FileNotFoundError('Missing required artifacts: '+', '.join(absent))
    words=gallery(a.gallery);print('[GALLERY]',len(words),'concepts')
    if a.deep:
        base=torch.load(a.train,map_location='cpu',weights_only=False)
        cache=torch.load(a.clip_images,map_location='cpu',weights_only=False)
        ids=[int(x['id']) for x in base['images']]
        cache_ids=list(map(int,cache['ids']))
        count=Counter(int(x['image_id']) for x in base['annotations'])
        if len(set(ids))!=len(ids):raise ValueError('Duplicate source image IDs')
        # Preserve the exact raw caption count for every image (e.g. 5, 6, 7).
        # The original Talk2DINO Dataset trains over every annotation row.
        source_image_ids = set(ids)
        orphan_image_ids = set(count) - source_image_ids
        if orphan_image_ids:
            raise ValueError(f'Annotations reference missing image IDs: {sorted(orphan_image_ids)[:10]}')
        empty_image_ids = [iid for iid in ids if count[iid] == 0]
        if empty_image_ids:
            raise ValueError(f'Images without captions: {empty_image_ids[:10]}')
        caption_ids = [int(a['id']) for a in base['annotations']]
        if len(caption_ids) != len(set(caption_ids)):
            raise ValueError('Duplicate annotation IDs in source train.pth')
        distribution = Counter(count[iid] for iid in ids)
        print('[CAPTIONS/IMAGE]', dict(sorted(distribution.items())),
              '[TOTAL]', sum(count.values()), flush=True)
        if not set(ids).issubset(set(cache_ids)):
            raise ValueError('CLIP image cache is incomplete: source images missing')
        sample=base['images'][0]
        if tuple(torch.as_tensor(sample['disentangled_self_attn']).shape)!=(12,768):
            raise ValueError('Expected DINO 12 heads x 768')
        if tuple(torch.as_tensor(base['annotations'][0]['ann_feats']).flatten().shape)!=(512,):
            raise ValueError('Expected raw CLIP 512-dimensional captions')
        print('[STRUCTURE PASS]',len(ids),'images,',len(base['annotations']),'captions,',len(cache_ids),'cached images')
    print('[PREFLIGHT PASS]')
if __name__=='__main__':main()
