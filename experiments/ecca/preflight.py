"""Audit imports, existing inputs, source code and basic model compatibility; no installs."""
import argparse
import importlib
from pathlib import Path
from .common import ROOT, gallery, digest, write
from .runtime import model_args, Models, load_pth, image_path


def main():
    p=argparse.ArgumentParser(description=__doc__);model_args(p)
    p.add_argument('--train',required=True);p.add_argument('--val',required=True)
    p.add_argument('--gallery',required=True);p.add_argument('--image-root',required=True)
    a=p.parse_args()
    for name in ['torch','clip','torchvision','PIL','numpy','yaml','webdataset','wandb']:
        importlib.import_module(name);print('existing import OK:',name)
    from .coco_masks import BACKEND
    print('COCO diagnostic masks:',BACKEND,'(no pycocotools required)')
    names=gallery(a.gallery)
    train=load_pth(a.train);val=load_pth(a.val)
    for data,field in [(train,'disentangled_self_attn'),(val,'avg_self_attn_out')]:
        if not data.get('annotations') or not data.get('images'):raise ValueError('Empty source features')
        if len({int(x['id']) for x in data['images']})!=len(data['images']):raise ValueError('Duplicate image IDs')
        if field not in data['images'][0]:raise ValueError('Missing feature '+field)
        if 'ann_feats' not in data['annotations'][0]:raise ValueError('Missing ann_feats')
    if {r['image_id'] for r in train['annotations']}&{r['image_id'] for r in val['annotations']}:
        raise ValueError('Train/validation image overlap')
    model=Models(a)
    import numpy as np
    from PIL import Image
    with Image.open(image_path(a.image_root,train['images'][0])) as im:feature=model.extract(im)
    text=model.encode(['a photo of '+names[0]])
    if model.project(text).shape!=(1,768):raise ValueError('Bad projector output')
    model.close()
    print('Ready:',len(train['images']),'train images;',len(train['annotations']),'captions;',len(names),'concepts')
    print('OVSS additionally needs existing mmcv/mmseg/timm/omegaconf/us and local benchmark data.')


if __name__=='__main__':main()
