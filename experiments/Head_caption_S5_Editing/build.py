"""Build edited 5-caption train.pth, preserving source IDs and image/head features.

Original Talk2DINO train.py consumes ann_feats, not the caption string directly;
every edited caption MUST be re-encoded by original CLIP ViT-B/16.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import torch
from experiments.Head_caption_completeness.core import ROOT,write_json

OUT=ROOT/'outputs/Head_caption_S5_Editing'


def iter_manifest(file):
    seen=set()
    with Path(file).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line);aid=int(row['annotation_id'])
            if aid in seen:raise ValueError(f'Duplicate manifest annotation_id={aid}')
            seen.add(aid)
            yield row


def plan(base, manifest, subset=False):
    """S5_KEEP_EXTRA_V1: keep unselected extra COCO14 captions unchanged.

    Every selected image has exactly five edited/KEEP decisions. Original
    COCO train.pth can contain more than five captions for some images.
    Their original caption and ann_feats are copied without re-encoding.
    """
    from collections import Counter

    images = base['images']
    anns = base['annotations']
    image_ids = [int(im['id']) for im in images]
    image_set = set(image_ids)
    if len(image_ids) != len(image_set):
        raise ValueError('Duplicate source image IDs')

    aid_map = {}
    original_counts = Counter()
    for ann in anns:
        aid = int(ann['id'])
        iid = int(ann['image_id'])
        if aid in aid_map:
            raise ValueError(f'Duplicate source annotation ID: {aid}')
        if iid not in image_set:
            raise ValueError(f'Annotation {aid} references unknown image {iid}')
        aid_map[aid] = ann
        original_counts[iid] += 1
    if any(original_counts[i] < 5 for i in image_ids):
        raise ValueError('Source data has an image with fewer than 5 captions')

    # Do NOT require one selection per source caption. Selection intentionally
    # processes five captions/image; remaining source captions are kept raw.
    decision = {}
    selection_counts = Counter()
    for row in manifest:
        aid = int(row['annotation_id'])
        iid = int(row['image_id'])
        if aid in decision:
            raise ValueError(f'Duplicate manifest annotation ID: {aid}')
        if aid not in aid_map:
            raise ValueError(f'Unknown annotation ID in manifest: {aid}')
        if iid != int(aid_map[aid]['image_id']):
            raise ValueError(f'Manifest image_id mismatch for annotation {aid}')
        if row['raw_caption'] != aid_map[aid]['caption']:
            raise ValueError(f'Stale raw_caption in manifest for annotation {aid}')
        decision[aid] = row
        selection_counts[iid] += 1

    selected_images = set(selection_counts)
    if not selected_images:
        raise ValueError('Manifest is empty')
    if any(n != 5 for n in selection_counts.values()):
        raise ValueError('Manifest must contain exactly 5 selected captions per image')
    if not subset and selected_images != image_set:
        raise ValueError('Full-data build requires selecting every original image')

    out_images = images if not subset else [im for im in images if int(im['id']) in selected_images]
    out_anns = []
    changed = []
    stats = Counter()
    for ann in anns:
        aid = int(ann['id'])
        iid = int(ann['image_id'])
        if subset and iid not in selected_images:
            continue
        row = decision.get(aid)
        if row is None:
            # Extra captions beyond the five per image: preserve EXACT original
            # text AND cached ann_feats (do not recompute or change original IDs).
            out_anns.append(ann)
            stats['KEEP_EXTRA'] += 1
            continue

        before = ann['caption']
        after = row['edited_caption']
        if not isinstance(after, str) or not after.strip():
            raise ValueError(f'Empty edited caption for annotation {aid}')
        op = row['operation']
        if after == before:
            if op != 'KEEP':
                raise ValueError(f'Unchanged annotation {aid} must use operation KEEP, got {op}')
            out_anns.append(ann)
            stats['KEEP'] += 1
            continue
        if op not in ('ADD', 'REMOVE', 'REMOVE_ADD'):
            raise ValueError(f'Edited annotation {aid} has invalid operation: {op}')
        modified = dict(ann)
        modified['caption'] = after
        modified['s5edit_raw_caption'] = before
        modified['s5edit_operation'] = op
        modified['s5edit_fixed_head'] = row.get('fixed_head')
        modified['s5edit_add'] = row.get('add_concept')
        modified['s5edit_remove'] = row.get('remove_concept')
        out_anns.append(modified)
        changed.append(modified)
        stats[op] += 1

    # Preserve the entire source annotation set in FULL mode, including extras.
    expected_total = (len(anns) if not subset else
                      sum(original_counts[i] for i in selected_images))
    if len(out_anns) != expected_total:
        raise ValueError(f'Lost annotations: expected {expected_total}, got {len(out_anns)}')
    output_counts = Counter(int(a['image_id']) for a in out_anns)
    if set(output_counts) != (image_set if not subset else selected_images):
        raise ValueError('Output image set differs from the required image set')
    for iid in output_counts:
        if output_counts[iid] != original_counts[iid]:
            raise ValueError(f'Image {iid} lost captions during build')

    return out_images, out_anns, changed, stats



@torch.no_grad()
def build(base_path,manifest_path,output,meta_path,subset=False,device='cuda',batch_size=256):
    base_path=Path(base_path).resolve();manifest_path=Path(manifest_path).resolve()
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    manifest=list(iter_manifest(manifest_path))
    ims,anns,edited,counts=plan(base,manifest,subset)
    if edited:
        import clip
        model,_=clip.load('ViT-B/16',device=device,jit=False)
        model.eval().requires_grad_(False)
        for start in range(0,len(edited),batch_size):
            group=edited[start:start+batch_size]
            # Hard-fail instead of truncating away recently appended concepts.
            tok=clip.tokenize([x['caption'] for x in group],truncate=False).to(device)
            feats=model.encode_text(tok).float().cpu()
            for a,f in zip(group,feats):a['ann_feats']=f
            if start%(batch_size*50)==0:print(f'[CLIP REENCODE] {min(start+batch_size,len(edited))}/{len(edited)}',flush=True)
    out=dict(base);out['images']=ims;out['annotations']=anns
    out['s5edit_meta']={'base_train':str(base_path),'selection':str(manifest_path),'counts':dict(counts),
                        'source_annotations':len(base['annotations']), 'source_images':len(base['images']),
                        'student':'fresh original train.py'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.temporary');torch.save(out,temporary);temporary.replace(output)
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    write_json(meta_path,{'output':str(output.resolve()),'base_train':str(base_path),
        'selection':str(manifest_path),'sha256_selection':digest,'num_images':len(ims),
        'num_captions':len(anns),'reencoded_captions':len(edited),'counts':dict(counts),
        'modified_annotation_ids':[int(r['id']) for r in edited]})
    print('[BUILD] train=',output,'images=',len(ims),'captions=',len(anns),'changed=',len(edited),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['E0','E1','E2','E3'],required=True)
    p.add_argument('--base-train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--selection-dir',type=Path,default=OUT/'discovery')
    p.add_argument('--output',type=Path)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--subset-train',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args(argv)
    meta_path=a.selection_dir/'metadata.json'
    if not meta_path.is_file():raise FileNotFoundError('Run select.py first: '+str(meta_path))
    info=json.loads(meta_path.read_text(encoding='utf-8'))
    if str(a.base_train.resolve())!=info['base_train']:raise ValueError('Base train does not match discovery metadata')
    path=a.selection_dir/f'{a.variant}.jsonl'
    if not path.is_file():raise FileNotFoundError(path)
    output=a.output or OUT/'training'/a.variant/'train.pth'
    manifest=a.manifest or OUT/'training'/a.variant/'manifest.json'
    build(a.base_train,path,output,manifest,a.subset_train,a.device,a.batch_size)
if __name__=='__main__':main()
