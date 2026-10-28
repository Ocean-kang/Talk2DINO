"""Lightweight reading of Talk2DINO's original COCO .pth format."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import torch


def torch_load_legacy(path):
    # Original Talk2DINO data uses a python dict with image/annotation lists.
    # Only load files produced by yourself / from a trusted source.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # older torch versions
        return torch.load(path, map_location='cpu')


def load_coco_feature_pth(path):
    obj = torch_load_legacy(path)
    if not isinstance(obj, dict) or not {'images', 'annotations'}.issubset(obj):
        raise ValueError('Expected COCO PTH with `images` and `annotations` keys')
    groups = defaultdict(list)
    for a in obj['annotations']:
        groups[int(a['image_id'])].append(a)
    return obj['images'], groups


def select_images(images, *, ids=None, offset=0, limit=20):
    if ids:
        wanted = set(map(int, ids))
        selected = [im for im in images if int(im['id']) in wanted]
        missing = wanted - {int(im['id']) for im in selected}
        if missing:
            raise ValueError(f'Image IDs not found in PTH: {sorted(missing)}')
        return selected
    return images[offset:offset + limit if limit else None]


def resolve_image(root, image_record):
    root = Path(root)
    file_name = Path(str(image_record['file_name']))
    basename = file_name.name
    split_candidates = ['train2014', 'val2014', 'test2014']
    if 'train' in str(file_name).lower():
        split_candidates = ['train2014', 'val2014', 'test2014']
    elif 'val' in str(file_name).lower():
        split_candidates = ['val2014', 'train2014', 'test2014']
    candidates = [root / file_name, root / basename]
    for split in split_candidates:
        candidates.extend([root / split / basename, root / 'coco2014' / split / basename])
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Cannot resolve image_id={image_record["id"]} file_name={file_name} under {root}. '
                            'Expected root/train2014/COCO_train2014_*.jpg or root/val2014/...')


def caption_rows(annotations, image_id):
    rows = annotations.get(int(image_id), [])
    if not rows:
        raise ValueError(f'No captions for image {image_id}')
    for a in rows:
        if not isinstance(a.get('caption'), str) or not a['caption'].strip():
            raise ValueError(f'Image {image_id}: missing raw caption')
    return rows
