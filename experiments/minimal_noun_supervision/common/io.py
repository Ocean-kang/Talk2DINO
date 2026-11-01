"""Small local IO helpers; input PTHs must come from a trusted source."""
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_features(path):
    import torch
    data = torch.load(path, map_location='cpu')
    if not isinstance(data, dict) or not {'images', 'annotations'} <= data.keys():
        raise ValueError('Expected original COCO-style PTH with images and annotations: ' + str(path))
    ids = [int(im['id']) for im in data['images']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image IDs in ' + str(path))
    known = set(ids)
    if any(int(a['image_id']) not in known for a in data['annotations']):
        raise ValueError('Annotation references an absent image')
    return data


def manifest(method, source, records, raw_annotations, bank):
    return {'schema_version': 1, 'method': method, 'source_features': str(Path(source).resolve()),
            'text_bank': str(Path(bank).resolve()), 'raw_annotation_count': raw_annotations,
            'records': records}
