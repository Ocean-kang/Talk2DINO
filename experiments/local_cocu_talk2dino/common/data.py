"""Input validation, lexical novelty checks, deterministic experiment records."""
from __future__ import annotations
from collections import defaultdict
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/local_cocu_talk2dino'
OUTPUT = ROOT / 'outputs/local_cocu_talk2dino'
WEIGHTS = ROOT / 'weights/local_cocu_talk2dino'

IRREGULAR = {'men':'man', 'women':'woman', 'people':'person', 'children':'child',
             'mice':'mouse', 'geese':'goose', 'feet':'foot', 'teeth':'tooth'}
WORD = re.compile(r'[a-z0-9]+')

def normalize(text):
    out = []
    for w in WORD.findall(str(text).casefold()):
        w = IRREGULAR.get(w, w)
        if len(w)>4 and w.endswith('ies'):
            w = w[:-3]+'y'
        elif len(w)>4 and w.endswith(('sses','shes','ches','xes')):
            w = w[:-2]
        elif len(w)>3 and w.endswith('s') and not w.endswith(('ss','us','is','ous')):
            w = w[:-1]
        out.append(w)
    return tuple(out)

def already_mentioned(concept, captions):
    needle = normalize(concept)
    if not needle: return True
    for caption in captions:
        hay = normalize(caption)
        if any(hay[i:i+len(needle)] == needle for i in range(len(hay)-len(needle)+1)):
            return True
    return False

def novel(items, captions):
    seen, out = set(), []
    for c in items:
        key = normalize(c)
        if not key or key in seen or already_mentioned(c, captions): continue
        seen.add(key)
        out.append(str(c).strip())
    return out

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def captions_and_images(coco_path):
    payload = read_json(coco_path)
    captions = defaultdict(list)
    for ann in payload['annotations']:
        captions[int(ann['image_id'])].append(str(ann['caption']))
    images = {int(i['id']): str(i['file_name']) for i in payload['images']}
    return dict(captions), images

def gallery_words(path):
    obj = read_json(path)
    if isinstance(obj, dict):
        arr = obj.get('concept', list(obj))
    else: arr = obj
    if not isinstance(arr, list): raise ValueError('Gallery must be list or {"concept": list}')
    unique={}
    for x in arr:
        v = (x.get('concept', '') if isinstance(x,dict) else x)
        if not isinstance(v,str): continue
        n = normalize(v)
        if n and n not in unique: unique[n] = ' '.join(WORD.findall(v.lower()))
    if not unique: raise ValueError('Empty concept gallery')
    return list(unique.values())

def audit_record(image_id, captions, rows, scheme, info=None):
    # rows contain confidence & best head, matching repository's detailed matcher format
    return {'image_id': int(image_id), 'scheme': scheme,
            'raw_captions': list(captions[:5]), 'concepts': rows,
            'num_concepts': len(rows), 'diagnostics': info or {}}

def validate_id_coverage(ids, captions, heads):
    missing = [i for i in ids if i not in captions or i not in heads or len(captions[i]) < 5]
    if missing: raise ValueError(f'Missing valid captions/head features for {len(missing)} IDs: {missing[:10]}')
