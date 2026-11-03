"""Small shared utilities. Standard library + existing NumPy only."""
import hashlib
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def object_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def offline():
    import runpy
    runpy.run_path(str(HERE / 'offline_guard/sitecustomize.py'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


@lru_cache(maxsize=16)
def repo_module(path):
    path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError('Required existing repository utility: ' + str(path))
    spec = importlib.util.spec_from_file_location('ecca_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lexical():
    return repo_module('experiments/local_cocu_novel/filter_novel.py')


def gallery(path):
    obj = read(path)
    # Repository validator, deterministic order (validator returns an unordered set).
    normalized = repo_module('experiments/local_cocu/utils/concept_utils.py').normalize_gallery(obj)
    lex = lexical()
    names = sorted({' '.join(lex.tokenize(x)) for x in normalized if isinstance(x, str)})
    names = [x for x in names if x]
    if len(names) < 3:
        raise ValueError('Gallery needs at least three distinct concepts')
    return names


def normalize(a, axis=-1):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-12)


def iou(pred, gt, valid):
    p, g = np.asarray(pred, bool) & valid, np.asarray(gt, bool) & valid
    union = np.count_nonzero(p | g)
    return float(np.count_nonzero(p & g) / union) if union else None


def top_mask(a, fraction=.2):
    a = np.asarray(a).reshape(-1)
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    k = max(1, int(np.ceil(len(a) * fraction)))
    mask = np.zeros(len(a), bool)
    mask[np.argsort(-a, kind='stable')[:k]] = True
    return mask


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def grouped_bootstrap(rows, metric, repetitions=1000, seed=123):
    """Resample images, preserving all concept rows within each image."""
    groups = {}
    for r in rows:
        if r.get(metric) is not None:
            groups.setdefault(str(r['image_id']), []).append(float(r[metric]))
    if not groups:
        return {'mean': None, 'ci95': None, 'images': 0}
    vals = list(groups.values())
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(repetitions):
        chosen = [vals[k] for k in rng.randint(len(vals), size=len(vals))]
        boots.append(float(np.mean([v for g in chosen for v in g])))
    return {'mean': float(np.mean([v for g in vals for v in g])),
            'ci95': np.percentile(boots, [2.5, 97.5]).tolist(), 'images': len(vals)}


def pool_region(patches, attention, evidence, heads):
    """Mean selected attention gated by positive contrastive patch evidence."""
    weight = np.mean(attention[heads], axis=0) * np.maximum(evidence, 0)
    if not np.isfinite(weight).all() or weight.sum() <= 1e-12:
        return None
    return normalize(weight @ patches / weight.sum())


def scored_evidence(patches, attention, projected, candidate_ids, negative_ids, temperature=.07):
    """Contrast each candidate against confusable vocabulary, masking aliases upstream.

    Scores are teacher proxies, never probabilities or object-presence GT.
    """
    if temperature <= 0 or not negative_ids:
        raise ValueError('Need positive temperature and negative vocabulary')
    sim = projected[candidate_ids] @ normalize(patches).T
    neg = projected[negative_ids] @ normalize(patches).T
    margin = sim - neg.max(axis=0, keepdims=True)
    evidence = 1 / (1 + np.exp(-np.clip(margin / temperature, -30, 30)))
    q = evidence @ attention.T
    return sim, margin, q


def selection_pass(row, config):
    return (row['q'] >= config['q_min'] and row['positive_mass'] >= config['mass_min']
            and row['flip_iou'] >= config['flip_min'] and row['head_score'] >= config['head_min'])
