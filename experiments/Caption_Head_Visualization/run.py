"""Run original Talk2DINO caption/head attention visualization on COCO2014.

Entry from repository ROOT:
python -m experiments.Caption_Head_Visualization.run --features ... --image-root ...
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import escape as html_escape
import json
from pathlib import Path
import sys
import traceback

import torch
import torch.nn.functional as F
from PIL import Image

from .core import compute_matches, describe_matches
from .data import caption_rows, load_coco_feature_pth, resolve_image, select_images, torch_load_legacy
from .visualize import save_visualizations
from .vision import DinoHeadExtractor

ROOT = Path(__file__).resolve().parents[2]


def jsonify(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)


def sha256_file(path):
    hash_obj = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hash_obj.update(block)
    return hash_obj.hexdigest()


def load_projector(config, weight, device):
    # Reuse EXACT original module. This imports clip via src.model, even though
    # cached ann_feats means CLIP text encoder need not be loaded in this run.
    from src.model import ProjectionLayer
    projector = ProjectionLayer.from_config(str(config))
    state = torch_load_legacy(weight)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    if not isinstance(state, dict):
        raise ValueError('Projector checkpoint should be state_dict or model/state_dict wrapper')
    state = {k.removeprefix('module.').removeprefix('projector.'): v for k, v in state.items()}
    projector.load_state_dict(state, strict=True)
    return projector.to(device).eval().requires_grad_(False)


def parse_ids(s):
    if not s:
        return []
    if isinstance(s, (list, tuple)):
        s = ','.join(s)
    return [int(t.strip()) for t in s.split(',') if t.strip()]


def require_path(path, kind):
    if not path.exists():
        raise FileNotFoundError(f'Missing {kind}: {path}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Visualize raw COCO captions -> DINOv2 heads using ORIGINAL Talk2DINO projector')
    p.add_argument('--features', type=Path, default=ROOT / 'outputs/coco2014_b14/train.pth',
                   help='COCO14 PTH with images/annotations (raw captions + ann_feats)')
    p.add_argument('--image-root', type=Path, required=True,
                   help='COCO root: parent of train2014/, val2014/ or exact JPEG directory')
    p.add_argument('--weights', type=Path, default=ROOT / 'weights/vitb_mlp_infonce.pth')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/vitb_mlp_infonce.yaml')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/Caption_Head_Visualization')
    p.add_argument('--limit', type=int, default=20, help='Number of images after --offset; 0 means all')
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--image-ids', default='', help='Comma-separated COCO image_ids; overrides offset/limit')
    p.add_argument('--device', default='cuda')
    p.add_argument('--size', type=int, default=448, help='Use 448 to reproduce original extracted heads')
    p.add_argument('--feature-source', choices=['auto', 'cached', 'recomputed'], default='auto',
                   help='auto: use original PTH head embeddings if present, or extracted otherwise')
    p.add_argument('--text-source', choices=['cached', 'clip'], default='cached',
                   help='cached uses original ann_feats; clip re-encodes original caption strings with ViT-B/16')
    p.add_argument('--attention-norm', choices=['per-head', 'global'], default='per-head')
    p.add_argument('--alpha', type=float, default=.65, help='Strength of heatmap overlay, 0..1')
    p.add_argument('--no-all-heads', action='store_true', help='Do not save all_12_heads.png; individual caption maps still saved')
    p.add_argument('--no-skip-existing', action='store_true', help='Reprocess images even if diagnostics.json exists')
    p.add_argument('--min-cache-cosine', type=float, default=.99,
                   help='If cached/recomputed head cosine below this, mark WARNING (no silent substitution)')
    p.add_argument('--strict-cache', action='store_true', help='Fail this image if cached/recomputed heads disagree')
    p.add_argument('--fail-fast', action='store_true', help='Abort instead of recording per-image errors')
    p.add_argument('--dinov2-repo', default='facebookresearch/dinov2',
                   help='Github torch.hub repo or local DINOv2 repository path')
    p.add_argument('--hub-source', choices=['github', 'local'], default='github')
    args = p.parse_args(argv)
    if args.limit < 0 or args.offset < 0:
        p.error('--limit and --offset must be nonnegative')
    if not (0 <= args.alpha <= 1):
        p.error('--alpha must be in [0,1]')
    if args.size != 448:
        p.error('This experiment uses --size 448 to match original Talk2DINO ViT-B/14-reg features')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA not available; use --device cpu for an intentionally slower run')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for path, name in [(args.features, 'feature PTH'), (args.weights, 'Talk2DINO weight'), (args.config, 'model config')]:
        require_path(path, name)
    if not args.image_root.is_dir():
        raise NotADirectoryError(f'Cannot find image root: {args.image_root}')
    images, by_image = load_coco_feature_pth(args.features)
    selected = select_images(images, ids=parse_ids(args.image_ids), offset=args.offset, limit=args.limit)
    if not selected:
        raise ValueError('Selection returned zero images')
    print(f'[DATA] {len(selected)} images selected; {len(images)} images in PTH', flush=True)
    projector = load_projector(args.config, args.weights, args.device)
    model = DinoHeadExtractor(args.device, args.size, repo=args.dinov2_repo, hub_source=args.hub_source)
    clip_encoder = None
    clip_pkg = None
    if args.text_source == 'clip':
        import clip
        clip_pkg = clip
        clip_encoder, _ = clip.load('ViT-B/16', device=args.device, jit=False)
        clip_encoder.eval().requires_grad_(False)
    config = {
        'features': str(args.features.resolve()),
        'weights': str(args.weights.resolve()),
        'weights_sha256': sha256_file(args.weights),
        'config': str(args.config.resolve()),
        'image_root': str(args.image_root.resolve()),
        'num_images': len(selected),
        'selected_image_ids': [int(im['id']) for im in selected],
        'device': str(args.device),
        'image_size': args.size,
        'image_model': 'dinov2_vitb14_reg',
        'text_model': 'CLIP ViT-B/16',
        'text_source': args.text_source,
        'feature_source': args.feature_source,
        'all_heads_grid': not args.no_all_heads,
        'strict_cache': args.strict_cache,
        'dinov2_repo': args.dinov2_repo,
        'hub_source': args.hub_source,
        'attention_norm': args.attention_norm,
        'attention_alpha': args.alpha,
        'min_cache_cosine': args.min_cache_cosine,
        'semantics': 'Top-1 cosine similarity per caption, 12 DINOv2 last-layer attention-head embeddings; CLS-to-patch heatmap',
    }
    (args.output / 'run_config.json').write_text(jsonify(config), encoding='utf-8')
    success, errors = [], []
    counts = Counter()
    try:
        for ix, image_record in enumerate(selected, start=1):
            iid = int(image_record['id'])
            out = args.output / 'images' / str(iid)
            if (out / 'diagnostics.json').is_file() and not args.no_skip_existing:
                try:
                    existing = json.loads((out / 'diagnostics.json').read_text(encoding='utf-8'))
                    if existing.get('weight_sha256') == config['weights_sha256'] and existing.get('config_fingerprint') == _fingerprint(config):
                        success.append(existing)
                        for h, n in enumerate(existing['head_use_counts']):
                            counts[h] += int(n)
                        print(f'[{ix}/{len(selected)}] image {iid}: SKIP identical run', flush=True)
                        continue
                except Exception:
                    pass
            print(f'[{ix}/{len(selected)}] image {iid}: extracting and matching', flush=True)
            try:
                rows = caption_rows(by_image, iid)
                captions = [a['caption'] for a in rows]
                ids = [a.get('id') for a in rows]
                filename = resolve_image(args.image_root, image_record)
                with Image.open(filename) as im:
                    original = im.convert('RGB').copy()
                new_heads, attn, cropped = model.extract(original)
                source = 'recomputed'
                cache_cosines = None
                warning = None
                if 'disentangled_self_attn' in image_record:
                    old_heads = torch.as_tensor(image_record['disentangled_self_attn']).float()
                    if old_heads.shape == new_heads.shape:
                        cache_cosines = F.cosine_similarity(old_heads, new_heads, dim=-1).tolist()
                        if min(cache_cosines) < args.min_cache_cosine:
                            warning = (f'Cached/re-extracted heads disagree: min cosine={min(cache_cosines):.5f} '
                                       f'< {args.min_cache_cosine}; inspect DINO version/preprocessing.')
                            if args.strict_cache:
                                raise ValueError(warning)
                    else:
                        warning = f'Cached heads shape {tuple(old_heads.shape)} != newly extracted {tuple(new_heads.shape)}'
                        if args.strict_cache:
                            raise ValueError(warning)
                else:
                    old_heads = None
                if args.feature_source == 'cached' and (old_heads is None or old_heads.shape != new_heads.shape):
                    raise ValueError('Cached head features unavailable/mismatched; use --feature-source recomputed')
                if args.feature_source != 'recomputed' and old_heads is not None and old_heads.shape == new_heads.shape:
                    matching_heads = old_heads
                    source = 'cached'
                else:
                    matching_heads = new_heads
                if args.text_source == 'cached':
                    if any('ann_feats' not in a for a in rows):
                        raise ValueError('Missing cached ann_feats. Run original text_features_extraction.py or use --text-source clip')
                    text_feats = torch.stack([torch.as_tensor(a['ann_feats']).float().flatten() for a in rows])
                else:
                    with torch.no_grad():
                        toks = clip_pkg.tokenize(captions, truncate=True).to(args.device)
                        text_feats = clip_encoder.encode_text(toks).float()
                if text_feats.shape[1] != 512 or matching_heads.shape != (12, 768):
                    raise ValueError(f'Expected [C,512] CLIP and [12,768] DINO; got {tuple(text_feats.shape)}, {tuple(matching_heads.shape)}')
                scores = compute_matches(projector, text_feats, matching_heads, device=args.device)
                description = describe_matches(scores, captions, ids, attn)
                description.update({
                    'image_id': iid,
                    'file_name': image_record['file_name'],
                    'resolved_image_path': str(filename.resolve()),
                    'weight_sha256': config['weights_sha256'],
                    'config_fingerprint': _fingerprint(config),
                    'feature_source_used': source,
                    'cached_recomputed_head_cosine': [round(float(x), 7) for x in cache_cosines] if cache_cosines is not None else None,
                    'feature_mismatch_warning': warning,
                    'normalized_heatmap_is_semantic_mask': False,
                })
                out.mkdir(parents=True, exist_ok=True)
                save_visualizations(cropped, attn, scores, description, out, alpha=args.alpha,
                                    norm_mode=args.attention_norm, all_heads=not args.no_all_heads,
                                    full_resized=model.resize(original))
                (out / 'diagnostics.json').write_text(jsonify(description), encoding='utf-8')
                if warning:
                    print(f'  [WARNING] {warning}', flush=True)
                success.append(description)
                for h, n in enumerate(description['head_use_counts']):
                    counts[h] += int(n)
            except Exception as e:
                err = {'image_id': iid, 'error': repr(e), 'traceback': traceback.format_exc()}
                errors.append(err)
                print(f'  [ERROR] image {iid}: {e}', file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
    finally:
        model.close()

    (args.output / 'matches.jsonl').write_text(
        ''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in success), encoding='utf-8')
    (args.output / 'errors.jsonl').write_text(
        ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in errors), encoding='utf-8')
    summary = {
        'num_selected': len(selected),
        'num_success': len(success),
        'num_errors': len(errors),
        'mean_unique_heads_used': round(sum(x['unique_heads_used'] for x in success) / len(success), 6) if success else None,
        'mean_matching_margin': round(sum(x['mean_matching_margin'] for x in success) / len(success), 6) if success else None,
        'head_caption_counts_1_to_12': [int(counts.get(i, 0)) for i in range(12)],
        'num_feature_mismatch_warnings': sum(bool(x['feature_mismatch_warning']) for x in success),
        'output': str(args.output.resolve()),
    }
    (args.output / 'summary.json').write_text(jsonify(summary), encoding='utf-8')
    write_gallery(args.output, success)
    print(f'[DONE] {len(success)} success, {len(errors)} errors. Output: {args.output}', flush=True)
    if errors:
        print('See errors.jsonl. Re-run with --fail-fast to see the first error.', file=sys.stderr)
        return 1
    return 0



def write_gallery(output, rows):
    """Offline index.html; all preview images are local relative PNGs."""
    blocks = []
    for r in rows:
        iid = int(r['image_id'])
        base = f'images/{iid}'
        cap = '<br>'.join(
            f'<b>H{v["head_number"]}</b> ({v["cosine_similarity"]:.3f}) - ' + html_escape(v['caption'])
            for v in r['captions']
        )
        all_heads_link = f'<a href="{base}/all_12_heads.png">12 heads</a> · ' if (Path(output) / base / 'all_12_heads.png').exists() else ''
        blocks.append(f'<article><a href="{base}/overview.png"><img loading="lazy" src="{base}/overview.png" alt="overview {iid}"></a>'
                      f'<h3>Image {iid} · {r["unique_heads_used"]} distinct heads</h3>'
                      f'<p>{cap}</p><p><a href="{base}/similarity_matrix.png">Similarity matrix</a> · '
                      f'{all_heads_link}<a href="{base}/diagnostics.json">Raw scores</a></p></article>')
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Original Talk2DINO: Caption-Head correspondence</title><style>
body{background:#f4f6fa;color:#1b2636;font:15px/1.5 system-ui,sans-serif;margin:24px}header{max-width:1150px;margin:0 auto 24px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px;max-width:1500px;margin:auto}
article{background:white;padding:16px;border-radius:12px;box-shadow:0 2px 12px #dde1e7}img{max-width:100%;border-radius:6px}
p{word-wrap:break-word}a{color:#1459a7}h3{margin-bottom:4px}</style></head><body>
<header><h1>Caption to Head - Original Talk2DINO</h1><p>CLS-to-patch head attention, not a semantic segmentation mask.</p></header>
<main>""" + "\n".join(blocks) + "</main></body></html>"
    (Path(output) / 'index.html').write_text(template, encoding='utf-8')

def _fingerprint(config):
    # Exclude absolute output path and selected IDs so incremental --image-ids
    # can safely reuse any previously completed sample from the same experiment.
    keys = ['features', 'weights_sha256', 'config', 'image_root', 'image_size', 'image_model',
            'text_source', 'feature_source', 'attention_norm', 'attention_alpha', 'min_cache_cosine',
            'all_heads_grid', 'strict_cache', 'dinov2_repo', 'hub_source']
    return hashlib.sha256(json.dumps({k: config[k] for k in keys}, sort_keys=True).encode()).hexdigest()


if __name__ == '__main__':
    sys.exit(main())
