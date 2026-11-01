"""Frozen Talk2DINO teacher -> full existing gallery top-K -> caption intersection."""
import argparse
import json
from pathlib import Path
import random
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.io import load_features, read_json, sha256, write_json
from common.selection import topk_gallery, caption_candidates, choose_concepts, exact_gallery_rank
from common.cache import offline, validate_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='outputs/minimal_noun_supervision/data')
    p.add_argument('--teacher', default='weights/vitb_mlp_infonce.pth')
    p.add_argument('--model-config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--device', default='cuda')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--gallery-chunk', type=int, default=4096)
    p.add_argument('--image-batch', type=int, default=16)
    p.add_argument('--max-concepts', type=int, default=5)
    p.add_argument('--min-sim', type=float, default=None, help='Override train-only quantile calibration')
    p.add_argument('--calibration-quantile', type=float, default=0.75)
    p.add_argument('--calibration-images', type=int, default=2048)
    p.add_argument('--min-head-margin', type=float, default=0.0)
    p.add_argument('--min-gallery-gap', type=float, default=None)
    args = p.parse_args()
    if not 0 <= args.calibration_quantile <= 1 or min(args.top_k, args.gallery_chunk, args.image_batch, args.max_concepts, args.calibration_images) <= 0:
        p.error('Invalid positive size or calibration quantile')
    if args.top_k < 2:
        p.error('top-k >= 2 is required for gallery-margin diagnostics')
    offline()
    import torch
    import torch.nn.functional as F
    import yaml
    from src.model import ProjectionLayer
    from tqdm import tqdm
    root = Path(args.data).resolve()
    dest = root / 'hvne'
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('HVNE output exists; select into a fresh data directory')
    base = read_json(root / 'cne/manifest.json')
    gallery_names = read_json(root / 'shared/gallery.json')
    concept_map = read_json(root / 'shared/concepts_by_image.json')
    provenance = read_json(root / 'shared/provenance.json')
    if sha256(base['source_features']) != provenance['train_source_sha256']:
        raise ValueError('Train feature source changed after preparation')
    if sha256(base['text_bank']) != provenance['text_bank_sha256']:
        raise ValueError('Text bank changed after preparation')
    bank = validate_bank(torch.load(base['text_bank'], map_location='cpu'),
                         provenance['source_metadata'], gallery_names)
    lookup = {t: i for i, t in enumerate(bank['texts'])}
    cfg = yaml.safe_load(Path(args.model_config).read_text())['model']
    if cfg.get('dino_embed_dim', 1024) != 768 or cfg.get('clip_embed_dim', 512) != 512 or cfg.get('weight_attn_heads') is not None:
        raise ValueError('Teacher must use the audited ViT-B 512-to-768 ProjectionLayer')
    if cfg.get('starting_checkpoint') is not None:
        raise ValueError('Use a config without starting_checkpoint; --teacher is explicit')
    teacher = ProjectionLayer.from_config(cfg)
    state = torch.load(args.teacher, map_location='cpu')
    if 'state_dict' in state: state = state['state_dict']
    elif 'model' in state: state = state['model']
    state = {k.removeprefix('module.').removeprefix('projector.'): v for k, v in state.items()}
    teacher.load_state_dict(state, strict=True)
    teacher = teacher.to(args.device).eval().requires_grad_(False)
    with torch.no_grad():
        projected = []
        ids = [lookup['a photo of ' + c] for c in gallery_names]
        for start in range(0, len(ids), 1024):
            projected.append(F.normalize(teacher.project_clip_txt(bank['embeddings'][ids[start:start + 1024]].to(args.device)), dim=-1))
        gallery = torch.cat(projected)
    raw = load_features(base['source_features'])
    images = {int(x['id']): x for x in raw['images']}
    records = base['records']
    gallery_lookup = {c: i for i, c in enumerate(gallery_names)}
    def process(batch):
        heads = torch.stack([images[int(r['image_id'])]['disentangled_self_attn'] for r in batch]).float().to(args.device)
        heads = F.normalize(heads, dim=-1)
        vals, gids = topk_gallery(heads.flatten(0, 1), gallery, args.top_k, args.gallery_chunk)
        vals = vals.view(len(batch), heads.shape[1], -1)
        gids = gids.view_as(vals).long()
        result = []
        for j, rec in enumerate(batch):
            iid = int(rec['image_id'])
            allowed = {gallery_lookup[c] for c in concept_map[str(iid)]['concepts']}
            candidates = caption_candidates(heads[j], gallery, vals[j], gids[j], allowed)
            result.append((iid, heads[j], allowed, candidates))
        return result
    with torch.no_grad():
        if args.min_sim is None:
            sample = random.Random(123).sample(records, min(args.calibration_images, len(records)))
            scores = []
            for start in tqdm(range(0, len(sample), args.image_batch), desc='Calibrating on train only'):
                for _, _, _, candidates in process(sample[start:start + args.image_batch]):
                    scores.extend(c['score'] for c in candidates)
            threshold = float(torch.quantile(torch.tensor(scores), args.calibration_quantile)) if scores else 1.0001
            # Empty calibration is abstention: force audited fallbacks, certify nothing.
        else:
            threshold, scores = args.min_sim, []
        dest.mkdir(parents=True, exist_ok=True)
        final, strict, fallback_count, total_candidates = [], [], 0, 0
        with (dest / 'selection_audit.jsonl').open('w', encoding='utf-8') as audit:
            for start in tqdm(range(0, len(records), args.image_batch), desc='HVNE full-gallery selection'):
                for iid, heads, allowed, candidates in process(records[start:start + args.image_batch]):
                    total_candidates += len(candidates)
                    selected = choose_concepts(candidates, threshold, args.min_head_margin, args.min_gallery_gap, args.max_concepts)
                    if not allowed:
                        raise ValueError('Empty C_i must have been jointly excluded in prepare.py')
                    fallback = not bool(selected)
                    if fallback:
                        fallback_count += 1
                        allowed_order = sorted(allowed)
                        local = heads @ gallery[allowed_order].T
                        flat = int(local.argmax())
                        h, col = divmod(flat, len(allowed_order))
                        gid = allowed_order[col]
                        selected = [{'gallery_id': gid, 'head': h, 'score': float(local[h, col]),
                                     'gallery_rank': exact_gallery_rank(heads[h], gallery, gid, args.gallery_chunk),
                                     'head_margin': None, 'gallery_gap': None, 'confidence_pass': False,
                                     'selection_reason': 'fallback_caption_best_score_not_high_confidence'}]
                    names = [gallery_names[c['gallery_id']] for c in selected]
                    rec = {'image_id': iid, 'prompt_pool': ['a photo of ' + c for c in names], 'fallback': fallback,
                           'selection_reason': 'fallback_caption_best_score' if fallback else 'full_gallery_topk_intersection_and_confidence'}
                    final.append(rec)
                    if not fallback:
                        strict.append(rec)
                    audit.write(json.dumps({'image_id': iid, 'threshold': threshold, 'fallback': fallback,
                        'retrieval_scope': 'complete_existing_gallery_then_caption_intersection',
                        'gallery_size_scanned_per_head': len(gallery_names), 'top_k': min(args.top_k, len(gallery_names)),
                        'fallback_reason': ('no_full_gallery_topk_intersection' if not candidates else 'no_candidate_passed_confidence') if fallback else None,
                        'selected': [dict(c, concept=gallery_names[c['gallery_id']]) for c in selected],
                        'caption_supported_topk_candidates': [dict(c, concept=gallery_names[c['gallery_id']]) for c in candidates]}) + '\n')
    output = dict(base, method='hvne', records=final)
    write_json(dest / 'manifest.json', output)
    # Diagnostic subset only: a new raw control on identical IDs is needed before training it.
    write_json(dest / 'strict_manifest.json', dict(output, method='hvne_strict_diagnostic_subset', records=strict))
    write_json(dest / 'statistics.json', {'images': len(final), 'strict_images': len(strict),
        'fallback_images': fallback_count, 'fallback_ratio': fallback_count / len(final),
        'mean_pool_size': sum(len(r['prompt_pool']) for r in final) / len(final),
        'caption_supported_topk_candidates': total_candidates, 'threshold': threshold,
        'threshold_calibration_samples': len(scores),
        'calibration_status': 'explicit_min_sim' if args.min_sim is not None else ('train_quantile' if scores else 'no_calibration_hits_all_fallback'),
        'full_gallery_size': len(gallery_names), 'fallback_is_high_confidence': False, 'teacher_sha256': sha256(args.teacher),
        'config_sha256': sha256(args.model_config), 'arguments': vars(args),
        'head_numbering': 'zero-based; no head assignment is imposed on student'})
    print(json.dumps({'output': str(dest), 'threshold': threshold, 'fallback_ratio': fallback_count / len(final)}))


if __name__ == '__main__':
    main()
