"""Existing gallery -> caption occurrence pools -> local frozen CLIP text cache."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.io import load_features, manifest, sha256, write_json
from common.nouns import GalleryMatcher, load_gallery, compose_prompt
from common.cache import offline, model_metadata, check_sources, validate_bank, text_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-features', required=True)
    p.add_argument('--val-features', required=True)
    p.add_argument('--gallery', required=True, help='Existing MSCOCO14 gallery; never built from selected images')
    p.add_argument('--clip-checkpoint', required=True, help='Existing local CLIP ViT-B/16 checkpoint FILE')
    p.add_argument('--dino-checkpoint', required=True, help='Existing local DINO checkpoint, provenance check only; never loaded')
    p.add_argument('--feature-metadata', required=True)
    p.add_argument('--text-cache', help='Optional verified CLIP cache: texts/embeddings/metadata')
    p.add_argument('--morphology', choices=['exact', 'repo_forms'], default='exact')
    p.add_argument('--output', default='outputs/minimal_noun_supervision/data')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--subset-seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--empty-policy', choices=['exclude', 'error'], default='exclude')
    args = p.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        p.error('limit must be nonnegative; batch-size must be positive')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh --output: ' + str(out))
    offline()
    import torch
    import clip
    expected = model_metadata(args.clip_checkpoint, args.dino_checkpoint)
    source_meta = check_sources(args.feature_metadata, expected, args.train_features, args.val_features)
    gallery, aliases = load_gallery(args.gallery)
    matcher = GalleryMatcher(gallery, args.morphology)
    train, val = load_features(args.train_features), load_features(args.val_features)
    train_ids = {int(x['id']) for x in train['images']}
    val_ids = {int(x['id']) for x in val['images']}
    if train_ids & val_ids:
        raise ValueError('Train/val image overlap')
    all_caps = defaultdict(list)
    for ann in train['annotations']:
        all_caps[int(ann['image_id'])].append(ann['caption'])
    if train_ids != set(all_caps):
        raise ValueError('Every train image needs original captions')
    chosen = sorted(train_ids)
    if args.limit and args.limit < len(chosen):
        chosen = sorted(random.Random(args.subset_seed).sample(chosen, args.limit))
    captions = {iid: all_caps[iid] for iid in chosen}
    invalid_caption_ids = [iid for iid in chosen if not captions[iid] or
        any(not isinstance(t, str) or not t.strip() for t in captions[iid])]
    if invalid_caption_ids:
        raise ValueError('Expected one or more nonempty ORIGINAL captions per selected train image; '
                         'invalid image IDs (first 20): ' + str(invalid_caption_ids[:20]))
    # COCO images can have more than five original annotations. Keep all of
    # them for concept matching and the Raw pool; never truncate to five.
    caption_count_distribution = dict(sorted(Counter(len(captions[i]) for i in chosen).items()))
    # Preserve previous subset and its annotation-based optimizer budget, even if
    # a few images are jointly excluded for no gallery match.
    budget_raw_count = sum(len(captions[i]) for i in chosen)
    concepts = {iid: matcher.extract(captions[iid]) for iid in chosen}
    excluded = [iid for iid in chosen if concepts[iid]['no_match']]
    keep = [iid for iid in chosen if not concepts[iid]['no_match']]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'shared/matching_audit.json', {
        'mode': args.morphology, 'requested_image_ids': chosen, 'retained_image_ids': keep,
        'exclusions': [{'image_id': i, 'reason': 'no_gallery_concept_in_original_captions',
                        'raw_captions': captions[i]} for i in excluded],
        'excluded_count': len(excluded), 'excluded_ratio': len(excluded) / len(chosen),
        'normalized_gallery_aliases': aliases, 'gallery_source': str(Path(args.gallery).resolve()),
        'gallery_source_sha256': sha256(args.gallery), 'original_budget_annotation_count': budget_raw_count,
        'retained_annotation_count': sum(len(captions[i]) for i in keep),
        'caption_count_distribution': caption_count_distribution,
        'caption_count_by_image': {str(i): len(captions[i]) for i in chosen}})
    if not keep or (excluded and args.empty_policy == 'error'):
        raise ValueError('No-match exclusions recorded in shared/matching_audit.json; no training manifests written')
    images = {int(x['id']): x for x in train['images']}
    for iid in keep:
        x = images[iid].get('disentangled_self_attn')
        if x is None or tuple(x.shape) != (12, 768) or not torch.isfinite(x).all() or bool((x.float().norm(dim=-1) == 0).any()):
            raise ValueError('Expected finite nonzero ViT-B heads [12,768], image %s' % iid)
    for im in val['images']:
        x = im.get('avg_self_attn_out')
        if x is None or tuple(x.shape) != (768,) or not torch.isfinite(x).all():
            raise ValueError('Validation requires original avg_self_attn_out [768]')
    def fits(text):
        try:
            clip.tokenize([text], truncate=False)
            return True
        except RuntimeError:
            return False
    # Do not silently shrink the gallery to concepts that occur in this subset.
    gallery_prompts = ['a photo of ' + c for c in gallery]
    for c, t in zip(gallery, gallery_prompts):
        if not fits(t):
            raise ValueError('Full-gallery concept exceeds CLIP context; no concept silently dropped: ' + c)
    pools = {m: [] for m in ('raw', 'cne', 'cnp')}
    composition = []
    for iid in keep:
        cs = concepts[iid]['concepts']
        compound, kept, removed = compose_prompt(cs, fits)
        pools['raw'].append({'image_id': iid, 'prompt_pool': captions[iid]})
        pools['cne'].append({'image_id': iid, 'prompt_pool': ['a photo of ' + c for c in cs]})
        pools['cnp'].append({'image_id': iid, 'prompt_pool': [compound]})
        composition.append({'image_id': iid, 'input_concepts': cs, 'kept': kept, 'removed': removed,
            'rule': 'caption_frequency_desc_then_alphabetical; greedy_whole_concept_skip_and_continue',
            'token_count': int((clip.tokenize([compound], truncate=False) != 0).sum())})
    val_caps = defaultdict(list)
    for ann in val['annotations']:
        val_caps[int(ann['image_id'])].append(ann['caption'])
    if val_ids != set(val_caps) or any(not val_caps[i][0].strip() for i in val_ids):
        raise ValueError('Every validation image needs a nonempty original caption')
    val_records = [{'image_id': iid, 'prompt_pool': [val_caps[iid][0]]} for iid in sorted(val_ids)]
    texts = sorted(set(gallery_prompts) | {t for recs in pools.values() for r in recs for t in r['prompt_pool']} |
                   {r['prompt_pool'][0] for r in val_records})
    lookup = {t: i for i, t in enumerate(texts)}
    embeds = torch.empty((len(texts), 512), dtype=torch.float32)
    filled = set()
    if args.text_cache:
        cache = validate_bank(torch.load(args.text_cache, map_location='cpu'), expected, gallery)
        for t, e in zip(cache['texts'], cache['embeddings']):
            if t in lookup:
                embeds[lookup[t]] = e.float()
                filled.add(t)
    cached_count = len(filled)
    # Reuse trusted original CLIP features only after model/file hashes match.
    for source, ids in ((train, set(keep)), (val, val_ids)):
        for ann in source['annotations']:
            t = ann['caption']
            if int(ann['image_id']) in ids and t in lookup and t not in filled and 'ann_feats' in ann:
                f = ann['ann_feats'].float()
                if tuple(f.shape) != (512,) or not torch.isfinite(f).all() or float(f.norm()) == 0:
                    raise ValueError('Invalid original CLIP ann_feats')
                embeds[lookup[t]] = f
                filled.add(t)
    missing = [t for t in texts if t not in filled]
    if missing:
        # FILE path, never a model name: clip.load cannot trigger a weight download.
        encoder, _ = clip.load(str(Path(args.clip_checkpoint).resolve()), device=args.device, jit=False)
        if int(encoder.text_projection.shape[-1]) != 512 or int(encoder.visual.conv1.weight.shape[-1]) != 16:
            raise ValueError('Expected existing CLIP ViT-B/16 checkpoint')
        encoder.eval().requires_grad_(False)
        from tqdm import tqdm
        for start in tqdm(range(0, len(missing), args.batch_size), desc='Local CLIP prompt features'):
            batch = missing[start:start + args.batch_size]
            with torch.no_grad():
                values = encoder.encode_text(clip.tokenize(batch, truncate=True).to(args.device)).float().cpu()
            embeds[[lookup[t] for t in batch]] = values
        del encoder
    bank_path = out / 'shared/text_bank.pth'
    metadata = dict(expected, prompt_template='a photo of {}', feature_space='clip_text_raw',
                    gallery_concepts=gallery, texts_sha256=text_digest(texts),
                    gallery_source_sha256=sha256(args.gallery), source_metadata=source_meta)
    bank = {'texts': texts, 'embeddings': embeds, 'metadata': metadata, 'clip_model': 'ViT-B/16'}
    validate_bank(bank, expected, gallery)
    torch.save(bank, bank_path)
    for method in pools:
        write_json(out / method / 'manifest.json', manifest(method, args.train_features, pools[method], budget_raw_count, bank_path))
    write_json(out / 'shared/validation.json', manifest('raw_fixed_validation', args.val_features, val_records, len(val['annotations']), bank_path))
    write_json(out / 'shared/concepts_by_image.json', {str(i): concepts[i] for i in chosen})
    # Ordered normalized copy of the existing FULL gallery, never a new gallery.
    write_json(out / 'shared/gallery.json', gallery)
    write_json(out / 'cnp/composition_audit.json', composition)
    diagnostics = {}
    for method, recs in pools.items():
        freq = Counter(t for r in recs for t in set(r['prompt_pool']))
        diagnostics[method] = {'images': len(recs), 'mean_pool_size': sum(len(r['prompt_pool']) for r in recs) / len(recs),
            'unique_prompts': len(freq), 'prompts_shared_across_images': sum(v > 1 for v in freq.values()),
            'ordered_image_pair_prompt_collisions': sum(v * (v - 1) for v in freq.values())}
    write_json(out / 'shared/statistics.json', {'arms': diagnostics, 'requested_images': len(chosen),
        'caption_count_distribution': caption_count_distribution,
        'original_budget_annotation_count': budget_raw_count,
        'retained_annotation_count': sum(len(captions[i]) for i in keep),
        'retained_images': len(keep), 'no_match_images': len(excluded), 'no_match_ratio': len(excluded) / len(chosen),
        'cnp_images_trimmed': sum(bool(r['removed']) for r in composition), 'gallery_size': len(gallery),
        'cnp_concept_retention': sum(len(r['kept']) for r in composition) / sum(len(concepts[i]['concepts']) for i in keep)})
    write_json(out / 'shared/provenance.json', {'arguments': vars(args), 'source_metadata': source_meta,
        'train_source_sha256': sha256(args.train_features), 'val_source_sha256': sha256(args.val_features),
        'gallery_sha256': sha256(args.gallery), 'text_bank_sha256': sha256(bank_path),
        'torch_version': torch.__version__, 'requested_image_ids': chosen, 'train_image_ids': keep,
        'val_image_ids': sorted(val_ids), 'raw_annotation_count': budget_raw_count,
        'reused_cached_texts': cached_count, 'reused_raw_texts': len(filled) - cached_count, 'encoded_new_texts': len(missing),
        'reused_helpers_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / 'experiments/local_cocu/utils/concept_utils.py', ROOT / 'experiments/local_cocu_novel/filter_novel.py',
            *([ROOT / 'experiments/Head_caption_S5_Editing/editing.py'] if args.morphology == 'repo_forms' else []))},
        'offline': True})
    print(json.dumps({'output': str(out), 'images': len(keep), 'excluded': len(excluded), 'full_gallery': len(gallery)}))


if __name__ == '__main__':
    main()
