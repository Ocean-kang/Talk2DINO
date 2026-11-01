"""Wrap a legacy raw CLIP cache only when its existing provenance is known.
Never use this to relabel bare-concept or Teacher-projected vectors.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.io import read_json, sha256
from common.nouns import load_gallery
from common.cache import offline, text_digest, validate_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--tensor-key', help='Required if legacy cache is a dictionary; no key is guessed')
    p.add_argument('--texts', required=True, help='Existing ordered JSON list of EXACT encoded texts')
    p.add_argument('--gallery', required=True)
    p.add_argument('--clip-checkpoint', required=True)
    p.add_argument('--source-template', required=True)
    p.add_argument('--confirm-raw-clip-provenance', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if not a.confirm_raw_clip_provenance or a.source_template != 'a photo of {}':
        p.error('Requires verified raw CLIP B/16 embeddings with EXACT source template a photo of {}; otherwise re-encode locally')
    if not Path(a.clip_checkpoint).is_file():
        raise FileNotFoundError(a.clip_checkpoint)
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    offline()
    import torch
    g, _ = load_gallery(a.gallery)
    texts = read_json(a.texts)
    # Existing gallery cache may contain additional raw caption texts, but the
    # full gallery prompt order must be explicit in the supplied texts list.
    gallery_prompts = ['a photo of ' + c for c in g]
    if [t for t in texts if t in set(gallery_prompts)] != gallery_prompts:
        raise ValueError('Existing cache full-gallery concept order differs from the supplied gallery')
    source = torch.load(a.cache, map_location='cpu')
    if isinstance(source, dict):
        if not a.tensor_key:
            p.error('Provide the actual --tensor-key for your legacy cache')
        source = source[a.tensor_key]
    expected = {'clip_model': 'ViT-B/16', 'clip_checkpoint_sha256': sha256(a.clip_checkpoint)}
    meta = dict(expected, prompt_template='a photo of {}', feature_space='clip_text_raw',
        gallery_concepts=g, texts_sha256=text_digest(texts), legacy_cache_sha256=sha256(a.cache),
        verification='user-attested from existing cache generation records')
    bank = {'texts': texts, 'embeddings': source, 'metadata': meta}
    validate_bank(bank, expected, g)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, a.output)
    print(a.output)


if __name__ == '__main__':
    main()
