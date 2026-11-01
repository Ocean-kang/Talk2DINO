"""Strict local model/cache provenance checks; no model-name based downloads."""
from pathlib import Path
import runpy
from .io import sha256, read_json


def offline():
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'offline_guard/sitecustomize.py'))


def model_metadata(clip_checkpoint, dino_checkpoint):
    for path in (clip_checkpoint, dino_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError('Supply an existing local checkpoint: ' + str(path))
    return {'clip_model': 'ViT-B/16', 'clip_checkpoint_sha256': sha256(clip_checkpoint),
            'dino_model': 'dinov2_vitb14_reg', 'dino_checkpoint_sha256': sha256(dino_checkpoint),
            'resize': 448, 'crop': 448, 'head_field': 'disentangled_self_attn',
            'head_shape': [12, 768], 'validation_field': 'avg_self_attn_out', 'clip_dim': 512,
            'raw_caption_template': '{}', 'raw_text_tokenizer_truncate': True}


def check_sources(meta_path, expected, train_path, val_path):
    meta = read_json(meta_path)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('Source provenance mismatch: ' + key)
    for key, path in [('train_features_sha256', train_path), ('val_features_sha256', val_path)]:
        if meta.get(key) != sha256(path):
            raise ValueError('Feature file changed or has wrong metadata: ' + key)
    if not meta.get('source_model_attestation'):
        raise ValueError('Feature models must be verified against existing extraction logs')
    return meta


def validate_bank(bank, expected, gallery, full_gallery=True):
    import torch
    if not isinstance(bank, dict) or not {'texts', 'embeddings', 'metadata'} <= bank.keys():
        raise ValueError('Cache needs texts, embeddings, metadata; unverifiable cache cannot be reused')
    texts, embeds, meta = bank['texts'], bank['embeddings'], bank['metadata']
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts) or len(texts) != len(set(texts)):
        raise ValueError('Cache text order must be explicit and unique')
    if not torch.is_tensor(embeds) or tuple(embeds.shape) != (len(texts), 512) or not torch.isfinite(embeds).all():
        raise ValueError('Invalid CLIP cache dimensions or values')
    if bool((embeds.float().norm(dim=-1) == 0).any()):
        raise ValueError('Zero text embedding')
    for key in ('clip_model', 'clip_checkpoint_sha256'):
        if meta.get(key) != expected[key]:
            raise ValueError('CLIP cache provenance mismatch: ' + key)
    if meta.get('prompt_template') != 'a photo of {}' or meta.get('feature_space') != 'clip_text_raw':
        raise ValueError('Wrong cache prompt template or feature space (projected vectors cannot be reused)')
    if full_gallery and meta.get('gallery_concepts') != gallery:
        raise ValueError('Gallery concept order mismatch; re-encode locally or provide matching cache')
    if meta.get('texts_sha256') != text_digest(texts):
        raise ValueError('Cache text-order hash mismatch')
    return bank


def text_digest(texts):
    import json
    import hashlib
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode('utf-8')).hexdigest()
