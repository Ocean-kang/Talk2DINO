"""Pure matching/diagnostic functions, independent of the COCO data loader."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def head_pool(patch_tokens: torch.Tensor, cls_patch_logits: torch.Tensor):
    """Reproduce dino_extraction_v2.py's `disentangled_self_attn`.

    patch_tokens: [B,P,D], cls_patch_logits: [B,H,P] (pre-softmax).
    Note the original paper/code divides by P AFTER the attention softmax.
    """
    if patch_tokens.ndim != 3 or cls_patch_logits.ndim != 3:
        raise ValueError('Expected [B,P,D] patches and [B,H,P] logits')
    if patch_tokens.shape[0] != cls_patch_logits.shape[0] or patch_tokens.shape[1] != cls_patch_logits.shape[2]:
        raise ValueError('Patch-token count differs from attention spatial size')
    weights = cls_patch_logits.softmax(dim=-1)
    pooled = (patch_tokens.unsqueeze(1) * weights.unsqueeze(-1)).mean(dim=2)
    return pooled, weights


@torch.no_grad()
def compute_matches(projector, caption_features, heads, *, device='cpu'):
    """Match EVERY COCO caption to the head with maximum cosine similarity.

    Text features [C,512], visual heads [H,768]. Return raw cosine matrix [C,H].
    Follows src.model.ProjectionLayer.forward(project_clip_txt + L2 normalize).
    """
    text = torch.as_tensor(caption_features, device=device, dtype=torch.float32)
    visual = torch.as_tensor(heads, device=device, dtype=torch.float32)
    if text.ndim != 2 or visual.ndim != 2:
        raise ValueError('Text features and head features must both be matrices')
    if not text.shape[0] or not visual.shape[0]:
        raise ValueError('Empty caption/head matrix')
    if not torch.isfinite(text).all() or not torch.isfinite(visual).all():
        raise ValueError('Non-finite caption/head features')
    projected = projector.project_clip_txt(text)
    if projected.shape[1] != visual.shape[1]:
        raise ValueError(f'Projection dimension {projected.shape[1]} differs from head {visual.shape[1]}')
    sims = F.normalize(projected.float(), p=2, dim=-1) @ F.normalize(visual.float(), p=2, dim=-1).T
    indices = sims.argmax(dim=-1)
    k = min(2, sims.shape[-1])
    best = sims.topk(k, dim=-1)
    margin = (best.values[:, 0] - best.values[:, 1]) if k == 2 else torch.zeros_like(best.values[:, 0])
    return {
        'similarities': sims.cpu(),
        'best_heads': indices.cpu(),  # zero-indexed, e.g. 0..11
        'top1_scores': best.values[:, 0].cpu(),
        'top2_scores': best.values[:, 1].cpu() if k == 2 else best.values[:, 0].cpu(),
        'top2_heads': best.indices[:, 1].cpu() if k == 2 else best.indices[:, 0].cpu(),
        'margins': margin.cpu(),
    }


def attention_concentration(attentions: torch.Tensor, fraction=0.10):
    """Top fraction's mass / expected mass for uniform attention (uniform => ~1)."""
    arr = torch.as_tensor(attentions).float().flatten(start_dim=1)
    if arr.ndim != 2 or not torch.isfinite(arr).all():
        raise ValueError('Attention maps must be finite [H,P]')
    a = arr / arr.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    n = max(1, math.ceil(a.shape[-1] * fraction))
    effective_fraction = n / a.shape[-1]
    return (a.topk(n, dim=-1).values.sum(dim=-1) / effective_fraction).tolist()


def describe_matches(scores: dict, captions: list, annotation_ids: list, attentions: torch.Tensor):
    if len(captions) != scores['similarities'].shape[0] or len(captions) != len(annotation_ids):
        raise ValueError('Caption/feature/ID count mismatch')
    best = scores['best_heads'].tolist()
    num_heads = scores['similarities'].shape[1]
    uses = [best.count(h) for h in range(num_heads)]
    unique = sum(n > 0 for n in uses)
    rows = []
    for i, caption in enumerate(captions):
        rows.append({
            'caption_index': i,
            'annotation_id': int(annotation_ids[i]) if annotation_ids[i] is not None else None,
            'caption': str(caption),
            'head_index': int(best[i]),  # 0-based index used to slice attention maps
            'head_number': int(best[i]) + 1,  # friendly 1-based display
            'cosine_similarity': round(float(scores['top1_scores'][i]), 8),
            'second_head_index': int(scores['top2_heads'][i]),
            'second_best_cosine': round(float(scores['top2_scores'][i]), 8),
            'top1_top2_margin': round(float(scores['margins'][i]), 8),
            'all_head_cosines': [round(float(n), 8) for n in scores['similarities'][i].tolist()],
        })
    return {
        'num_captions': len(captions),
        'num_heads': num_heads,
        'head_use_counts': uses,
        'unique_heads_used': unique,
        'unused_head_numbers': [h + 1 for h, n in enumerate(uses) if n == 0],
        'mean_matching_margin': round(sum(float(x) for x in scores['margins']) / len(captions), 8),
        'head_attention_concentration_top10pct': [round(float(x), 6) for x in attention_concentration(attentions)],
        'captions': rows,
    }
