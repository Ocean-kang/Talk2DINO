"""Chunked exact gallery search and explicitly audited caption intersection."""
import torch
import torch.nn.functional as F


def topk_gallery(queries, gallery, k=20, chunk_size=4096):
    if gallery.ndim != 2 or queries.shape[-1] != gallery.shape[-1] or len(gallery) < 2:
        raise ValueError('Need compatible vectors and a gallery of at least two concepts')
    k = min(k, len(gallery))
    best_values = queries.new_empty((len(queries), 0))
    best_ids = torch.empty((len(queries), 0), device=queries.device, dtype=torch.long)
    for start in range(0, len(gallery), chunk_size):
        scores = queries @ gallery[start:start + chunk_size].T
        ids = torch.arange(start, start + scores.shape[1], device=queries.device).expand(len(queries), -1)
        values = torch.cat([best_values, scores], dim=1)
        indices = torch.cat([best_ids, ids], dim=1)
        pos = torch.argsort(values, dim=1, descending=True, stable=True)[:, :min(k, values.shape[1])]
        best_values = values.gather(1, pos)
        best_ids = indices.gather(1, pos)
    return best_values, best_ids


def caption_candidates(heads, gallery, top_values, top_ids, allowed):
    """Margins are diagnostics; rank>1 gallery gap can legitimately be negative."""
    candidates = []
    for h in range(heads.shape[0]):
        for r, gid in enumerate(top_ids[h].tolist()):
            if gid not in allowed:
                continue
            score = float(top_values[h, r])
            by_head = heads @ gallery[gid]
            other = torch.cat([by_head[:h], by_head[h + 1:]])
            top_competitor = float(top_values[h, 1 if r == 0 else 0])
            candidates.append({'gallery_id': gid, 'head': h, 'gallery_rank': r + 1,
                'score': score, 'head_margin': score - float(other.max()),
                'gallery_gap': score - top_competitor,
                'head_top1_top2_gap': float(top_values[h, 0] - top_values[h, 1])})
    return candidates


def choose_concepts(candidates, threshold, min_head_margin=0.0, min_gallery_gap=None, cap=5):
    per_head = {}
    for c in sorted(candidates, key=lambda c: (-c['score'], c['gallery_id'], c['head'])):
        c['confidence_pass'] = False
        if c['score'] < threshold:
            c['selection_reason'] = 'below_similarity_threshold'
        elif c['head_margin'] < min_head_margin:
            c['selection_reason'] = 'below_head_margin'
        elif min_gallery_gap is not None and c['gallery_gap'] < min_gallery_gap:
            c['selection_reason'] = 'below_gallery_gap'
        else:
            c['confidence_pass'] = True
            if c['head'] in per_head:
                c['selection_reason'] = 'head_already_assigned_to_higher_score_candidate'
            else:
                per_head[c['head']] = c
    distinct = {}
    for c in sorted(per_head.values(), key=lambda c: (-c['score'], c['gallery_id'])):
        if c['gallery_id'] in distinct:
            c['selection_reason'] = 'same_concept_already_selected_for_other_head'
        elif len(distinct) >= cap:
            c['selection_reason'] = 'image_concept_cap'
        else:
            distinct[c['gallery_id']] = c
            c['selection_reason'] = 'selected_confidence_pass'
    return list(distinct.values())


def exact_gallery_rank(query, gallery, gid, chunk_size=4096):
    """1-based exact full-gallery fallback rank; ties use smaller gallery ID."""
    block_start = (gid // chunk_size) * chunk_size
    score = (query @ gallery[block_start:block_start + chunk_size].T)[gid - block_start]
    ahead = 0
    for start in range(0, len(gallery), chunk_size):
        values = query @ gallery[start:start + chunk_size].T
        ids = torch.arange(start, start + len(values), device=query.device)
        ahead += int(((values > score) | ((values == score) & (ids < gid))).sum())
    return 1 + ahead
