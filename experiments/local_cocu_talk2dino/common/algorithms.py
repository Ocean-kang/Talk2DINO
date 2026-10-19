"""Scoring functions separated from IO/GPU, to allow deterministic unit tests."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .data import novel


def head_score(heads, embeddings):
    h = F.normalize(torch.as_tensor(heads).float(), dim=-1)
    z = F.normalize(torch.as_tensor(embeddings,device=h.device).float(), dim=-1)
    sim = h @ z.T
    return sim.max(dim=0).values, sim.argmax(dim=0), sim


def top_concepts(words, heads, embeddings, captions, k=3, threshold=None, per_head_k=10, permitted_heads=None):
    """Head-wise top-k union, then one global score per concept, with lexical novelty."""
    scores, best, sim = head_score(heads, embeddings)
    permitted_heads = list(range(sim.shape[0])) if permitted_heads is None else list(permitted_heads)
    if not permitted_heads: return []
    eligible = set()
    for h in permitted_heads:
        eligible.update(sim[h].topk(min(per_head_k, len(words))).indices.tolist())
    accepted = novel((words[j] for j in sorted(eligible)), captions)
    index = {word:j for j,word in enumerate(words)}
    rows=[]
    for name in accepted:
        idx=index[name]
        # Score MUST be measured on the selected head set, not uncovered head-ineligible heads.
        sub = sim[permitted_heads,idx]
        val, arg = sub.max(dim=0)
        if threshold is not None and val.item()<threshold: continue
        rows.append({'concept':name, 'score':float(val.item()), 'best_head':int(permitted_heads[arg.item()])})
    return sorted(rows,key=lambda x:(-x['score'],x['concept']))[:k]


def uncovered_heads(heads, original_embeddings, n):
    q = head_score(heads, original_embeddings)[2].max(dim=1).values
    order = torch.argsort(q)[:n].tolist()
    return order, [float(v) for v in q.tolist()]


def consistency_score(target, supports, min_support=2, gap=0.05, rare_high=0.55):
    if not supports:
        return float(target), 0, 0.0, 0.0
    s = torch.as_tensor(supports).float()
    count = int((s >= target-gap).sum())
    # Retain rare, image-specific high-certainty concepts instead of hard-deleting all.
    accepted = count>=min_support or target>=rare_high
    score = 0.7*target + 0.3*float(s.mean()) - 0.1*float(s.std(unbiased=False))
    return (float(score) if accepted else -1e9),count,float(s.mean()),float(s.std(unbiased=False))


def counterfactual_score(target, negatives, lam=0.3):
    return target-lam*(sum(negatives)/len(negatives)) if negatives else target


def view_agreement(values, min_views=2, threshold=None, variability=0.2):
    values = list(map(float,values))
    if not values: return None
    threshold = min(values) if threshold is None else threshold
    if sum(x>=threshold for x in values)<min_views: return None
    t = torch.as_tensor(values)
    return float(t.mean()-variability*t.std(unbiased=False))


def bbox_from_map(attn, quantile=0.9, expand=0.15, pixels=448):
    """Attentive patch bounding rectangle in the 448x448 center-crop coordinate system."""
    a = torch.as_tensor(attn).float().reshape(-1)
    g = int(a.numel()**0.5)
    if g*g!=a.numel(): raise ValueError('Attention must be square patch grid')
    a = a.reshape(g,g)
    mask = a>=torch.quantile(a,quantile)
    ys,xs = torch.where(mask)
    if not len(xs): return (0,0,pixels,pixels)
    x0, x1 = xs.min().item(), xs.max().item()+1
    y0, y1 = ys.min().item(), ys.max().item()+1
    w,h = x1-x0,y1-y0
    d = max(w,h)*expand
    x0=max(0,int((x0-d)*pixels/g)); y0=max(0,int((y0-d)*pixels/g))
    x1=min(pixels,int((x1+d)*pixels/g+1)); y1=min(pixels,int((y1+d)*pixels/g+1))
    return x0,y0,x1,y1
