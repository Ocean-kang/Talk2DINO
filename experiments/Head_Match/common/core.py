"""Pure, testable decisions for head-conditioned semantic refinement.

Scores are embedding proxies, NOT calibrated object-presence probabilities.
"""
from __future__ import annotations
import math
import re
import torch
import torch.nn.functional as F

IRREGULAR = {'men':'man','women':'woman','people':'person','children':'child',
             'mice':'mouse','feet':'foot','teeth':'tooth'}


def words(s):
    out=[]
    for x in re.findall(r'[a-z0-9]+',str(s).lower()):
        x=IRREGULAR.get(x,x)
        if len(x)>4 and x.endswith('ies'): x=x[:-3]+'y'
        elif len(x)>4 and x.endswith(('sses','shes','ches','xes')): x=x[:-2]
        elif len(x)>3 and x.endswith('s') and not x.endswith(('ss','us','is','ous')): x=x[:-1]
        out.append(x)
    return tuple(out)


def mentioned(concept, caption):
    c=words(concept); w=words(caption)
    return not c or any(w[i:i+len(c)]==c for i in range(len(w)-len(c)+1))


def clean_vocabulary(names):
    seen=set(); result=[]
    for name in names:
        if not isinstance(name,str): continue
        name=name.strip()
        key=words(name)
        if key and key not in seen:
            result.append(name);seen.add(key)
    return result


def unit(x): return F.normalize(torch.as_tensor(x).float(),dim=-1)


def assignments(heads, caption_projected, temperature=.07, mode='soft', top_heads=4):
    """Caption-conditioned head distribution; shape H, sum=1."""
    sim=unit(heads) @ unit(caption_projected).flatten()
    if mode=='argmax':
        return F.one_hot(sim.argmax(),num_classes=sim.numel()).float(),sim
    if temperature<=0:raise ValueError('temperature must be positive')
    top_heads=min(max(1,int(top_heads)),sim.numel())
    idx=torch.topk(sim,top_heads).indices
    logits=torch.full_like(sim,-torch.inf)
    logits[idx]=sim[idx]/temperature
    return logits.softmax(0),sim


def scored_concepts(heads, head_distribution, concept_projected, caption_clip, concept_clip,
                    candidates, caption="", coverage_threshold=.60, covered_sim=.37, min_visual=.24,
                    novelty_max=.42, max_concepts=2):
    """Head support weighted by soft head assignment and lexical/CLIP novelty.

    Coverage is relative to the supplied candidate gallery, not ground truth.
    """
    if not candidates:return {'coverage':1.,'ranked':[],'selected':[],'gate':'empty_candidates'}
    h=unit(heads)
    z=unit(concept_projected)
    visual=h @ z.T
    support=(head_distribution @ visual).float()
    caption_vec=unit(caption_clip).flatten()
    novelty_sim=unit(concept_clip) @ caption_vec
    # Don't infer absence merely from an embedding score; lexical coverage is a separate check.
    top_n=min(max(3,max_concepts),len(candidates))
    top_idx=support.topk(top_n).indices.tolist()
    positive=(support[top_idx]/.07).softmax(dim=0)  # relative proxy
    covered_flags=[mentioned(candidates[i],caption) or bool(novelty_sim[i]>=covered_sim) for i in top_idx]
    coverage=float(sum(float(w) for w,flag in zip(positive.tolist(),covered_flags) if flag))
    rows=[]
    for idx in top_idx:
        c=candidates[idx]
        s=float(support[idx]); t=float(novelty_sim[idx])
        rows.append({'concept':c,'head_support':round(s,6),'caption_concept_sim':round(t,6),
                     'best_head':int(visual[:,idx].argmax()),
                     'head_score':round(float(visual[:,idx].max()),6),
                     'novel':bool(t<novelty_max and not mentioned(c,caption)),
                     'rank_score':round(s + .15*max(0.,novelty_max-t),6)})
    rows.sort(key=lambda r:(-r['rank_score'],r['concept']))
    selected=[r for r in rows if r['head_support']>=min_visual and r['novel']][:max_concepts]
    if coverage>=coverage_threshold:selected=[]
    return {'coverage':round(coverage,6),'ranked':rows,'selected':selected,
            'gate':'covered' if coverage>=coverage_threshold else 'missing'}


def compose(caption, concepts):
    """Conservative additive text: does not hallucinate object relations."""
    if not concepts:return caption
    s=caption.rstrip().rstrip('.!?')
    return s+'. Also visible: '+', '.join(concepts)+'.'


def consensus(proposals, threshold, n_required):
    """A concept gets one vote per teacher, not per synonym or head."""
    tally={}
    for teacher_scores in proposals:
        for key,value in teacher_scores.items():
            if float(value)>=threshold:
                tally[key]=tally.get(key,0)+1
    return {k:v for k,v in tally.items() if v>=n_required}


def shift(p,q):
    """Total variation of two head assignments in [0,1]."""
    return float(.5*torch.abs(p-q).sum())
