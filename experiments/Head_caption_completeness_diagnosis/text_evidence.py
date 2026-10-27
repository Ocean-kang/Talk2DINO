"""Caption -> independent gallery-concept support.

S0/S1 deliberately reproduce legacy whole-sentence CLIP evidence.
S2 uses high precision lexical/curated alias entailment.
S3+ fuse lexical anchors with local phrase CLIP and concept-wise null calibration.
"""
from __future__ import annotations
import re
import torch
import torch.nn.functional as F
from experiments.Head_caption_completeness.core import norm, lexical_gallery

# Directional entailments. E.g., "man" -> person, NOT "person" -> man.
DEFAULT_ALIASES = {
    'person': ['person','people','man','men','woman','women','human','humans','child','children','boy','girl'],
    'bicycle': ['bicycle','bike','bicycles','bikes'],
    'bike': ['bike','bicycle','bikes','bicycles'],
    'car': ['car','automobile','cars','automobiles'],
    'automobile': ['automobile','car','automobiles','cars'],
    'sofa': ['sofa','couch','sofas','couches'],
    'couch': ['couch','sofa','couches','sofas'],
    'motorcycle': ['motorcycle','motorbike','motorcycles','motorbikes'],
    'motorbike': ['motorbike','motorcycle','motorbikes','motorcycles'],
    'television': ['television','tv','televisions','tvs'],
    'tv': ['tv','television','tvs','televisions'],
    'cell phone': ['cell phone','mobile phone','cellphone','smartphone'],
    'mobile phone': ['mobile phone','cell phone','cellphone','smartphone'],
}
STRICT_EQUIVALENCE = [
    ['bicycle','bike'],['car','automobile'],['sofa','couch'],
    ['motorcycle','motorbike'],['television','tv'],['cell phone','mobile phone']
]
NEGATION = {'no', 'without', 'not', 'never', 'neither'}
IGNORE = {'a','an','the','on','in','by','and','of','to','with','at','from','for','is','are','was','were'}


def aliases_for_vocab(vocab, extras=None):
    extras=extras or {}
    aliases={}
    for i,c in enumerate(vocab):
        for name in [c]+DEFAULT_ALIASES.get(c.lower(),[])+extras.get(c,[]):
            key=norm(name)
            if key:aliases.setdefault(key,set()).add(i)
    return aliases


def lexical_support(text,vocab,alias_index):
    """Returns explicit Concept support and negation index; matching is contiguous."""
    tokens=norm(text)
    supported=set(); negated=set(); max_len=max(map(len,alias_index),default=1)
    for start in range(len(tokens)):
        for length in range(1,min(max_len,len(tokens)-start)+1):
            hits=alias_index.get(tokens[start:start+length],set())
            if hits:
                context=tokens[max(0,start-2):start]
                if any(w in NEGATION for w in context):negated.update(hits)
                else:supported.update(hits)
    return supported-negated,negated


def phrases(text,max_n=4,max_phrases=72):
    """Simple bounded n-grams. No external parser/model; transparent limitations."""
    tokens=re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?",text.lower())
    out=[];seen=set()
    for n in range(1,max_n+1):
        for i in range(len(tokens)-n+1):
            words=tokens[i:i+n]
            if not any(w not in IGNORE for w in words):continue
            if any(w in NEGATION for w in words):continue
            if words[0] in IGNORE or words[-1] in IGNORE:continue
            p=' '.join(words)
            if p not in seen:
                out.append(p);seen.add(p)
            if len(out)>=max_phrases:return out
    return out or [text.strip()]


def clip_null(clip_words,sample_count=256,quantile=.95,seed=123):
    """Reference distribution from unpaired sampled gallery concept prompts.

    This is a background distribution for *text*, NOT semantic ground truth.
    """
    n=clip_words.shape[0];g=torch.Generator(device='cpu').manual_seed(seed)
    sample=torch.randperm(n,generator=g)[:min(n,sample_count)].to(clip_words.device)
    sim=clip_words[sample]@clip_words.T
    # Remove identity anchor if present: self similarity must not inflate null.
    for row,col in enumerate(sample.tolist()):sim[row,col]=-1
    return torch.quantile(sim,quantile,dim=0)


class TextScorer:
    def __init__(self,teacher,vocab,clip_words,null_text,aliases=None,low=.44,high=.60,
                 phrase_offset=.015,phrase_margin=.16,unknown_low=.15,unknown_high=.40,
                 max_ngram=4,max_phrases=72):
        self.teacher=teacher;self.vocab=vocab
        self.words=clip_words
        self.null_text=null_text
        self.alias_index=aliases_for_vocab(vocab,aliases)
        self._legacy_lex={norm(c):c for c in vocab}
        self._legacy_idx={c:i for i,c in enumerate(vocab)}
        self.low=low;self.high=high
        self.phrase_offset=phrase_offset;self.phrase_margin=phrase_margin
        self.unknown_low=unknown_low;self.unknown_high=unknown_high
        self.max_ngram=max_ngram;self.max_phrases=max_phrases
        self._cache={}

    def clear(self):
        self._cache.clear()

    @torch.no_grad()
    def score(self,text,method,raw_feature=None):
        key=(text,method,raw_feature is not None)
        if key in self._cache:return self._cache[key]
        n=len(self.vocab)
        if method in ('S0','S1'):
            # Exact original lexical_gallery, NOT expanded directional alias lexicon.
            matched={self._legacy_idx[c] for c in lexical_gallery(text,self._legacy_lex)}
            negated=set()  # preserve source behaviour for S0/S1
        else:
            matched,negated=lexical_support(text,self.vocab,self.alias_index)
        if method in ('S0','S1'):
            feat=raw_feature if raw_feature is not None else self.teacher.embed([text])[0]
            sim=F.normalize(torch.as_tensor(feat,device=self.words.device,dtype=torch.float32),dim=-1)@self.words.T
            vals=torch.where(sim>=self.high,torch.tensor(.8,device=sim.device),torch.tensor(0.,device=sim.device))
            unknown=(sim>self.low)&(sim<self.high)
        elif method=='S2':
            vals=torch.zeros(n,device=self.words.device);unknown=torch.zeros(n,device=self.words.device,dtype=torch.bool)
        else:
            p=phrases(text,self.max_ngram,self.max_phrases)
            feats=self.teacher.embed(p,'a photo of a {}').to(self.words.device)
            feats=F.normalize(feats.float(),dim=-1)
            sim=(feats@self.words.T).amax(dim=0)
            vals=((sim-self.null_text-self.phrase_offset)/self.phrase_margin).clamp(0,1)
            unknown=(vals>self.unknown_low)&(vals<self.unknown_high)
            vals=torch.where(unknown,torch.zeros_like(vals),vals)
        if matched:
            ii=list(matched);vals[ii]=1;unknown[ii]=False
        if negated:
            ii=list(negated);vals[ii]=0;unknown[ii]=False
        result=(vals.detach().cpu().tolist(),unknown.detach().cpu().tolist())
        self._cache[key]=result
        return result
