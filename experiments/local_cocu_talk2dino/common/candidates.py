"""Extract gallery concepts from the captions of retrieved images (B/E/F).

Also supports gallery noun phrases via a bounded n-gram matcher; unlike the old
Local-CoCu extractor, it does not drop multiword gallery entries silently.
"""
from __future__ import annotations
from .data import normalize


def from_neighbors(ids,captions,gallery,cap=300):
    lex={normalize(x):x for x in gallery}
    if not lex:return []
    maxlen=max(len(k) for k in lex)
    seen,out=set(),[]
    for iid in ids:
        for text in captions.get(int(iid),[])[:5]:
            words=normalize(text)
            for start in range(len(words)):
                for size in range(1,min(maxlen,len(words)-start)+1):
                    name=lex.get(words[start:start+size])
                    if name is None or name in seen: continue
                    seen.add(name);out.append(name)
                    if len(out)>=cap: return out
    return out
