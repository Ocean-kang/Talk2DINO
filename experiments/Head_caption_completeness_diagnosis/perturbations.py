"""Deterministic, transparently labeled synthetic interventions.

High/low visual evidence is NOT a verified object-presence/absence label.
"""
from __future__ import annotations
import re
from .text_evidence import lexical_support
from experiments.Head_caption_completeness.core import make_add, norm

SYNONYM_PAIRS=[('bicycle','bike'),('bike','bicycle'),('sofa','couch'),('couch','sofa'),
               ('car','automobile'),('automobile','car'),('motorcycle','motorbike'),('motorbike','motorcycle'),
               ('tv','television'),('television','tv')]


def remove_mention(text,concept):
    """Conservative synthetic string edit, not a fluent-text guarantee."""
    escaped=r'\s+'.join(map(re.escape,concept.split()))
    pattern=r'\b'+escaped+r'\b'
    # Prefer removing determiner + noun, and optionally an immediately adjacent adjective.
    update,n=re.subn(r'\b(?:a|an|the)\s+'+pattern,'',text,count=1,flags=re.I)
    if n==0:update,n=re.subn(pattern,'',text,count=1,flags=re.I)
    if not n:return None
    update=re.sub(r'\s+([,.!?])',r'\1',update)
    update=re.sub(r'\s{2,}',' ',update).strip()
    update=re.sub(r'\b(and|with|on|near|beside|of|by)\s*[.!?]?$','',update,flags=re.I).strip()
    if len(norm(update))<3 or update==text:return None
    if not re.search(r'[.!?]$',update):update=update.rstrip(',')+'.'
    return update


def make_cases(raw,vocab,aliases,head_v,image_v,other_v,
               min_high=.60,min_image=.30,max_low=.12,max_other=.30,max_low_image=.18):
    """Pick at most one case per kind using visual evidence; fixed Head throughout."""
    support,negated=lexical_support(raw,vocab,aliases)
    candidates=[]
    for i in sorted(support,key=lambda x:(-head_v[x],vocab[x])):
        updated=remove_mention(raw,vocab[i])
        if updated and head_v[i]>=min_high:
            candidates.append({'kind':'delete_supported','target':i,'modified':updated,
                               'control_label':'visual_high_edit_not_ground_truth'})
            break
    for i in sorted(range(len(vocab)),key=lambda x:(-head_v[x],-image_v[x],vocab[x])):
        if i in support or i in negated:continue
        if head_v[i]>=min_high and image_v[i]>=min_image and len(vocab[i].split())<=4:
            candidates.append({'kind':'add_high_evidence','target':i,'modified':make_add(raw,[vocab[i]]),
                               'control_label':'high_visual_evidence_not_verified'})
            break
    for i in sorted(range(len(vocab)),key=lambda x:(head_v[x]+image_v[x]+other_v[x],vocab[x])):
        if i in support or i in negated:continue
        if head_v[i]<=max_low and image_v[i]<=max_low_image and other_v[i]<=max_other and len(vocab[i].split())<=3:
            candidates.append({'kind':'add_low_evidence','target':i,'modified':make_add(raw,[vocab[i]]),
                               'control_label':'low_visual_evidence_not_proven_absent'})
            break
    for a,b in SYNONYM_PAIRS:
        pattern=r'\b'+re.escape(a)+r'\b'
        if re.search(pattern,raw,re.I):
            updated=re.sub(pattern,b,raw,count=1,flags=re.I)
            if updated!=raw:
                target=next((i for i,c in enumerate(vocab) if c.lower() in (a,b)),None)
                if target is not None:
                    candidates.append({'kind':'synonym','target':target,'modified':updated,
                                       'control_label':'manually_curated_synonym'})
                    break
    # Semantically near-neutral prefix. It deliberately tests global text-context sensitivity.
    candidates.append({'kind':'neutral_prefix','target':-1,'modified':'Here, '+raw[0].lower()+raw[1:] if raw else raw,
                       'control_label':'approximate_semantic_preservation'})
    return [x for x in candidates if x['modified']!=raw]
