"""Deterministic noun-only editing policy, no LLM or repository-core changes.

Intentional abstention: a gallery entry is not proof it is a visible object.
The curated noun set is a type filter, NOT object-presence ground truth.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict

# Common COCO object categories plus conservative concrete MSCOCO gallery nouns.
# Add externally verified object names via --object-allowlist (one per line).
OBJECTS = set('''airplane|apple|backpack|banana|baseball bat|baseball glove|bear|bed|bench|bicycle|bird|boat|book|bottle|bowl|broccoli|bus|cake|car|carrot|cat|cell phone|chair|clock|couch|cow|cup|dining table|dog|donut|elephant|fire hydrant|fork|frisbee|giraffe|handbag|horse|hot dog|keyboard|kite|knife|laptop|microwave|motorcycle|mouse|orange|oven|parking meter|pizza|potted plant|refrigerator|remote|sandwich|scissors|sheep|sink|skateboard|skis|snowboard|spoon|sports ball|stop sign|suitcase|surfboard|teddy bear|tennis racket|tie|toaster|toilet|toothbrush|traffic light|train|truck|tv|umbrella|vase|wine glass|zebra|wheel|plate|table|tree|flower|grass|mountain|building|window|door|road|street sign|ball|bag|helmet|bike|stroller|blanket|shirt|hat|shoe|tire|fence|camera|guitar|computer|monitor|lamp|bookshelf|basket|sidewalk|garden|sofa|bicycle rack|boot|watch|rug|pillow|pan|pot|desk|flowerpot|crane|flag|bridge|paper plane|raincoat|bouquet'''.split('|'))
# Refuse human, verb, and non-object entries. Objects like "mouse" can be ambiguous;
# semantic type must still be manually audited in training candidates.
EXCLUDE = set('''person|people|man|woman|boy|girl|child|children|men|women|running|walking|riding|standing|sitting|eating|playing|wearing|holding|beautiful|red|blue|sunny|outside|background'''.split('|'))
IRREG_PLURALS = {'mouse':'mice','knife':'knives','leaf':'leaves','bus':'buses','dish':'dishes','box':'boxes','person':'people'}


def eligible(concept, extra=None):
    c=str(concept).strip().lower()
    return bool(c) and c not in EXCLUDE and (c in OBJECTS or (extra and c in extra))


def forms(concept):
    """Surface spellings; avoid stemming unrelated words or substring replacements."""
    c=concept.strip().lower()
    if not c:return []
    final=c.split()[-1]
    if final.endswith(('s','x','z','ch','sh')):p=final+'es'
    elif final.endswith('y') and len(final)>1 and final[-2] not in 'aeiou':p=final[:-1]+'ies'
    else:p=final+'s'
    plural=IRREG_PLURALS.get(final,p)
    a=c.rsplit(' ',1)[0]+' '+plural if ' ' in c else plural
    return [(a,True),(c,False)] if a!=c else [(c,False)]


def matches(text,concept):
    out=[]
    for form,plural in forms(concept):
        pat=r'(?<!\w)'+r'\s+'.join(re.escape(w) for w in form.split())+r'(?!\w)'
        out.extend((m.start(),m.end(),plural) for m in re.finditer(pat,text,re.I))
    # Skip overlapping detections; prefer longer match at a shared start.
    out.sort(key=lambda x:(x[0],-(x[1]-x[0])))
    final=[]
    for m in out:
        if final and m[0]<final[-1][1]:continue
        final.append(m)
    return final


def mentioned(text,concept):return bool(matches(text,concept))


@dataclass
class EditResult:
    text:str
    applied:list
    skipped:list
    spans:list


def neutralize(raw,concept,extra=None):
    """Replace every exact object-noun mention with thing/things; preserve clauses."""
    if not eligible(concept,extra):return EditResult(raw,[],[{'concept':concept,'reason':'not_curated_object_noun'}],[])
    spans=matches(raw,concept)
    if not spans:return EditResult(raw,[],[{'concept':concept,'reason':'no_exact_surface_match'}],[])
    out=raw
    for lo,hi,plural in reversed(spans):
        out=out[:lo]+('things' if plural else 'thing')+out[hi:]
    out=re.sub(r'\ban\s+(?=thing\b)',lambda m: 'A ' if m.group(0)[0].isupper() else 'a ',out,flags=re.I)
    return EditResult(out,[concept],[],[{'start':x,'end':y,'plural':p} for x,y,p in spans])


def noun_phrase(concept):
    c=concept.strip().lower()
    if not c:return c
    # Preserve existing plural surface forms in the gallery.
    if c.endswith('s') and not c.endswith(('ss','us')):return c
    return ('an ' if c[0] in 'aeiou' else 'a ')+c


def append_visible(raw,concepts,template='with_also_visible_v1',extra=None):
    kept=[];skipped=[]
    seen=set()
    for c in concepts:
        key=c.lower().strip()
        if not eligible(c,extra):skipped.append({'concept':c,'reason':'not_curated_object_noun'});continue
        if key in seen or mentioned(raw,c):skipped.append({'concept':c,'reason':'duplicate_or_already_mentioned'});continue
        seen.add(key);kept.append(c)
    if not kept:return EditResult(raw,[],skipped,[])
    if len(kept)==1:joined=noun_phrase(kept[0])
    elif len(kept)==2:joined=' and '.join(map(noun_phrase,kept))
    else:joined=', '.join(map(noun_phrase,kept[:-1]))+', and '+noun_phrase(kept[-1])
    base=raw.rstrip().rstrip('.!?').rstrip()
    if template=='with_also_visible_v1':edited=f'{base}, with {joined} also visible.'
    elif template=='with_visible_v0':edited=f'{base}, with visible {", ".join(kept)}.'
    else:raise ValueError('Unknown template '+template)
    return EditResult(edited,kept,skipped,[])


def edit(raw,remove=None,add=None,template='with_also_visible_v1',extra=None):
    rem=neutralize(raw,remove,extra) if remove else EditResult(raw,[],[],[])
    added=append_visible(rem.text,[add] if add else [],template,extra)
    return {'caption':added.text,'applied_remove':rem.applied,'applied_add':added.applied,
            'skipped':rem.skipped+added.skipped,'remove_spans':rem.spans,'template_id':template}
