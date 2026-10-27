"""Pure bookkeeping: fixed-mask coverage, drift attribution, strict concept groups.

The values are gallery-conditional proxies rather than image-level ground truth.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import math


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def coverage(v, t, valid, groups=None):
    """Return full precision D_add and D_remove; undefined denominator -> None.

    groups is a sequence of index groups. Within a group, max V and max T
    represent one strict synonym concept (NOT a semantic-hypernym cluster).
    For grouping, ALL member validity must be true to avoid unknown-as-zero.
    """
    if len(v) != len(t) or len(v) != len(valid):
        raise ValueError('V/T/valid length mismatch')
    if groups is None:
        pairs = [(float(v[i]), float(t[i])) for i in range(len(v)) if valid[i]]
    else:
        pairs = [(max(float(v[i]) for i in g), max(float(t[i]) for i in g))
                 for g in groups if g and all(valid[i] for i in g)]
    sv = sum(a for a, _ in pairs)
    st = sum(b for _, b in pairs)
    return dict(d_add=sum(a*(1-b) for a,b in pairs)/sv if sv > 1e-9 else None,
                d_remove=sum(b*(1-a) for a,b in pairs)/st if st > 1e-9 else None,
                valid_count=len(pairs), visual_mass=sv, text_mass=st)


def comparison_mask(v, old_t, new_t, old_unknown, new_unknown, target, threshold=.25, legacy=False):
    """Fixed shared evaluation universe, with separate UNKNOWN policy.

    S0: legacy union of per-state valid sets, reproducing UNKNOWN->zero bug.
    S1+: select once with pre/post evidence, then require known in BOTH states.
    An unknown target is NOT forcibly made known.
    """
    n=len(v)
    if not all(len(x)==n for x in (old_t,new_t,old_unknown,new_unknown)):
        raise ValueError('Incompatible evidence array lengths')
    base=[(v[i]>=threshold or old_t[i]>=.75 or new_t[i]>=.75 or i==target) for i in range(n)]
    if legacy:
        old=[(v[i]>=threshold or old_t[i]>=.75) and not old_unknown[i] for i in range(n)]
        new=[(v[i]>=threshold or new_t[i]>=.75) and not new_unknown[i] for i in range(n)]
        mask=[a or b for a,b in zip(old,new)]
        if 0<=target<n:mask[target]=True  # reproduce legacy forced-valid target
        return mask
    return [base[i] and not old_unknown[i] and not new_unknown[i] for i in range(n)]


def drift_decomposition(v, old_t, new_t, valid, target, groups=None):
    """Exact additive target vs off-target decomposition of change in D_add.

    IMPORTANT: only valid for fixed V, fixed mask, and ungrouped aggregation.
    Unobserved UNKNOWN concepts are excluded, not imputed zero.
    """
    if groups is not None:
        return {'defined': False, 'reason':'group max aggregation is not additive per concept'}
    mass=sum(float(v[i]) for i in range(len(v)) if valid[i])
    if mass<=1e-9:
        return {'defined':False, 'reason':'no visual mass'}
    weighted=[(-float(v[i])*(float(new_t[i])-float(old_t[i]))/mass) if valid[i] else 0.0
              for i in range(len(v))]
    target_change=weighted[target] if 0<=target<len(v) else 0.0
    off_target=sum(x for i,x in enumerate(weighted) if i!=target)
    off_abs=sum(abs(x) for i,x in enumerate(weighted) if i!=target)
    ratio=off_abs/(abs(target_change)+1e-9)
    return {'defined':True,'target_d_add_contribution':target_change,
            'off_target_d_add_contribution':off_target,
            'off_target_absolute_contribution':off_abs,
            'leakage_ratio':ratio,
            'sum_contribution':target_change+off_target}


def strict_groups(vocab, equivalences):
    """Group strictly equivalent gallery strings without merging hypernyms.

    Any vocabulary item belongs to exactly one disjoint group.
    """
    loc={s.lower():i for i,s in enumerate(vocab)}
    parent=list(range(len(vocab)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for alternatives in equivalences:
        present=[loc[x.lower()] for x in alternatives if x.lower() in loc]
        if len(present)<2:continue
        a=present[0]
        for b in present[1:]: parent[find(b)]=find(a)
    groups=defaultdict(list)
    for i in range(len(vocab)): groups[find(i)].append(i)
    return list(groups.values())


def delta(after, before):
    return None if after is None or before is None else after-before


def is_finite(x):
    return x is not None and isinstance(x,(int,float)) and math.isfinite(x)
