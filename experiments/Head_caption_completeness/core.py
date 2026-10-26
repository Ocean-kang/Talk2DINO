"""BSC-CR pure semantic bookkeeping. Thresholds are validation hyperparameters,
not ground-truth semantic labels. No external model/dependency required here.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

BASELINES = {'voc20':88.54, 'voc':65.77, 'coco_stuff':30.23, 'ade20k':22.53, 'context59':42.42}
BENCHMARKS = {
    'voc20': ('voc/dinotext_voc_vitb_mlp_infonce.yml', 'voc/eval_voc_pamr.yml'),
    'voc': ('voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml', 'voc_bg/eval_voc_bg_pamr.yml'),
    'coco_stuff': ('stuff/dinotext_stuff_vitb_mlp_infonce.yml', 'stuff/eval_stuff_pamr.yml'),
    'ade20k': ('ade/dinotext_ade_vitb_mlp_infonce.yml', 'ade/eval_ade_pamr.yml'),
    'context59': ('context/dinotext_context_vitb_mlp_infonce.yml', 'context/eval_context_pamr.yml'),
}
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IRREG = {'men':'man','women':'woman','people':'person','children':'child','mice':'mouse','feet':'foot','teeth':'tooth','geese':'goose'}
PAT = re.compile(r'[a-z0-9]+')


def norm(s):
    out=[]
    for word in PAT.findall(str(s).lower()):
        word=IRREG.get(word,word)
        if len(word)>4 and word.endswith('ies'):word=word[:-3]+'y'
        elif len(word)>4 and word.endswith(('sses','shes','ches','xes')):word=word[:-2]
        elif len(word)>3 and word.endswith('s') and not word.endswith(('ss','us','is','ous')):word=word[:-1]
        out.append(word)
    return tuple(out)


def mentioned(concept,text):
    a,b=norm(concept),norm(text)
    return bool(a) and any(a==b[i:i+len(a)] for i in range(len(b)-len(a)+1))


def lexical_gallery(text, vocab):
    """Bounded contiguous n-grams; lexical evidence is high confidence only."""
    words=norm(text)
    vocab_map=vocab if isinstance(vocab,dict) else {norm(x):x for x in vocab}
    upper=max((len(x) for x in vocab_map),default=0)
    result=[]
    for i in range(len(words)):
        for n in range(1,min(upper,len(words)-i)+1):
            x=vocab_map.get(words[i:i+n])
            if x is not None and x not in result: result.append(x)
    return result


def coverage(v,t,valid=None,eps=1e-9):
    """Equation in BSC_CR document; excludes uncertain concepts explicitly."""
    if len(v)!=len(t):raise ValueError('V/T size mismatch')
    valid = [True]*len(v) if valid is None else valid
    values=[(float(a),float(b)) for a,b,good in zip(v,t,valid) if good]
    sv=sum(a for a,_ in values); st=sum(b for _,b in values)
    add=sum(a*(1-b) for a,b in values)/(sv+eps) if sv>eps else 0.
    rem=sum(b*(1-a) for a,b in values)/(st+eps) if st>eps else 0.
    return {'d_add':round(add,6),'d_remove':round(rem,6),'valid_count':len(values),
            'visual_mass':round(sv,5),'text_mass':round(st,5)}


def make_add(raw,concepts):
    if not concepts:return raw
    # No fabricated spatial, action, or object-object relationships.
    return raw.rstrip().rstrip('.')+'; also visible: '+', '.join(concepts)+'.'


def make_remove(raw, concept):
    """Only remove safe *trailing* separable noun phrases. Never regex out core nouns.

    This deliberately abstains on most captions: semantic safety > edit volume.
    """
    noun=re.escape(concept).replace(r'\ ',r'\s+')
    pats=[
        r',\s*(?:and\s+)?(?:a|an|the)\s+'+noun+r'(?=\s*[.!?]?\s*$)',
        r'\s+and\s+(?:a|an|the)\s+'+noun+r'(?=\s*[.!?]?\s*$)',
        r',\s*'+noun+r'(?=\s*[.!?]?\s*$)',
        r'\s+and\s+'+noun+r'(?=\s*[.!?]?\s*$)',
    ]
    for pat in pats:
        updated,count=re.subn(pat,'',raw,flags=re.I)
        if count and len(norm(updated))>=3 and not mentioned(concept,updated):
            return re.sub(r'\s+([.!?])',r'\1',updated).rstrip()
    return None


def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(p)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def gallery(path):
    data=read_json(path)
    if isinstance(data,dict):data=data.get('concept',list(data))
    if not isinstance(data,list):raise ValueError('Gallery must be list or {"concept":list}')
    uniq={}
    for x in data:
        if isinstance(x,dict):x=x.get('concept','')
        if not isinstance(x,str):continue
        key=norm(x)
        if key and key not in uniq:uniq[key]=x.strip().lower()
    if not uniq:raise ValueError('Gallery empty')
    return list(uniq.values())


def view_changed(records):
    return sum(bool(x.get('accepted')) for x in records)
