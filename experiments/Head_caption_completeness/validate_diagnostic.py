"""Step-1 controlled perturbations on 500-1000 COCO images at FIXED raw Head.
Tests directional sensitivity, not true concept correctness or model accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import torch
import torch.nn.functional as F
from .core import ROOT,mentioned,lexical_gallery,coverage,make_add,write_json
from .diagnose import Context,parser as diagnose_parser,ramp,text_values

SYNONYMS=[('bicycle','bike'),('bike','bicycle'),('sofa','couch'),('couch','sofa'),
          ('automobile','car'),('car','automobile'),('motorbike','motorcycle'),('motorcycle','motorbike')]


@torch.no_grad()
def check(ctx,im,ann):
    a=ctx.a
    iid=int(im['id']);h=F.normalize(torch.as_tensor(im['disentangled_self_attn'],device=ctx.device).float(),dim=-1)
    old=torch.as_tensor(ann['ann_feats'],device=ctx.device).float().view(1,-1)  # unnormalized input, as original Teacher
    raw=str(ann['caption'])
    original_head=int((h@ctx.teacher.project(old).T).argmax())
    vs=ramp((h[original_head]@ctx.proj_words.T).clone(),ctx.null_head+a.head_offset,a.head_margin)
    image=F.normalize(ctx.clip_images[iid].to(ctx.device),dim=-1)
    iv=ramp((image@ctx.clip_words.to(ctx.device).T).clone(),ctx.null_img+a.image_offset,a.image_margin)
    visual=a.head_weight*vs+(1-a.head_weight)*iv
    texts=[]
    found=lexical_gallery(raw,ctx.lex)
    if found:
        c=max(found,key=len)
        changed=re.sub(r'\b'+re.escape(c)+r'\b','',raw,count=1,flags=re.I)
        if changed!=raw and len(changed.split())>=3:texts.append(('delete_existing',changed,c))
    low=(h[original_head]@ctx.proj_words.T+.3*(image@ctx.clip_words.to(ctx.device).T)).argsort()
    for idx in low[:min(len(low),150)].tolist():
        c=ctx.vocab[idx]
        if c not in found and not mentioned(c,raw):
            texts.append(('add_wrong',make_add(raw,[c]),c));break
    for word,syn in SYNONYMS:
        if re.search(r'\b'+re.escape(word)+r'\b',raw,re.I):
            texts.append(('synonym',re.sub(r'\b'+re.escape(word)+r'\b',syn,raw,count=1,flags=re.I),word))
            break
    lex_old={ctx.idx[x] for x in found}
    def score(text,feat=None):
        if feat is None:feat=ctx.teacher.embed([text])[0].to(ctx.device)
        sim=F.normalize(feat,dim=-1)@ctx.clip_words.to(ctx.device).T
        lex={ctx.idx[x] for x in lexical_gallery(text,ctx.lex)}
        t,unknown=text_values(sim,lex,a.text_low,a.text_high)
        # Fixed concept set of caption lexical concepts + visual evidence. Any
        # perturbed concept is also forced to valid for controlled comparison.
        valid=((visual>=.25)|(t>=.75))&~unknown
        return t,valid
    old_t,old_valid=score(raw,old.flatten())
    result=[]
    for kind,text,concept in texts:
        new_t,new_valid=score(text)
        j=ctx.idx[concept]
        # same valid mask for both scores so denominator shifts are comparable
        valid=(old_valid|new_valid);valid[j]=True
        before=coverage(visual.tolist(),old_t.tolist(),valid.tolist())
        after=coverage(visual.tolist(),new_t.tolist(),valid.tolist())
        result.append({'kind':kind,'original':raw,'modified':text,'concept':concept,
                       'image_id':iid,'annotation_id':int(ann['id']), 'head_fixed':original_head,
                       'd_add_before':before['d_add'],'d_add_after':after['d_add'],
                       'd_remove_before':before['d_remove'],'d_remove_after':after['d_remove']})
    ctx.teacher.cache.clear()
    return result


def main(argv=None):
    p=diagnose_parser()
    p.add_argument('--report',type=Path,default=ROOT/'outputs/Head_caption_completeness/validation/report.json')
    p.set_defaults(limit=800)
    a=p.parse_args(argv)
    for key in ['base_train','gallery','teacher','config','clip_images','clip_neighbors']:
        setattr(a,key,getattr(a,key).resolve())
    torch.set_grad_enabled(False)
    ctx=Context(a)
    results=[]
    for k,im in enumerate(ctx.images):
        anns=ctx.groups[int(im['id'])]
        for ann in anns[:1]:results.extend(check(ctx,im,ann))
        if k%100==0:print('[PERTURB]',k+1,'/',len(ctx.images),flush=True)
    groups={kind:[x for x in results if x['kind']==kind] for kind in ['delete_existing','add_wrong','synonym']}
    summary={}
    for kind,records in groups.items():
        if not records:summary[kind]={'n':0,'warning':'No perturbable captions; no validation conclusion'};continue
        da=[x['d_add_after']-x['d_add_before'] for x in records]
        dr=[x['d_remove_after']-x['d_remove_before'] for x in records]
        summary[kind]={'n':len(records),'mean_delta_add':statistics.mean(da),
                       'mean_delta_remove':statistics.mean(dr),
                       'frac_d_add_increased':sum(v>0 for v in da)/len(da),
                       'frac_d_remove_increased':sum(v>0 for v in dr)/len(dr)}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.report,{'sample_images':len(ctx.images),'summary':summary,'records':results,
                         'caveat':'Fixed Teacher argmax head. Synthetic perturbations do not establish actual caption truth.'})
    print('[VALIDATION]',a.report,summary,flush=True)
if __name__=='__main__':main()
