"""Frozen Teacher/CLIP bidirectional semantic completeness -> E1..E4 edits.
Outputs are *gallery-conditional proxies*, NOT semantic ground truth.
Run with `python -m experiments.Head_caption_completeness.diagnose`.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import sys
import torch
import torch.nn.functional as F
from .core import ROOT,read_json,write_json,gallery,norm,mentioned,lexical_gallery,coverage,make_add,make_remove


def ramp(scores,null,margin):
    """0 below empirical wrong-image null; 1 past a fixed positive margin."""
    return ((scores-null)/margin).clamp_(min=0,max=1)


def text_values(sim,lex_ids,low=.44,high=.60):
    """CLIP text-context evidence: lexical=1, strong paraphrase=.8, ambiguous=UNKNOWN."""
    vals=torch.zeros_like(sim)
    vals[sim>=high]=0.8
    unknown=(sim>low)&(sim<high)
    for i in lex_ids: vals[i]=1.;unknown[i]=False
    return vals,unknown


def group_coverage(captions,vocab,lexicon):
    s=set()
    for caption in captions:s.update(lexical_gallery(caption,lexicon))
    return s


class Context:
    def __init__(self,a):
        from experiments.Head_Match.common.teacher import TeacherBank
        self.a=a
        self.device=torch.device(a.device)
        self.base=torch.load(a.base_train,map_location='cpu',weights_only=False)
        self.images=self.base['images'][:a.limit or None]
        self.groups=defaultdict(list)
        for ann in self.base['annotations']:self.groups[int(ann['image_id'])].append(ann)
        self.vocab=gallery(a.gallery)
        self.lex={norm(x):x for x in self.vocab}
        self.idx={x:j for j,x in enumerate(self.vocab)}
        self.teacher=TeacherBank([a.teacher],a.config,device=a.device,batch_size=a.batch_size)
        self.clip_words=F.normalize(self.teacher.embed(self.vocab,'a photo of a {}').float(),dim=-1)
        self.proj_words=self.teacher.project(self.teacher.embed(self.vocab,'a photo of a {}')).float()
        # CLIP concept embeddings are retained; release the TeacherBank's text cache
        # before scanning up to 400k captions to keep memory bounded.
        self.teacher.cache.clear()
        saved=torch.load(a.clip_images,map_location='cpu',weights_only=False)
        self.clip_images={int(i):v.float() for i,v in zip(saved['ids'],saved['features'])}
        self.neighbors=read_json(a.clip_neighbors)['neighbors']
        full_ids={int(im['id']) for im in self.images}
        missing=full_ids-set(self.clip_images)
        if missing:raise ValueError(f'CLIP image cache misses {len(missing)} images: {list(missing)[:5]}')
        if len(self.vocab)<2:raise ValueError('Require at least two distinct gallery concepts')
        self.null_head,self.null_img=self.calibrate()
        self.counters=Counter()
        self.audit=[]
        self.proposals=[]

    @torch.no_grad()
    def calibrate(self):
        a=self.a
        rows=random.Random(a.seed).sample(self.images,min(a.calibration_images,len(self.images)))
        hs=[];ci=[]
        for im in rows:
            x=torch.as_tensor(im['disentangled_self_attn']).float()
            if x.ndim!=2 or x.shape[1]!=768:raise ValueError(f'Expected [12,768] heads at {im["id"]}, got {tuple(x.shape)}')
            hs.append(F.normalize(x,dim=-1))
            ci.append(F.normalize(self.clip_images[int(im['id'])],dim=-1))
        heads=torch.stack(hs).to(self.device)
        clips=torch.stack(ci).to(self.device)
        vh=[];vc=[]
        for j in range(0,len(self.vocab),a.gallery_chunk):
            g=self.proj_words[j:j+a.gallery_chunk]
            # per-concept distribution over randomly sampled DIFFERENT image heads;
            # max across heads is consistent with source selection, but not an oracle.
            hv=torch.einsum('nhd,kd->nhk',heads,g).amax(dim=1)
            iv=clips@self.clip_words[j:j+a.gallery_chunk].to(self.device).T
            vh.append(torch.quantile(hv,a.null_quantile,dim=0).cpu())
            vc.append(torch.quantile(iv,a.null_quantile,dim=0).cpu())
        print('[NULL] calibrated on',len(rows),'sampled COCO images and',len(self.vocab),'gallery concepts',flush=True)
        return torch.cat(vh).to(self.device),torch.cat(vc).to(self.device)

    @torch.no_grad()
    def per_image(self,im):
        a=self.a;iid=int(im['id']);anns=self.groups[iid]
        if not anns:
            raise ValueError(f"Image {iid} has no captions")

        if len(anns) != 5:
            print(
                f"[WARN] Image {iid} has {len(anns)} captions; "
                "preserving all original annotations."
            )
        heads=F.normalize(torch.as_tensor(im['disentangled_self_attn'],device=self.device).float(),dim=-1)
        if heads.ndim!=2 or heads.shape!=(12,768):raise ValueError(f'Expected [12,768] at image={iid}: {tuple(heads.shape)}')
        feat=F.normalize(self.clip_images[iid].to(self.device),dim=-1)
        # IMPORTANT: ORIGINAL max_score projects RAW CLIP ann_feats before cosine normalization.
        # Never normalize input before the nonlinear Teacher MLP.
        raw_feats=torch.stack([torch.as_tensor(x['ann_feats']).flatten().float() for x in anns]).to(self.device)
        expected_shape = (len(anns), 512)
        if tuple(raw_feats.shape) != expected_shape:
            raise ValueError(
                f"Image {iid}: expected CLIP features "
                f"{expected_shape}, got {tuple(raw_feats.shape)}"
            )
        raw_proj=self.teacher.project(raw_feats)
        head_match=heads@raw_proj.T
        assignments=head_match.argmax(dim=0).tolist()
        teacher_sims=heads@self.proj_words.T
        image_sims=feat@self.clip_words.to(self.device).T
        v_all=ramp(teacher_sims.clone(),self.null_head[None,:]+a.head_offset,a.head_margin)
        vimg=ramp(image_sims.clone(),self.null_img+a.image_offset,a.image_margin)
        v_all=(a.head_weight*v_all+(1-a.head_weight)*vimg[None,:]).clamp(0,1)
        # Reassign uses independent 12-head support, not a fabricated caption loss.
        text_cos=F.normalize(raw_feats,dim=-1)@self.clip_words.to(self.device).T
        lexical=[lexical_gallery(str(ann['caption']),self.lex) for ann in anns]
        same_words=set().union(*map(set,lexical))
        neighborhood=[]
        for nei in self.neighbors.get(str(iid),[])[:a.neighbor_topk]:
            for ann in self.groups.get(int(nei),[])[:5]:
                neighborhood.extend(lexical_gallery(str(ann['caption']),self.lex))
        sources={}
        for x in set(neighborhood):sources.setdefault(x,set()).add('cocu_clip_neighbor')
        for x in same_words:sources.setdefault(x,set()).add('same_image')
        # Gallery-wide Teacher/head retrieval; bounded rank for candidate listing.
        for row in teacher_sims:
            for j in row.topk(min(a.head_topk,len(self.vocab))).indices.tolist():
                sources.setdefault(self.vocab[j],set()).add('head_guided')
        rows=[]
        caption_strings=[str(x['caption']) for x in anns]
        old_union=group_coverage(caption_strings,self.vocab,self.lex)
        for q,ann in enumerate(anns):
            head=int(assignments[q]);v=v_all[head]
            t,unknown=text_values(text_cos[q],set(self.idx[x] for x in lexical[q]),a.text_low,a.text_high)
            valid=((v>=a.relevant_visual)|(t>=.75))&~unknown
            orig=coverage(v.tolist(),t.tolist(),valid.tolist())
            head_support=teacher_sims[head]
            other=teacher_sims.clone();other[head]=-100
            alternate=int(other[:,self.idx[lexical[q][0]]].argmax()) if lexical[q] else None
            # candidate enumeration filters on calibrated evidence + caption novelty.
            candidates=[]
            for c in sources:
                j=self.idx[c]
                if mentioned(c,caption_strings[q]):continue
                if bool(unknown[j]) or t[j]>=.75:continue
                vr=float(v[j]);imv=float(vimg[j])
                if vr<a.min_visual or imv<a.min_image:continue
                candidates.append({'concept':c,'visual':round(vr,4),'image':round(imv,4),
                                   'score':round(.7*vr+.3*imv,5),'sources':sorted(sources[c]),
                                   'global_missing':c not in old_union})
            candidates.sort(key=lambda r:(-r['score'],r['concept']))
            best_add=None
            for can in candidates[:a.try_top]:
                new=make_add(caption_strings[q],[can['concept']])
                if len(new)>a.max_caption_chars or not self.tokens_fit(new):continue
                check=self.counterfactual(heads[head],raw_proj[q],new,can['concept'],v,valid,orig,
                                         old_union,caption_strings,q,add=True)
                if check is not None:
                    best_add={**can,**check,'caption':new,'op':'ADD'};break
            best_remove=None
            # Only literal nouns can be surgically removed safely; never remove
            # a phrase supported by another head, other than the current one.
            for c in lexical[q]:
                j=self.idx[c]
                if c not in set().union(*(set(s) for k,s in enumerate(lexical) if k!=q)):continue
                if float(v[j])>a.remove_visual_ceiling or float(v_all[:,j].max())>a.other_head_ceiling:continue
                if float(vimg[j])>a.remove_image_ceiling:continue
                new=make_remove(caption_strings[q],c)
                if not new or not self.tokens_fit(new):continue
                check=self.counterfactual(heads[head],raw_proj[q],new,c,v,valid,orig,old_union,caption_strings,q,add=False)
                if check is not None:
                    best_remove={'concept':c,'visual':round(float(v[j]),5),'image':round(float(vimg[j]),5),
                                 'score':round(1-float(v[j]),5),'sources':['caption_lexical'],
                                 **check,'caption':new,'op':'REMOVE'}
                    break
            action='KEEP'
            if orig['d_remove']>=a.remove_gate and not best_remove and lexical[q]:
                if any(float(v_all[:,self.idx[x]].max())>a.other_head_ceiling for x in lexical[q]):action='REASSIGN_RECOMMENDED'
                else:action='RECHECK'
            elif orig['d_add']>=a.add_gate and not best_add:action='RECHECK'
            row={'image_id':iid,'annotation_id':int(ann['id']),'raw_caption':caption_strings[q],
                 'head_argmax':head,'head_score':round(float(head_match[head,q]),6),
                 'head_alternative':alternate,'before':orig,'recommendation':action,
                 'head_vs_all_heads':'current_vs_other_heads',
                 'missing_scope':{'other_captions_mention':[x for x in candidates[:a.audit_top] if not x['global_missing']],
                                  'global_missing':[x['concept'] for x in candidates[:a.audit_top] if x['global_missing']]},
                 'candidates':candidates[:a.audit_top],
                 'proposals':{'ADD':best_add,'REMOVE':best_remove}}
            rows.append(row)
            self.counters['captions']+=1
        self.counters['images']+=1
        self.teacher.cache.clear()  # bounded caption cache: only this image
        return rows

    @torch.no_grad()
    def tokens_fit(self,text):
        try:self.teacher.clip.tokenize(text,truncate=False);return True
        except RuntimeError:return False

    @torch.no_grad()
    def counterfactual(self,head,raw_proj,new,concept,v,valid,before,old_union,raw_captions,q,add):
        emb=self.teacher.embed([new])[0].to(self.device)
        projected=self.teacher.project(emb[None,:])[0]
        delta=float(head@(projected-raw_proj))
        j=self.idx[concept]
        after_sim=F.normalize(emb,dim=-1)@self.clip_words.to(self.device).T
        lex=set(self.idx[x] for x in lexical_gallery(new,self.lex))
        t,u=text_values(after_sim,lex,self.a.text_low,self.a.text_high)
        # Keep the concept under test in the valid set even when CLIP gives UNKNOWN.
        vv=valid.clone();vv[j]=True;u[j]=False
        old_sim=coverage(v.tolist(),text_values(F.normalize(self.teacher.embed([raw_captions[q]])[0].to(self.device),dim=-1)@
                             self.clip_words.to(self.device).T,
                             set(self.idx[x] for x in lexical_gallery(raw_captions[q],self.lex)),
                             self.a.text_low,self.a.text_high)[0].tolist(),vv.tolist())
        aft=coverage(v.tolist(),t.tolist(),(vv&~u).tolist())
        # no lost gallery concepts from the 5-caption union, and at least one desired
        # directional loss decreases. Do not reward false head switches.
        new_caps=raw_captions.copy();new_caps[q]=new
        new_union=group_coverage(new_caps,self.vocab,self.lex)
        if not old_union.issubset(new_union):return None
        original=int(len(raw_captions[q]));modified=int(len(new))
        # Counterfactual negative: add a gallery concept with minimal current-head
        # evidence, at the SAME original head. This is a low-evidence control,
        # not a certified incorrect ground-truth label.
        wrong_delta=None
        if add:
            low_order=v.argsort().tolist()
            wrong=next((self.vocab[z] for z in low_order if not mentioned(self.vocab[z],raw_captions[q])
                        and self.vocab[z]!=concept),None)
            if wrong:
                wrong_caption=make_add(raw_captions[q],[wrong])
                if self.tokens_fit(wrong_caption):
                    wrong_feat=self.teacher.embed([wrong_caption])[0].to(self.device)
                    wrong_proj=self.teacher.project(wrong_feat[None,:])[0]
                    wrong_delta=float(head@(wrong_proj-raw_proj))
                    if delta-wrong_delta<self.a.min_wrong_margin:return None
        if add:
            if aft['d_add']>=old_sim['d_add']-self.a.min_improvement:return None
            if aft['d_remove']>old_sim['d_remove']+self.a.max_other_regression:return None
            if delta<self.a.min_teacher_delta:return None
        else:
            if aft['d_remove']>=old_sim['d_remove']-self.a.min_improvement:return None
            if aft['d_add']>old_sim['d_add']+self.a.max_other_regression:return None
            if delta<self.a.min_teacher_delta:return None
        return {'teacher_delta_fixed_head':round(delta,6),
                'low_evidence_control_delta':None if wrong_delta is None else round(wrong_delta,6),
                'before_fixed':old_sim,
                'after_fixed':aft,'length_delta':modified-original}


def output_rows(a,ctx):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    variants={n:(out/f'{n}.jsonl').open('w',encoding='utf-8') for n in ['E0','E1','E2','E3']}
    e4_candidates=[];accepted=Counter()
    audit=(out/'diagnostics.jsonl').open('w',encoding='utf-8')
    def row_for(row,variant):
        cand=row['proposals']
        chosen=None
        if variant in ('E1','E3') and row['before']['d_add']>=a.add_gate:
            chosen=cand['ADD']
        if variant in ('E2','E3') and row['before']['d_remove']>=a.remove_gate and cand['REMOVE']:
            if not chosen or row['before']['d_remove']>row['before']['d_add']:chosen=cand['REMOVE']
        return chosen
    try:
        for ix,im in enumerate(ctx.images):
            for row in ctx.per_image(im):
                if ix<a.audit_images:audit.write(json.dumps(row,ensure_ascii=False)+'\n')
                for variant,f in variants.items():
                    selected=row_for(row,variant)
                    if variant=='E0':selected=None
                    rec={'image_id':row['image_id'],'annotation_id':row['annotation_id'],
                         'raw_caption':row['raw_caption'],
                         'caption':selected['caption'] if selected else row['raw_caption'],
                         'operation':selected['op'] if selected else 'KEEP',
                         'concept':selected['concept'] if selected else None,
                         'head_argmax':row['head_argmax'],
                         'recommendation':row['recommendation'],
                         'diagnosis':row['before'],
                         'after':selected['after_fixed'] if selected else row['before'],
                         'teacher_delta':selected['teacher_delta_fixed_head'] if selected else 0.0}
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    if selected:accepted[(variant,selected['op'])]+=1
                for op in ('ADD','REMOVE'):
                    cand=row['proposals'][op]
                    if cand:e4_candidates.append((float(cand['score']),row['annotation_id'],op,cand))
            if ix%100==0:print(f'[DIAGNOSE] {ix+1}/{len(ctx.images)}',flush=True)
    finally:
        audit.close()
        for f in variants.values():f.close()
    # E4: no completeness gate, but *same number of accepted ADD and REMOVE*
    # as E3 where feasible, with ranking by semantic confidence instead of D.
    budget={op:accepted[('E3',op)] for op in ('ADD','REMOVE')}
    selected={};remaining=dict(budget)
    for _,aid,op,cand in sorted(e4_candidates,key=lambda x:(-x[0],x[1],x[2])):
        if aid not in selected and remaining[op]>0:
            selected[aid]=cand;remaining[op]-=1
    with (out/'E3.jsonl').open(encoding='utf-8') as source,(out/'E4.jsonl').open('w',encoding='utf-8') as target:
        for line in source:
            r=json.loads(line);cand=selected.get(int(r['annotation_id']))
            if cand:
                r.update(caption=cand['caption'],operation=cand['op'],concept=cand['concept'],
                         after=cand['after_fixed'],teacher_delta=cand['teacher_delta_fixed_head'])
                accepted[('E4',cand['op'])]+=1
            else:r.update(caption=r['raw_caption'],operation='KEEP',concept=None,
                          after=r['diagnosis'],teacher_delta=0.)
            target.write(json.dumps(r,ensure_ascii=False)+'\n')
    meta={'base_train':str(a.base_train),'gallery':str(a.gallery),'teacher':str(a.teacher),
          'clip_images':str(a.clip_images),'clip_neighbors':str(a.clip_neighbors),
          'image_count':len(ctx.images),'caption_count':ctx.counters['captions'],
          'E4_unfilled_budget':remaining,'edited':{f'{var}_{op}':int(n) for (var,op),n in accepted.items()},
          'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
          'interpretation':'gallery-conditional proxy; UNKNOWN excluded; teacher-only offline'}
    write_json(out/'metadata.json',meta)
    print('[DIAGNOSTICS]',out/'metadata.json',meta['edited'],flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--gallery',type=Path,default=ROOT/'experiments/local_cocu/MSCOCO14_train_gallery.json')
    p.add_argument('--teacher',type=Path,default=ROOT/'weights/vitb_mlp_infonce.pth')
    p.add_argument('--config',type=Path,default=ROOT/'configs/vitb_mlp_infonce.yaml')
    p.add_argument('--clip-images',type=Path,default=ROOT/'outputs/Head_caption_completeness/clip_images.pth')
    p.add_argument('--clip-neighbors',type=Path,default=ROOT/'outputs/Head_caption_completeness/clip_neighbors.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/Head_caption_completeness/discovery')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=123)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--gallery-chunk',type=int,default=512)
    p.add_argument('--calibration-images',type=int,default=256)
    p.add_argument('--null-quantile',type=float,default=.9)
    p.add_argument('--head-offset',type=float,default=.01)
    p.add_argument('--head-margin',type=float,default=.10)
    p.add_argument('--image-offset',type=float,default=.01)
    p.add_argument('--image-margin',type=float,default=.10)
    p.add_argument('--head-weight',type=float,default=.8)
    p.add_argument('--text-low',type=float,default=.44)
    p.add_argument('--text-high',type=float,default=.60)
    p.add_argument('--head-topk',type=int,default=8)
    p.add_argument('--neighbor-topk',type=int,default=10)
    p.add_argument('--audit-top',type=int,default=6)
    p.add_argument('--audit-images',type=int,default=1000)
    p.add_argument('--try-top',type=int,default=2)
    p.add_argument('--relevant-visual',type=float,default=.25)
    p.add_argument('--min-visual',type=float,default=.65)
    p.add_argument('--min-image',type=float,default=.30)
    p.add_argument('--add-gate',type=float,default=.20)
    p.add_argument('--remove-gate',type=float,default=.25)
    p.add_argument('--remove-visual-ceiling',type=float,default=.25)
    p.add_argument('--other-head-ceiling',type=float,default=.45)
    p.add_argument('--remove-image-ceiling',type=float,default=.25)
    p.add_argument('--min-improvement',type=float,default=.008)
    p.add_argument('--max-other-regression',type=float,default=.04)
    p.add_argument('--min-teacher-delta',type=float,default=-.005)
    p.add_argument('--min-wrong-margin',type=float,default=.005)
    p.add_argument('--max-caption-chars',type=int,default=320)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    for key in ['base_train','gallery','teacher','config','clip_images','clip_neighbors']:
        p=getattr(a,key).expanduser().resolve()
        if not p.is_file():raise FileNotFoundError(f'--{key.replace("_","-")}: {p}')
        setattr(a,key,p)
    if a.limit<0 or a.calibration_images<2:raise ValueError('Bad sample size')
    torch.set_grad_enabled(False)
    ctx=Context(a);output_rows(a,ctx)
if __name__=='__main__':main()
