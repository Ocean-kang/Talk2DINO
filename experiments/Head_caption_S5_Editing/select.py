"""Frozen-head candidates -> Editing Policy -> S5 qualification -> E0-E3 manifests.

Run from repository root with python -m experiments.Head_caption_S5_Editing.select.
The implementation reuses S5 FrozenData, TextScorer, comparison_mask,
coverage, drift_decomposition, and original TeacherBank without modification.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import torch
import torch.nn.functional as F

from experiments.Head_caption_completeness.core import ROOT, write_json
from experiments.Head_caption_completeness_diagnosis.run import FrozenData, build_parser as s5_parser
from experiments.Head_caption_completeness_diagnosis.core import comparison_mask,coverage,drift_decomposition
from experiments.Head_caption_completeness_diagnosis.text_evidence import lexical_support
from .editing import eligible,mentioned,edit

OUT=ROOT/'outputs/Head_caption_S5_Editing'


def tokenize_info(teacher, text):
    """Use EXACT CLIP tokenizer with truncation prohibited."""
    try:
        t=teacher.clip.tokenize(text,truncate=False)
    except (RuntimeError,ValueError):
        return False,None
    return True,int((t!=0).sum().item())


def score_change(v,t_before,u_before,t_after,u_after,target):
    """S5 fixed-universe counterfactual; None means abstain, never UNKNOWN->0."""
    mask=comparison_mask(v,t_before,t_after,u_before,u_after,target,.25)
    before=coverage(v,t_before,mask);after=coverage(v,t_after,mask)
    j=target
    known=bool(mask[j] and not u_before[j] and not u_after[j])
    leak=drift_decomposition(v,t_before,t_after,mask,j)
    off=[abs(float(t_after[k])-float(t_before[k])) for k in range(len(v)) if mask[k] and k!=j]
    return {'target_known':known,'target_delta':float(t_after[j]-t_before[j]) if known else None,
            'd_add_before':before['d_add'],'d_add_after':after['d_add'],
            'd_remove_before':before['d_remove'],'d_remove_after':after['d_remove'],
            'delta_add':None if before['d_add'] is None or after['d_add'] is None else after['d_add']-before['d_add'],
            'delta_remove':None if before['d_remove'] is None or after['d_remove'] is None else after['d_remove']-before['d_remove'],
            'off_target_mean_abs_delta':sum(off)/len(off) if off else 0.,
            'target_d_add_contribution':leak.get('target_d_add_contribution'),
            'off_target_abs_contribution':leak.get('off_target_absolute_contribution'),
            'leakage_ratio':leak.get('leakage_ratio'),'valid_count':sum(mask)}


def admissible(metrics,kind,a):
    """Precommitted conservative diagnostic thresholds, not an entailment oracle."""
    m=metrics
    if not m['target_known'] or m['target_delta'] is None:return False,'target_unknown'
    if m['valid_count']<1 or m['off_target_mean_abs_delta']>a.max_off_drift:
        return False,'insufficient_valid_or_off_target_drift'
    if kind=='ADD':
        if (m['target_delta'] < a.min_target_effect or m['delta_add'] is None or
            m['delta_add'] >= -a.min_aggregate_effect):return False,'weak_add_completeness_gain'
        target=m['target_d_add_contribution'];off=m['off_target_abs_contribution']
        if target is None or off is None or abs(target)<off*a.target_dominance:
            return False,'add_off_target_dominates'
    else:
        if (m['target_delta']>-a.min_target_effect or m['delta_remove'] is None or
            m['delta_remove']>=-a.min_aggregate_effect):return False,'weak_remove_completeness_gain'
        if m['delta_add'] is not None and m['delta_add']>a.max_remove_add_penalty:
            return False,'remove_harms_visual_coverage'
    return True,'S5_pass'


def candidate_lists(vocab,sim,raw,topk,extra,remove_pool='caption_low10'):
    """Top/bottom gallery per fixed caption-head, anchored caption-low for edits."""
    ordered=sorted(range(len(vocab)),key=lambda i:(-float(sim[i]),vocab[i]))
    hi=ordered[:topk];lo=list(reversed(ordered[-topk:]))
    # Remove's lexical guard: low10 of the entire gallery usually never occurs in raw.
    # Keep global bottom10 as a negative-control log, then optionally restrict to raw.
    present=[i for i in ordered if eligible(vocab[i],extra) and mentioned(raw,vocab[i])]
    rem=list(reversed(present[-topk:])) if remove_pool=='caption_low10' else [i for i in lo if i in present]
    return hi,lo,rem


@torch.no_grad()
def analyze_annotation(ctx,image,ann,a,extra):
    iid=int(image['id']);aid=int(ann['id']);raw=str(ann['caption'])
    ev=ctx.evidence(image,ann)
    online=ctx.teacher.embed([raw])[0].to(ctx.device)
    cached_cos=float(F.cosine_similarity(F.normalize(ev['raw_feat'],dim=0),
                                           F.normalize(online,dim=0),dim=0))
    if cached_cos<a.min_cache_online_cos and not a.allow_feature_drift:
        raise RuntimeError(f'CLIP feature mismatch annotation={aid}, cosine={cached_cos:.6f}; '
                           'source ann_feats and current CLIP tokenizer/checkpoint are not consistent')
    # Frozen Teacher fixed-head raw cosine, no Projector retraining.
    similarities=(ev['head_feat']@ctx.proj_words.T).detach().cpu().tolist()
    hi,lo,rem=candidate_lists(ctx.vocab,similarities,raw,a.topk,extra,a.remove_pool)
    old,uold=ctx.text.score(raw,'S3')
    v=ev['v_head'];vi=ev['v_image'];vo=ev['v_other']
    chosen={};audit=[]
    raw_teacher=ev['head_score']
    candidates=[('ADD',i) for i in hi]+[('REMOVE',i) for i in rem]
    support,negated=lexical_support(raw,ctx.vocab,ctx.text.alias_index)
    n_eval={'ADD':0,'REMOVE':0}
    for kind,i in candidates:
        concept=ctx.vocab[i]
        item={'kind':kind,'concept':concept,'gallery_index':i,'raw_similarity':float(similarities[i]),
              'head_visual':float(v[i]),'image_visual':float(vi[i]),'other_heads_visual':float(vo[i])}
        reason=None
        if not eligible(concept,extra):reason='not_curated_object_noun'
        if kind=='ADD' and reason is None:
            if i in support or i in negated or mentioned(raw,concept):reason='already_expressed_or_negated'
            elif v[i]<a.min_add_visual or vi[i]<a.min_add_image:reason='insufficient_visual_evidence'
            elif old[i]>a.max_initial_text or uold[i]:reason='text_present_or_unknown'
        elif kind=='REMOVE' and reason is None:
            if not mentioned(raw,concept):reason='no_literal_object_mention'
            elif v[i]>a.max_remove_visual:reason='head_not_low_evidence'
            elif a.remove_guard=='all_views' and (vi[i]>a.max_remove_image or vo[i]>a.max_remove_other):
                reason='other_view_may_support_object'
        if reason:
            item.update(accepted=False,reason=reason);audit.append(item);continue
        if n_eval[kind]>=a.max_evals_per_operation:
            item.update(accepted=False,reason='evaluation_budget');audit.append(item);continue
        n_eval[kind]+=1
        result=edit(raw,remove=concept if kind=='REMOVE' else None,
                    add=concept if kind=='ADD' else None,template=a.template,extra=extra)
        edited=result['caption']
        valid,ntok=tokenize_info(ctx.teacher,edited)
        if edited==raw or not valid or len(edited)>a.max_caption_chars:
            item.update(accepted=False,reason='unchanged_or_clip_token_length_exceeded',token_count=ntok)
            audit.append(item);continue
        new,unew=ctx.text.score(edited,'S3')
        met=score_change(v,old,uold,new,unew,i)
        okay,reason=admissible(met,kind,a)
        if okay and a.min_teacher_delta is not None:
            raw_online=ctx.teacher.embed([edited])[0].to(ctx.device)
            teacher_s=float(ev['head_feat']@ctx.teacher.project(raw_online.unsqueeze(0))[0])
            if teacher_s-raw_teacher<a.min_teacher_delta:
                okay=False;reason='teacher_similarity_drop'
            met['teacher_head_delta']=teacher_s-raw_teacher
        item.update(accepted=okay,reason=reason,edited_caption=edited if okay else None,
                    token_count=ntok,metrics=met,policy=result)
        audit.append(item)
        if okay:
            benefit= (-met['delta_add'] if kind=='ADD' else -met['delta_remove'])
            score=(benefit,abs(met['target_delta']),-i)
            if kind not in chosen or score>chosen[kind][0]:chosen[kind]=(score,item)
        ctx.text.clear()  # keep memory bounded (raw re-evaluated if necessary)
    add=chosen.get('ADD',(None,None))[1];remove=chosen.get('REMOVE',(None,None))[1]
    e1=edit(raw,remove=remove['concept'] if remove else None,template=a.template,extra=extra)['caption']
    e2=edit(raw,add=add['concept'] if add else None,template=a.template,extra=extra)['caption']
    combined=edit(raw,remove=remove['concept'] if remove else None,
                  add=add['concept'] if add else None,template=a.template,extra=extra)
    e3=combined['caption'];combined_reason='passed_individually'
    if remove and add:
        valid,_=tokenize_info(ctx.teacher,e3)
        if not valid:combined_reason='combined_token_overflow';e3=e2
        else:
            new,unew=ctx.text.score(e3,'S3')
            # Explicit shared-universe checks: both intended directions should improve.
            ma=score_change(v,old,uold,new,unew,add['gallery_index'])
            mr=score_change(v,old,uold,new,unew,remove['gallery_index'])
            # Combination effects interact, so do not enforce single-operation off-target
            # dominance (each other's concept is intentionally altered).
            if not (ma['target_known'] and mr['target_known'] and
                    ma['target_delta']>=a.min_target_effect and
                    mr['target_delta']<=-a.min_target_effect and
                    (ma['delta_add'] is not None and ma['delta_add']<a.max_combined_add_change) and
                    (mr['delta_remove'] is not None and mr['delta_remove']<a.max_combined_remove_change)):
                combined_reason='combined_failed_S5';e3=e2 if add['metrics']['delta_add']<remove['metrics']['delta_remove'] else e1
    selections={'E0':raw,'E1':e1,'E2':e2,'E3':e3}
    record={'image_id':iid,'annotation_id':aid,'head_fixed':ev['head_id'],
            'raw_caption':raw,'selected_add':add['concept'] if add else None,
            'selected_remove':remove['concept'] if remove else None,
            'global_high_top10':[{'concept':ctx.vocab[i],'similarity':float(similarities[i]),'V':v[i]} for i in hi],
            'global_low_top10':[{'concept':ctx.vocab[i],'similarity':float(similarities[i]),'V':v[i]} for i in lo],
            'caption_low_top10':[{'concept':ctx.vocab[i],'similarity':float(similarities[i]),'V':v[i]} for i in rem],
            'combination_reason':combined_reason,'candidates':audit,'captions':selections,
            'template_id':a.template,'teacher_head_raw_similarity':float(raw_teacher),
            'raw_cached_vs_online_clip_cosine':cached_cos}
    return record


def build_parser():
    p=s5_parser()
    # Start with EXACT S5 arguments/default values then override training-discovery defaults.
    p.set_defaults(limit=0,captions_per_image=0,methods='S5',output=OUT/'discovery')
    p.add_argument('--topk',type=int,default=10)
    p.add_argument('--max-evals-per-operation',type=int,default=10,
                   help='Max candidates per ADD/REMOVE to run expensive S5 CLIP rescoring; all TopK candidates are still logged')
    p.add_argument('--allow-feature-drift',action='store_true')
    p.add_argument('--remove-pool',choices=['caption_low10','global_low10'],default='caption_low10')
    p.add_argument('--remove-guard',choices=['all_views','head_only'],default='all_views')
    p.add_argument('--object-allowlist',type=Path)
    p.add_argument('--template',choices=['with_also_visible_v1','with_visible_v0'],default='with_also_visible_v1')
    p.add_argument('--min-add-visual',type=float,default=.60)
    p.add_argument('--min-add-image',type=float,default=.30)
    p.add_argument('--max-initial-text',type=float,default=.25)
    p.add_argument('--max-remove-visual',type=float,default=.12)
    p.add_argument('--max-remove-image',type=float,default=.18)
    p.add_argument('--max-remove-other',type=float,default=.30)
    p.add_argument('--min-target-effect',type=float,default=.20)
    p.add_argument('--min-aggregate-effect',type=float,default=.001)
    p.add_argument('--max-off-drift',type=float,default=.10)
    p.add_argument('--target-dominance',type=float,default=1.)
    p.add_argument('--max-remove-add-penalty',type=float,default=.01)
    p.add_argument('--max-combined-add-change',type=float,default=.02)
    p.add_argument('--max-combined-remove-change',type=float,default=.02)
    p.add_argument('--min-teacher-delta',type=float,default=-.03)
    p.add_argument('--flush-every',type=int,default=200)
    return p


def main(argv=None):
    a=build_parser().parse_args(argv)
    if a.topk<1 or a.topk>100 or a.max_evals_per_operation<1:
        raise ValueError('--topk must be in [1,100], --max-evals-per-operation>=1')
    if a.captions_per_image < 0 or a.limit < 0:raise ValueError('limit and captions-per-image must be nonnegative (0=all)')
    for key in ('base_train','gallery','teacher','config','clip_images'):
        path=getattr(a,key).resolve()
        if not path.is_file():raise FileNotFoundError(f'{key}: {path}')
        setattr(a,key,path)
    a.methods=['S5'];a.output=a.output.resolve()
    extra=set()
    if a.object_allowlist:
        extra={s.strip().lower() for s in a.object_allowlist.read_text(encoding='utf8').splitlines() if s.strip() and not s.startswith('#')}
    random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_grad_enabled(False)
    ctx=FrozenData(a)
    a.output.mkdir(parents=True,exist_ok=True)
    group=defaultdict(list)
    for x in ctx.data['annotations']:group[int(x['image_id'])].append(x)
    # Guard against a missing/duplicated image or annotation before writing manifests.
    original_image_ids = [int(im['id']) for im in ctx.data['images']]
    original_ann_ids = [int(ann['id']) for ann in ctx.data['annotations']]
    if len(set(original_image_ids)) != len(original_image_ids):raise ValueError('Duplicate image IDs')
    if len(set(original_ann_ids)) != len(original_ann_ids):raise ValueError('Duplicate annotation IDs')
    if set(group) - set(original_image_ids):raise ValueError('Orphan caption image IDs')
    files={k:(a.output/f'{k}.jsonl').open('w',encoding='utf8') for k in ('E0','E1','E2','E3')}
    audit=(a.output/'candidates.jsonl').open('w',encoding='utf8')
    cnt=Counter()
    try:
        for pos,im in enumerate(ctx.images):
            iid=int(im['id']);rows=group.get(iid,[])
            if not rows:raise ValueError(f'image {iid} has no captions')
            # 0 = ALL original captions for this image; positive values are diagnostic caps.
            selected_rows = rows if a.captions_per_image == 0 else rows[:a.captions_per_image]
            for ann in selected_rows:
                r=analyze_annotation(ctx,im,ann,a,extra)
                audit.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
                cnt['captions']+=1
                for k in files:
                    final=r['captions'][k]
                    if final==r['raw_caption']:op='KEEP'
                    elif k=='E1':op='REMOVE'
                    elif k=='E2':op='ADD'
                    elif k=='E3' and final==r['captions']['E1']:op='REMOVE'
                    elif k=='E3' and final==r['captions']['E2']:op='ADD'
                    elif k=='E3':op='REMOVE_ADD'
                    else:raise AssertionError('Unexpected E0 edit')
                    uses_add=op in ('ADD','REMOVE_ADD')
                    uses_remove=op in ('REMOVE','REMOVE_ADD')
                    row={'image_id':iid,'annotation_id':r['annotation_id'],
                         'raw_caption':r['raw_caption'],'edited_caption':final,
                         'operation':op,'fixed_head':r['head_fixed'],
                         'add_concept':r['selected_add'] if uses_add else None,
                         'remove_concept':r['selected_remove'] if uses_remove else None,
                         'combination_reason':r['combination_reason'] if k=='E3' else None,
                         'template_id':a.template}
                    files[k].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    cnt[f'{k}_{op}']+=1
                ctx.text.clear();ctx.teacher.cache.clear()
            cnt['images']+=1
            if (pos+1)%a.flush_every==0 or pos==0:
                for f in list(files.values())+[audit]:f.flush()
                print(f'[DISCOVERY] {pos+1}/{len(ctx.images)} images, edits={dict(cnt)}',flush=True)
    finally:
        for f in list(files.values())+[audit]:f.close()
    write_json(a.output/'metadata.json',{'base_train':str(a.base_train),'gallery':str(a.gallery),
        'teacher':str(a.teacher),'clip_images':str(a.clip_images),'selected_images':cnt['images'],
        'selected_captions':cnt['captions'],'total_source_images':len(ctx.data['images']),
        'total_source_captions':len(ctx.data['annotations']),
        'full_coverage':cnt['images']==len(ctx.data['images']) and cnt['captions']==len(ctx.data['annotations']),
        'counts':dict(cnt),'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'interpretation':'S5 calibrated diagnostic proxy; not ground-truth object presence; offline teacher fixed'})
    print('[DONE]',a.output,flush=True)

if __name__=='__main__':main()
