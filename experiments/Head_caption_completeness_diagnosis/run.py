"""S0--S6 independent, frozen-Teacher completeness benchmark.

Run from Talk2DINO root:
python -m experiments.Head_caption_completeness_diagnosis.run --limit 800 --device cuda
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import random
import sys
import torch
import torch.nn.functional as F

from experiments.Head_caption_completeness.core import ROOT, gallery
from .analysis import save_reports
from .core import comparison_mask,coverage,delta,drift_decomposition,dump_json,is_finite,strict_groups
from .perturbations import make_cases
from .text_evidence import TextScorer,clip_null,STRICT_EQUIVALENCE

ALL_METHODS=('S0','S1','S2','S3','S4','S5','S6')


def ramp(score,null,margin):
    return ((score-null)/margin).clamp(0,1)


class FrozenData:
    def __init__(self,a):
        # Depend on existing feature loaders/models, not a duplicate CLIP/DINO implementation.
        from experiments.Head_Match.common.teacher import TeacherBank
        self.a=a;self.device=torch.device(a.device)
        self.data=torch.load(a.base_train,map_location='cpu',weights_only=False)
        self.images=self.data['images'][:a.limit or None]
        self.groups=defaultdict(list)
        for ann in self.data['annotations']:self.groups[int(ann['image_id'])].append(ann)
        self.vocab=gallery(a.gallery)
        self.teacher=TeacherBank([a.teacher],a.config,device=a.device,batch_size=a.batch_size)
        self.clip_words=F.normalize(self.teacher.embed(self.vocab,'a photo of a {}').float().to(self.device),dim=-1)
        # ProjectionLayer MUST receive unnormalized raw CLIP text embeddings.
        raw_words=self.teacher.embed(self.vocab,'a photo of a {}')
        self.proj_words=self.teacher.project(raw_words).to(self.device)
        self.teacher.cache.clear()
        cached=torch.load(a.clip_images,map_location='cpu',weights_only=False)
        self.clip_images={int(i):F.normalize(torch.as_tensor(x).float(),dim=-1)
                          for i,x in zip(cached['ids'],cached['features'])}
        missing=[int(im['id']) for im in self.images if int(im['id']) not in self.clip_images]
        if missing:raise ValueError(f'CLIP image cache missing {len(missing)} images; e.g. {missing[:5]}. Re-run original prepare.py.')
        if len(self.vocab)<3:raise ValueError('Need at least 3 gallery concepts.')
        self.max_null,self.single_null,self.image_null=self.calibrate()
        self.text_null=clip_null(self.clip_words,a.text_null_samples,a.text_null_quantile,a.seed)
        extras=json.loads(a.aliases.read_text(encoding='utf-8')) if a.aliases else None
        self.text=TextScorer(self.teacher,self.vocab,self.clip_words,self.text_null,extras,
                             low=a.text_low,high=a.text_high,phrase_offset=a.phrase_offset,
                             phrase_margin=a.phrase_margin,unknown_low=a.unknown_low,
                             unknown_high=a.unknown_high,max_ngram=a.max_ngram,
                             max_phrases=a.max_phrases)
        self.strict_groups=strict_groups(self.vocab,STRICT_EQUIVALENCE)
        print(f'[DATA] images={len(self.images)} gallery={len(self.vocab)}',flush=True)

    @torch.no_grad()
    def calibrate(self):
        a=self.a
        rng=random.Random(a.seed)
        selected=rng.sample(self.images,min(len(self.images),a.calibration_images))
        if len(selected)<2:raise ValueError('Need at least 2 calibration images')
        heads=[];single=[];images=[]
        for im in selected:
            h=torch.as_tensor(im['disentangled_self_attn'],dtype=torch.float32)
            if h.shape != (12,768):raise ValueError(f"image={im['id']}, head shape {tuple(h.shape)} (expected 12x768)")
            h=F.normalize(h,dim=-1)
            heads.append(h)
            single.append(h[rng.randrange(h.shape[0])])
            images.append(self.clip_images[int(im['id'])])
        heads=torch.stack(heads).to(self.device)
        single=torch.stack(single).to(self.device)
        images=torch.stack(images).to(self.device)
        max_null=[];single_null=[];image_null=[]
        for i in range(0,len(self.vocab),a.gallery_chunk):
            p=self.proj_words[i:i+a.gallery_chunk]
            c=self.clip_words[i:i+a.gallery_chunk]
            head_score=torch.einsum('nhd,kd->nhk',heads,p)
            max_null.append(torch.quantile(head_score.amax(dim=1),a.null_quantile,dim=0))
            single_null.append(torch.quantile(single@p.T,a.null_quantile,dim=0))
            image_null.append(torch.quantile(images@c.T,a.null_quantile,dim=0))
        return torch.cat(max_null),torch.cat(single_null),torch.cat(image_null)

    @torch.no_grad()
    def evidence(self,image,ann):
        a=self.a
        head=F.normalize(torch.as_tensor(image['disentangled_self_attn'],device=self.device,dtype=torch.float32),dim=-1)
        raw=torch.as_tensor(ann['ann_feats'],device=self.device,dtype=torch.float32).flatten()
        if head.shape!=(12,768) or raw.shape!=(512,):
            raise ValueError(f"Image {image['id']}: expected head (12,768), ann_feats (512,), got {tuple(head.shape)},{tuple(raw.shape)}")
        # EXACT source max-score assignment: raw ann_feat -> frozen nonlinear Teacher -> cosine.
        projected=self.teacher.project(raw.unsqueeze(0))[0]
        h_scores=head@projected
        selected=int(h_scores.argmax().item())
        visual_sims=head@self.proj_words.T
        clip_image=self.clip_images[int(image['id'])].to(self.device)
        image_sim=clip_image@self.clip_words.T
        calibrated_max=ramp(visual_sims,self.max_null+a.head_offset,a.head_margin)
        calibrated_single=ramp(visual_sims,self.single_null+a.head_offset,a.head_margin)
        image_support=ramp(image_sim,self.image_null+a.image_offset,a.image_margin)
        fused=(a.head_weight*calibrated_max[selected]+(1-a.head_weight)*image_support).clamp(0,1)
        other=calibrated_single.clone();other[selected]=-1
        other_support=other.amax(dim=0)
        return dict(head_id=selected,head_score=float(h_scores[selected]),raw_feat=raw,
                    head_feat=head[selected],v_fused=fused.detach().cpu().tolist(),
                    v_head=calibrated_single[selected].detach().cpu().tolist(),
                    v_other=other_support.detach().cpu().tolist(),
                    v_image=image_support.detach().cpu().tolist())


def variant_source(method):
    return method if method in ('S0','S1','S2') else 'S3'


@torch.no_grad()
def evaluate_case(a,ctx,image,ann,cases,ev,raw_reencoded_cos):
    iid=int(image['id']);aid=int(ann['id']);raw=str(ann['caption'])
    entries=[]
    for case in cases:
        text=case['modified'];k=case['target'];kind=case['kind']
        # An appended concept beyond CLIP's 77-token window is NOT a valid intervention.
        if len(text)>a.max_caption_chars:
            continue
        if getattr(ctx.teacher,'clip',None) is not None:
            try:
                ctx.teacher.clip.tokenize(text,truncate=False)
            except RuntimeError:
                continue
        modified_clip=ctx.teacher.embed([text])[0].to(ctx.device)
        modified_proj=ctx.teacher.project(modified_clip.unsqueeze(0))[0]
        head_delta=float(ev['head_feat']@(modified_proj-ctx.teacher.project(ev['raw_feat'].unsqueeze(0))[0]))
        for method in a.methods:
            tm=variant_source(method)
            t_old,u_old=ctx.text.score(raw,tm,ev['raw_feat'] if method in ('S0','S1') else None)
            t_new,u_new=ctx.text.score(text,tm)
            v=ev['v_head'] if method in ('S5','S6') else ev['v_fused']
            mask=comparison_mask(v,t_old,t_new,u_old,u_new,k,a.relevant_visual,legacy=(method=='S0'))
            nvalid=sum(mask)
            grouping=ctx.strict_groups if method=='S6' else None
            before=coverage(v,t_old,mask,grouping)
            after=coverage(v,t_new,mask,grouping)
            da=delta(after['d_add'],before['d_add'])
            dr=delta(after['d_remove'],before['d_remove'])
            target_known=(0<=k<len(v) and mask[k] and not u_old[k] and not u_new[k])
            if target_known and grouping is not None:
                # A strict synonym group with one UNKNOWN member is excluded.
                target_known=any(k in g and all(mask[x] for x in g) for g in grouping)
            target_delta=float(t_new[k]-t_old[k]) if target_known else None
            leakage=drift_decomposition(v,t_old,t_new,mask,k,grouping)
            off_diff=[abs(float(t_new[i])-float(t_old[i])) for i in range(len(v)) if mask[i] and i!=k]
            off_mean=sum(off_diff)/len(off_diff) if off_diff else None
            # The denominator of D_remove changes on additions: aggregate monotonicity
            # is only mathematically predicted when (1-V_k)>D_remove_old.
            remove_monotonic=(target_known and before['d_remove'] is not None and
                              (1-float(v[k]))>before['d_remove']+1e-8) if k>=0 else False
            success=None
            if nvalid>0 and is_finite(da) and is_finite(dr):
                if kind=='delete_supported' and target_known:success=da>1e-6
                elif kind=='add_high_evidence' and target_known:success=da< -1e-6
                elif kind=='add_low_evidence' and target_known and remove_monotonic:success=dr>1e-6
                elif kind=='synonym' and target_known:success=abs(target_delta)<=a.stable_target
                elif kind=='neutral_prefix':success=abs(da)<=a.stable_d and off_mean is not None and off_mean<=a.stable_off
            pass_cf=False
            if is_finite(da) and is_finite(dr):
                if kind=='delete_supported':pass_cf=target_known and target_delta<=-a.cf_target and (off_mean or 0)<=a.cf_off
                elif kind in ('add_high_evidence','add_low_evidence'):
                    pass_cf=target_known and target_delta>=a.cf_target and (off_mean or 0)<=a.cf_off
                elif kind=='synonym':pass_cf=target_known and abs(target_delta)<=a.stable_target and (off_mean or 0)<=a.cf_off
                elif kind=='neutral_prefix':pass_cf=(off_mean or 0)<=a.stable_off and abs(da)<=a.stable_d
            diffs=[(i,float(t_new[i])-float(t_old[i])) for i in range(len(v)) if mask[i] and i!=k]
            diffs.sort(key=lambda x:abs(x[1]),reverse=True)
            entry={
                'method':method,'image_id':iid,'annotation_id':aid,'head_fixed':ev['head_id'],
                'head_score':ev['head_score'],'kind':kind,'control_label':case['control_label'],
                'original_caption':raw,'modified_caption':text,
                'target_concept':ctx.vocab[k] if k>=0 else None,'target_index':k,
                'target_known':bool(target_known),'target_text_delta':target_delta,
                'target_visual_head':ev['v_head'][k] if k>=0 else None,
                'target_visual_image':ev['v_image'][k] if k>=0 else None,
                'target_visual_other_heads':ev['v_other'][k] if k>=0 else None,
                'target_visual_used':v[k] if k>=0 else None,
                'head_teacher_score_delta':head_delta,
                'raw_embedding_cos_cached_vs_online':raw_reencoded_cos,
                'comparable':is_finite(da) and is_finite(dr) and nvalid>0,
                'valid_count':nvalid,'valid_fraction':nvalid/len(v),
                'd_add_before':before['d_add'],'d_add_after':after['d_add'],
                'd_remove_before':before['d_remove'],'d_remove_after':after['d_remove'],
                'delta_add':da,'delta_remove':dr,
                'off_target_mean_abs_delta':off_mean,
                'target_d_add_contribution':leakage.get('target_d_add_contribution'),
                'off_target_d_add_contribution':leakage.get('off_target_d_add_contribution'),
                'off_target_absolute_contribution':leakage.get('off_target_absolute_contribution'),
                'leakage_ratio':leakage.get('leakage_ratio'),
                'attribution_available':leakage['defined'],
                'low_evidence_remove_monotonic_condition':bool(remove_monotonic),
                'direction_success':success,
                'counterfactual_pass':bool(pass_cf) if method=='S4' else None,
                'top_off_target_changes':[{'concept':ctx.vocab[i],'text_delta':d} for i,d in diffs[:a.audit_top]],
            }
            # Sanity: exact additive decomposition requires fixed V/valid and ungrouped aggregation.
            if leakage['defined'] and is_finite(da) and abs(leakage['sum_contribution']-da)>1e-5:
                raise AssertionError(f'Attribution mismatch: {leakage["sum_contribution"]} vs {da}')
            entries.append(entry)
    return entries


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--gallery',type=Path,default=ROOT/'experiments/local_cocu/MSCOCO14_train_gallery.json')
    p.add_argument('--teacher',type=Path,default=ROOT/'weights/vitb_mlp_infonce.pth')
    p.add_argument('--config',type=Path,default=ROOT/'configs/vitb_mlp_infonce.yaml')
    p.add_argument('--clip-images',type=Path,default=ROOT/'outputs/Head_caption_completeness/clip_images.pth')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/Head_caption_completeness_diagnosis/run800')
    p.add_argument('--aliases',type=Path,help='Optional JSON mapping canonical concept -> list of aliases')
    p.add_argument('--methods',default=','.join(ALL_METHODS))
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=800,help='Number of images; 0 means all')
    p.add_argument('--captions-per-image',type=int,default=1,help='1 fast; 5 covers every caption')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--gallery-chunk',type=int,default=512)
    p.add_argument('--calibration-images',type=int,default=256)
    p.add_argument('--null-quantile',type=float,default=.90)
    p.add_argument('--head-offset',type=float,default=.01)
    p.add_argument('--head-margin',type=float,default=.10)
    p.add_argument('--image-offset',type=float,default=.01)
    p.add_argument('--image-margin',type=float,default=.10)
    p.add_argument('--head-weight',type=float,default=.8)
    p.add_argument('--text-low',type=float,default=.44)
    p.add_argument('--text-high',type=float,default=.60)
    p.add_argument('--text-null-samples',type=int,default=256)
    p.add_argument('--text-null-quantile',type=float,default=.95)
    p.add_argument('--phrase-offset',type=float,default=.015)
    p.add_argument('--phrase-margin',type=float,default=.16)
    p.add_argument('--unknown-low',type=float,default=.15)
    p.add_argument('--unknown-high',type=float,default=.40)
    p.add_argument('--max-ngram',type=int,default=4)
    p.add_argument('--max-phrases',type=int,default=72)
    p.add_argument('--max-caption-chars',type=int,default=320)
    p.add_argument('--relevant-visual',type=float,default=.25)
    p.add_argument('--min-high',type=float,default=.60)
    p.add_argument('--min-image',type=float,default=.30)
    p.add_argument('--max-low',type=float,default=.12)
    p.add_argument('--max-other',type=float,default=.30)
    p.add_argument('--max-low-image',type=float,default=.18)
    p.add_argument('--stable-target',type=float,default=.15)
    p.add_argument('--stable-off',type=float,default=.10)
    p.add_argument('--stable-d',type=float,default=.05)
    p.add_argument('--cf-target',type=float,default=.20)
    p.add_argument('--cf-off',type=float,default=.15)
    p.add_argument('--audit-top',type=int,default=8)
    p.add_argument('--bootstrap',type=int,default=300)
    p.add_argument('--min-cache-online-cos',type=float,default=.995,
                   help='Warning threshold for cached vs re-encoded raw caption.')
    p.add_argument('--strict-encoder-consistency',action='store_true',help='Raise on inconsistent caption embeddings')
    return p


def main(argv=None):
    a=build_parser().parse_args(argv)
    for key in ['base_train','gallery','teacher','config','clip_images']:
        val=getattr(a,key).expanduser().resolve()
        if not val.is_file():raise FileNotFoundError(f'--{key.replace("_","-")} missing: {val}')
        setattr(a,key,val)
    if a.aliases:
        a.aliases=a.aliases.expanduser().resolve()
        if not a.aliases.is_file():raise FileNotFoundError(a.aliases)
    if a.limit<0 or a.captions_per_image not in range(1,6):raise ValueError('Bad limit or captions-per-image')
    if a.calibration_images<2:raise ValueError('calibration-images must be >= 2')
    a.methods=[s.strip().upper() for s in a.methods.split(',') if s.strip()]
    if not a.methods or any(m not in ALL_METHODS for m in a.methods):raise ValueError('Unknown methods: '+str(a.methods))
    if len(set(a.methods))!=len(a.methods):raise ValueError('Duplicate methods')
    torch.set_grad_enabled(False)
    random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    ctx=FrozenData(a)
    a.output.mkdir(parents=True,exist_ok=True)
    counts=Counter();encoder_cos=[];records=[]
    with (a.output/'records.jsonl').open('w',encoding='utf-8') as file:
        for index,image in enumerate(ctx.images):
            iid=int(image['id']);anns=ctx.groups[iid][:a.captions_per_image]
            if not anns:continue
            for ann in anns:
                ev=ctx.evidence(image,ann)
                old_online=ctx.teacher.embed([str(ann['caption'])])[0].to(ctx.device)
                similarity=float(F.cosine_similarity(F.normalize(ev['raw_feat'],dim=-1),
                                                      F.normalize(old_online,dim=-1),dim=0))
                encoder_cos.append(similarity)
                if similarity<a.min_cache_online_cos:
                    counts['encoder_mismatch']+=1
                    if a.strict_encoder_consistency:
                        raise RuntimeError(f'Cached/online CLIP drift at annotation_id={ann["id"]}: cosine={similarity:.6f}')
                cases=make_cases(str(ann['caption']),ctx.vocab,ctx.text.alias_index,
                                 ev['v_head'],ev['v_image'],ev['v_other'],
                                 a.min_high,a.min_image,a.max_low,a.max_other,a.max_low_image)
                counts['captions_processed']+=1
                for kind in set(x['kind'] for x in cases):counts['intervention_'+kind]+=1
                rows=evaluate_case(a,ctx,image,ann,cases,ev,similarity)
                for r in rows:
                    file.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
                    records.append(r)
                ctx.text.clear();ctx.teacher.cache.clear()
            if (index+1)%25==0 or index==0:
                print(f'[DIAG] {index+1}/{len(ctx.images)} images, {len(records)} method-cases',flush=True)
    counts['images_processed']=len(ctx.images)
    meta=dict(run_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='output'},
              output=str(a.output),counts=dict(counts),
              raw_cache_vs_online_min_cos=min(encoder_cos) if encoder_cos else None,
              raw_cache_vs_online_mean_cos=sum(encoder_cos)/len(encoder_cos) if encoder_cos else None,
              gallery_size=len(ctx.vocab),groups_size=len(ctx.strict_groups),
              head_assignment='frozen Teacher original argmax, never reassign',
              visual_s5_s6='selected head calibrated by random single-head null; no image mixing',
              output_note='synthetic interventions / proxy diagnostics only; no model training')
    dump_json(a.output/'metadata.json',meta)
    table=save_reports(a.output,records,meta,a.bootstrap)
    print('[SAVED]',a.output,'rows=',len(records),'summaries=',len(table),flush=True)
    if counts['encoder_mismatch']:
        print('[WARN] Cached raw embeddings and re-encoded CLIP text differ in',
              counts['encoder_mismatch'],'captions; inspect metadata and consider --strict-encoder-consistency',flush=True)

if __name__=='__main__':main()
