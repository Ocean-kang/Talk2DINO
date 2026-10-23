"""Offline Teacher-guided 5-caption refinement, not an alternative student loss."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import torch
from tqdm import tqdm
from .core import assignments,scored_concepts,consensus,compose,mentioned,unit,shift,clean_vocabulary
from .io import ROOT,corpus,candidates_from_json,gallery_from_json,write_json
from .teacher import TeacherBank


def global_top(heads,projected,words,per_head):
    if not words:return []
    sim=unit(heads) @ projected.T
    take=min(max(0,per_head),len(words))
    if not take:return []
    ids=sim.topk(take,dim=1).indices.flatten().unique().tolist()
    return [words[i] for i in ids]


def _evaluate_candidate(bank,raw_clip,raw_project,heads,p,candidate,new_caption,which=0):
    """Counterfactual delta on SAME weighted head mixture; no head switching reward."""
    feature=bank.embed([new_caption])[0]
    v=bank.project(feature,which).flatten()
    old=float(p @ (unit(heads) @ raw_project.flatten()))
    new=float(p @ (unit(heads) @ v))
    return new-old


def pick_for_caption(variant,bank,heads,raw_caption,raw_clip,concepts,clip_candidates,projected_candidates,
                     args,teacher_indices):
    """One caption: raw Head weights -> accepted novel concepts -> optional new Head weights."""
    primary=teacher_indices[-1]
    raw_project=bank.project(raw_clip,primary).flatten()
    p,similarities=assignments(heads,raw_project,args.temperature,args.assignment,args.top_heads)
    original_p=p.clone()
    cand_projected=projected_candidates[primary]
    def score(current_p):
        return scored_concepts(heads,current_p,cand_projected,raw_clip,clip_candidates,concepts,
            caption=raw_caption,coverage_threshold=args.coverage_threshold,covered_sim=args.covered_sim,
            min_visual=args.min_visual,novelty_max=args.novelty_max,max_concepts=args.proposal_count)
    initial=score(p)
    eligible=initial['ranked']
    allowed=None; votes={}; needed=1
    if variant=='D':
        # Round 1 is a clearly labeled single-teacher warmup, NOT a consensus round.
        individual=[]
        for t in teacher_indices:
            tproj=bank.project(raw_clip,t).flatten()
            tp,_=assignments(heads,tproj,args.temperature,args.assignment,args.top_heads)
            scores=(tp @ (unit(heads) @ projected_candidates[t].T)).tolist()
            individual.append(dict(zip(concepts,scores)))
        needed=min(args.min_votes,len(teacher_indices))
        votes=consensus(individual,args.min_visual,needed)
        allowed=set(votes)
    decisions=[];tried=set(); attempts=0
    while len(decisions)<args.max_concepts and initial['gate']!='covered':
        pending=[x for x in eligible if x['concept'] not in tried and
                 (allowed is None or x['concept'] in allowed)]
        if not pending:break
        row=pending[0];attempts+=1
        if attempts>args.proposal_count*2:break
        name=row['concept'];tried.add(name)
        if not row['novel'] or row['head_support']<args.min_visual or mentioned(name,raw_caption):continue
        additions=[x['concept'] for x in decisions]+[name]
        new_text=compose(raw_caption,additions)
        if variant in ('B','C','D'):
            delta=_evaluate_candidate(bank,raw_clip,raw_project,heads,p,name,new_text,primary)
            if delta<args.min_delta:continue
        else:delta=None
        row=dict(row)
        row['counterfactual_delta']=None if delta is None else round(delta,6)
        if variant=='C':
            # Alternating refinement within the SAME frozen Teacher round.
            upd=bank.project(bank.embed([new_text])[0],primary).flatten()
            next_p,_=assignments(heads,upd,args.temperature,args.assignment,args.top_heads)
            movement=shift(original_p,next_p)
            if movement>args.max_head_shift:continue
            row['head_shift']=round(movement,6)
            p=next_p
            eligible=score(p)['ranked']  # Recompute AND rerank remaining concepts.
        decisions.append(row)
    return {'raw_caption':raw_caption,'augmented_caption':compose(raw_caption,[r['concept'] for r in decisions]),
            'head_weights_raw':[round(float(x),6) for x in original_p.tolist()],
            'head_weights_final':[round(float(x),6) for x in p.tolist()],
            'head_argmax_raw':int(similarities.argmax()),'coverage':initial['coverage'],
            'coverage_gate':initial['gate'],'added':decisions,
            'teacher_votes':votes,'teacher_count':len(teacher_indices),
            'votes_required':needed,'candidates_considered':attempts}


def mine(args,round_idx,teacher_checkpoints,output):
    """Write image_id/ann_id traceable decisions and metadata for original training PTH."""
    _,images=corpus(args.base_train)
    if args.limit:images=images[:args.limit]
    source=candidates_from_json(args.concepts) if args.concepts else {}
    gallery=gallery_from_json(args.gallery) if args.gallery else []
    if not source and not gallery:raise ValueError('Need --concepts or --gallery')
    all_words=clean_vocabulary([w for lst in source.values() for w in lst]+gallery)
    if not all_words:raise ValueError('Empty candidate vocabulary')
    print('[VOCAB]',len(all_words),'source images',len(source),'gallery',len(gallery),flush=True)
    bank=TeacherBank(teacher_checkpoints,args.config,args.device,args.batch_size)
    # Shared CLIP encoder: its weights stay fixed; all teachers have independent Projector.
    all_clip=bank.embed(all_words,'a photo of a {}')
    word_index={w:i for i,w in enumerate(all_words)}
    projected=bank.project_all(all_clip)
    # Gallery projection on all images requires no extra DINO forward pass.
    gallery_idx=torch.tensor([word_index[w] for w in gallery],device=bank.device,dtype=torch.long)
    gal_proj=projected[-1][gallery_idx] if gallery else None
    audit={};counter=Counter()
    progress=tqdm(images,desc=f'Head_Match {args.variant} round {round_idx}')
    for image_id,heads_cpu,anns in progress:
        heads=unit(heads_cpu.to(bank.device))
        local=source.get(image_id,[])
        dynamic=global_top(heads,gal_proj,gallery,args.gallery_per_head) if gallery else []
        candidates=clean_vocabulary(local[:args.local_candidate_cap]+dynamic)
        if len(candidates)>args.max_candidates:
            # Candidate priority: existing Local-CoCu results first, then gallery discoveries.
            candidates=candidates[:args.max_candidates]
        if not candidates:raise ValueError(f'No candidates for image {image_id}')
        ind=torch.tensor([word_index[w] for w in candidates],device=bank.device,dtype=torch.long)
        clip_candidates=all_clip[ind.cpu()].to(bank.device)
        concept_projected=[x[ind] for x in projected]
        captions=[]
        for ann in anns:
            feat=torch.as_tensor(ann['ann_feats']).flatten().to(bank.device).float()
            row=pick_for_caption(args.variant,bank,heads,str(ann['caption']),feat,candidates,
                                 clip_candidates,concept_projected,args,list(range(len(bank.projectors))))
            row['annotation_id']=int(ann['id'])
            captions.append(row)
            counter['captions_total']+=1
            counter['captions_modified']+=bool(row['added'])
            counter['concepts_added']+=len(row['added'])
        audit[str(image_id)]={'image_id':image_id,'round':round_idx,'variant':args.variant,
                              'teacher_weights':[str(x) for x in teacher_checkpoints],
                              'candidates_count':len(candidates),'captions':captions}
        counter['images']+=1
        if args.save_every and len(audit)%args.save_every==0:
            write_json(output.with_name('selection_partial.json'),{'meta':{'round':round_idx,'partial':True},'images':audit})
    meta={'round':round_idx,'variant':args.variant,'teacher_weights':[str(x) for x in teacher_checkpoints],
          'source_concepts':[str(x) for x in args.concepts],'gallery':str(args.gallery),
          'base_train':str(args.base_train),'num_teacher':len(bank.projectors),
          'counts':dict(counter),'args':{k:([str(z) for z in v] if isinstance(v,list) else str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}}
    write_json(output,{'meta':meta,'images':audit})
    write_json(output.parent/'selection_stats.json',meta)
    print('[SAVED]',output,dict(counter),flush=True)
