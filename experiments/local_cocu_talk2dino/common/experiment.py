"""Independent A-G discovery strategies. H delegates repeatedly to A in runner."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from .algorithms import (top_concepts,head_score,uncovered_heads,consistency_score,
                         counterfactual_score,view_agreement,bbox_from_map)
from .data import novel, already_mentioned, audit_record


def mine(scheme, ids, heads, captions, gallery, teacher, args, neighbors=None,
         neighbor_details=None, cls_neighbors=None, image_paths=None):
    """Yields serializable image-level full audit rows; no training code present."""
    from .vision import Vision
    if scheme not in 'ABCDEFG' or len(scheme)!=1: raise ValueError(scheme)
    z=teacher.encode(gallery)
    concept_index={n:i for i,n in enumerate(gallery)}
    vision=Vision(device=args.device) if scheme in 'CDG' else None
    try:
        for pos,iid in enumerate(ids):
            original=captions[iid][:5]
            h=torch.as_tensor(heads[iid]).float().to(teacher.device)
            diag={}
            if scheme in 'ACDGH':
                rows=top_concepts(gallery,h,z,original,k=args.pool_k,
                    per_head_k=args.per_head_k,threshold=args.threshold)
            elif scheme in 'BEF':
                from .candidates import from_neighbors
                source=neighbors[str(iid)] if neighbors is not None else []
                choices=from_neighbors(source,captions,gallery,cap=args.candidate_cap)
                choices=novel(choices,original)
                zz=teacher.encode(choices)
                rows=top_concepts(choices,h,zz,original,k=args.pool_k,
                    per_head_k=max(args.per_head_k,args.pool_k),threshold=args.threshold) if choices else []
                diag['neighbor_ids']=source[:args.neighbor_topk]
            if scheme=='C':
                caption_z=teacher.encode_captions(original)
                low,q=uncovered_heads(h,caption_z,args.uncovered_heads)
                vis,attn,_=vision.get(vision.image(image_paths[iid]))
                concentration=vision.attention_strength(attn)
                valid=[j for j in low if concentration[j]>=args.attention_min_concentration]
                diag.update({'caption_coverage':q,'uncovered_heads':low,
                             'attention_concentration':concentration,'valid_heads':valid})
                # Low caption coverage without valid spatial response is NOT novelty.
                rows=top_concepts(gallery,h,z,original,k=args.pool_k,per_head_k=args.per_head_k,
                                  threshold=args.threshold,permitted_heads=valid)
            if scheme=='D':
                _,attn,transformed=vision.get(vision.image(image_paths[iid]))
                scored=[]; region_cache={}
                for item in rows:
                    head=item['best_head']
                    if head not in region_cache:
                        bbox=bbox_from_map(attn[head],quantile=args.crop_quantile)
                        crop=transformed.crop(bbox)
                        local_head,_,_=vision.get(crop)
                        region_cache[head]=(bbox,local_head)
                    bbox,local_head=region_cache[head]
                    vec=z[concept_index[item['concept']]].unsqueeze(0)
                    local=float(head_score(local_head.to(teacher.device),vec)[0][0])
                    # Require local evidence as well as original-image evidence.
                    score=args.global_ratio*item['score']+(1-args.global_ratio)*local
                    if local>=args.local_threshold:
                        scored.append({**item,'score':score,'global_score':item['score'],
                                       'local_score':local,'bbox_xyxy_448':list(bbox)})
                rows=scored
                diag['crop_note']='bbox refers to resized, center-cropped 448px image, NOT original pixels'
            if scheme=='E':
                matches=neighbor_details.get(str(iid),[]) if neighbor_details else []
                matched=[]
                for m in matches[:args.support_neighbors]:
                    peer=int(m['image_id']); head_id=int(m['neighbor_head'])
                    if peer in heads:
                        matched.append((peer,head_id))
                scored=[]
                if matched and rows:
                    evidence=torch.stack([torch.as_tensor(heads[p])[hd] for p,hd in matched])
                    vectors=teacher.encode([r['concept'] for r in rows])
                    # One matrix operation, not a per-concept × per-neighbor GPU call.
                    support_matrix=teacher.sim(evidence,vectors).cpu()
                else:
                    support_matrix=torch.empty((0,len(rows)))
                for j,item in enumerate(rows):
                    support_scores=[float(x) for x in support_matrix[:,j]]
                    val,count,avg,std=consistency_score(item['score'],support_scores,
                        min_support=args.min_support,gap=args.support_margin,
                        rare_high=args.rare_high)
                    if val>-1e8:
                        support_ids=[int(matched[k][0]) for k,s in enumerate(support_scores)
                            if s>=item['score']-args.support_margin]
                        scored.append({**item,'score':val,'target_score':item['score'],
                                       'support_count':count,'support_mean':avg,'support_std':std,
                                       'evaluated_neighbors':len(support_scores),
                                       'support_image_ids':support_ids})
                rows=scored
            if scheme=='F':
                negs=(cls_neighbors or {}).get(str(iid),[])[:args.negative_topk]
                peers=[int(n) for n in negs if int(n) in heads and int(n)!=iid]
                scored=[]
                if peers and rows:
                    neghead=torch.stack([torch.as_tensor(heads[p]).float() for p in peers])
                    vectors=teacher.encode([r['concept'] for r in rows])
                    neg_matrix=teacher.sim(neghead.reshape(-1,neghead.shape[-1]),vectors)
                    neg_matrix=neg_matrix.reshape(len(peers),-1,len(rows)).max(dim=1).values.cpu()
                else: neg_matrix=torch.empty((0,len(rows)))
                for j,item in enumerate(rows):
                    negatives=sorted(map(float,neg_matrix[:,j]))[:max(1,len(peers)//2)]
                    new=counterfactual_score(item['score'],negatives,args.counterfactual_lambda)
                    scored.append({**item,'score':new,'target_score':item['score'],
                                   'negative_mean':sum(negatives)/len(negatives) if negatives else None,
                                   'negative_count':len(negatives)})
                rows=scored
            if scheme=='G':
                view_heads=vision.views(image_paths[iid])
                scored=[]
                if rows:
                    vectors=teacher.encode([r['concept'] for r in rows])
                    stack=torch.stack([v.float() for v in view_heads])
                    view_sim=teacher.sim(stack.reshape(-1,stack.shape[-1]),vectors)
                    view_scores=view_sim.reshape(len(view_heads),-1,len(rows)).max(dim=1).values.cpu()
                else: view_scores=torch.empty((len(view_heads),0))
                for j,item in enumerate(rows):
                    scores=[float(x) for x in view_scores[:,j]]
                    val=view_agreement(scores,min_views=args.min_views,
                                      threshold=args.view_threshold,variability=args.view_variability)
                    if val is not None:
                        scored.append({**item,'score':val,'original_score':item['score'],
                                       'view_scores':scores,'visible_views':[0,1,2]})
                rows=scored
            rows=sorted(rows,key=lambda x:(-x['score'],x['concept']))[:args.max_concepts]
            yield str(iid),audit_record(iid,original,rows,scheme,diag)
            if (pos+1)%100==0:print(f'[Discovery {scheme}] {pos+1}/{len(ids)}',flush=True)
    finally:
        if vision:vision.close()
