"""Idea B: local gallery retrieval, evidence screening, compact frozen region cache.

No masks or diagnostic manifests are accepted. One frozen teacher, two image views.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .common import (gallery, lexical, read, write, digest, object_digest, normalize,
                     top_mask, iou, pool_region, selection_pass)
from .runtime import Models, model_args, captions_data, image_path, save_pth, load_pth


def score_candidates(model, features, flipped, candidates, projected, raw_text, covered, config):
    """Per-concept confusable negatives; suppress lexical/CLIP aliases & likely positives.

    Negatives are conservative teacher pseudo-negatives, not verified absence labels.
    Abstain if no usable negative. All metrics available for independent audit.
    """
    names=config['gallery']; lex=lexical()
    patches,attn=features['patches'],features['attention']
    sim=projected[candidates]@normalize(patches).T
    flip_sim=projected[candidates]@normalize(flipped['patches']).T
    head_sims=projected[candidates]@normalize(features['heads']).T
    max_head=head_sims.max(axis=1)
    raw_unit=normalize(raw_text[candidates])
    clip_pair=raw_unit@raw_unit.T
    words=[lex.norm_tokens(names[g]) for g in candidates]
    def contains(a,b):
        return any(b[i:i+len(a)]==a for i in range(len(b)-len(a)+1))
    records=[]; side=model.size//14
    for j,gid in enumerate(candidates):
        # Alternatives must not be known positives, aliases or strong suspected positives.
        negatives=[]
        for k,other in enumerate(candidates):
            if other==gid or other in covered or max_head[k]>=config['negative_presence_max']:
                continue
            if contains(words[j],words[k]) or contains(words[k],words[j]):
                continue
            similarity=float(clip_pair[j,k])
            if similarity>=config['alias_cosine']:
                continue
            negatives.append((similarity,k,other))
        negatives=sorted(negatives,reverse=True)[:config['negatives']]
        if not negatives:
            continue
        ki=[k for _,k,_ in negatives]
        margin=sim[j]-sim[ki].max(axis=0)
        flip_margin=flip_sim[j]-flip_sim[ki].max(axis=0)
        evidence=1/(1+np.exp(-np.clip(margin/config['temperature'],-30,30)))
        q=attn@evidence
        best=int(q.argmax())
        positive_mass=float(attn[best,margin>0].sum())
        count=max(1,int(np.ceil(len(margin)*config['area_fraction'])))
        # Include ties to preserve flip equivariance on flat/highly quantized maps.
        mask=(margin>=np.sort(margin)[-count])&(margin>0)
        mask_f=(flip_margin>=np.sort(flip_margin)[-count])&(flip_margin>0)
        mask_f=mask_f.reshape(side,side)[:,::-1].ravel()
        stability=iou(mask,mask_f,np.ones(len(mask),bool)) or 0.
        supported=[h for h in range(len(q)) if q[h]>=max(config['q_min'],float(q.max())-config['multi_q_delta'])
                   and float(attn[h,margin>0].sum())>=config['mass_min']]
        region=pool_region(patches,attn,np.maximum(margin,0),[best])
        multi=pool_region(patches,attn,np.maximum(margin,0),supported) if supported else None
        # Adapt only when several heads have near-best support and union region is
        # still discriminative. Fixed backbone, no new module, no GT-based selection.
        adaptive=region
        if region is not None and multi is not None and len(supported)>1:
            if float(multi@projected[gid]) >= float(region@projected[gid])-config['multi_score_tolerance']:
                adaptive=multi
        r={'gallery_id':int(gid),'concept':names[gid],'covered':gid in covered,
           'q':float(q[best]),'positive_mass':positive_mass,'flip_iou':float(stability),
           'head_score':float(max_head[j]),'selected_head':best,'support_heads':supported,
           'negative_ids':[int(x[2]) for x in negatives], 'region':region,'adaptive_region':adaptive,
           'head_region':normalize(features['heads'][int(head_sims[j].argmax())]),
           'head_rank_score':float(max_head[j])}
        r['accepted']=bool(region is not None and selection_pass(r,config))
        records.append(r)
    return sorted(records,key=lambda r:(-r['head_rank_score'],r['gallery_id']))


def export_panels(out,iid,features,records,projected,size):
    from .diagnose import overlay
    from PIL import ImageDraw
    folder=out/'audit_panels';folder.mkdir(exist_ok=True)
    crop=features['crop'];side=size//14;patches=normalize(features['patches'])
    for r in records:
        gid=r['gallery_id'];head=r['selected_head']
        sim=projected[gid]@patches.T
        neg=(projected[r['negative_ids']]@patches.T).max(axis=0)
        weight=features['attention'][head]*np.maximum(sim-neg,0)
        panels=[crop,overlay(crop,features['attention'][head].reshape(side,side)),
                overlay(crop,sim.reshape(side,side)),overlay(crop,weight.reshape(side,side))]
        canvas=Image.new('RGB',(896,260),'white');draw=ImageDraw.Draw(canvas)
        for j,(panel,label) in enumerate(zip(panels,['DINO crop','Selected head','Concept cosine','Training region weight'])):
            canvas.paste(panel.resize((224,224)),(224*j,0));draw.text((224*j+3,230),label,fill='black')
        canvas.save(folder/(str(iid)+'_'+str(gid)+'.jpg'))


def defaults():
    # Starting operating points only. Tune on a separate calibration split, then freeze.
    return {'q_min':.55,'mass_min':.6,'flip_min':.35,'head_min':.2,
            'negative_presence_max':.25,'alias_cosine':.92,'negatives':16,
            'temperature':.07,'area_fraction':.2,'multi_q_delta':.02,'multi_score_tolerance':.01}


def main():
    p=argparse.ArgumentParser(description=__doc__);model_args(p)
    p.add_argument('--captions',required=True,help='Original caption JSON or original train.pth')
    p.add_argument('--gallery',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--selection-config',help='Frozen JSON thresholds from separate calibration')
    p.add_argument('--feature-check-images',type=int,default=3,help='Verify existing head/text features against local models')
    p.add_argument('--concept-allowlist',help='Optional one-per-line semantic-type allowlist; intersects existing gallery')
    p.add_argument('--candidate-file',help='Optional existing concepts_by_image.json; values list[str]')
    p.add_argument('--limit',type=int,default=0,help='0 = full source; not silently subset to 5k')
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--top-per-head',type=int,default=8)
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--audit-panels',action='store_true',help='Save candidate/head/dense/region panels for independent grounding audit')
    p.add_argument('--recovery',action='store_true',help='Diagnostic-only leave-one-concept-out coverage; cannot train')
    p.add_argument('--calibration-ids',help='Optional newline IDs: restrict preparation to calibration split')
    a=p.parse_args()
    if min(a.top_per_head,a.top_k)<=0:raise ValueError('top sizes must be positive')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    config=defaults()
    if a.selection_config:config.update(read(a.selection_config))
    names=gallery(a.gallery);lex=lexical()
    if a.concept_allowlist:
        allow={' '.join(lex.tokenize(x)) for x in Path(a.concept_allowlist).read_text().splitlines() if x.strip()}
        names=[n for n in names if n in allow]
        if len(names)<3:raise ValueError('Allowlist/gallery intersection too small')
    config['gallery']=names
    images,captions=captions_data(a.captions)
    image_ids=sorted(captions)
    if a.calibration_ids:
        allow={int(x) for x in Path(a.calibration_ids).read_text().split()}
        image_ids=[i for i in image_ids if i in allow]
    image_ids=image_ids[a.offset:(a.offset+a.limit) if a.limit else None]
    if not image_ids:raise ValueError('No images selected')
    model=Models(a)
    source={'models':model.metadata,'captions_sha256':digest(a.captions),'gallery_sha256':digest(a.gallery),
            'selection':config,'image_ids':image_ids,'candidate_sha256':digest(a.candidate_file) if a.candidate_file else None,
            'recovery_only':a.recovery,'top_k':a.top_k,'top_per_head':a.top_per_head,
            'feature_checks_requested':a.feature_check_images,'audit_panels':a.audit_panels,
            'code_sha256':{n:digest(Path(__file__).parent/n) for n in ('prepare.py','runtime.py','common.py')}}
    signature=object_digest(source)
    if (out/'source.json').exists():
        if read(out/'source.json')['signature']!=signature:
            raise ValueError('Output belongs to different sources/settings; use a new output directory')
    else:write(out/'source.json',{'signature':signature,**source})
    cache=out/'text_bank.pth'
    if cache.exists():
        bank=load_pth(cache)
        if bank['names']!=names or bank['model']!=model.metadata:raise ValueError('Text cache source mismatch')
        raw=np.asarray(bank['raw']);projected=np.asarray(bank['projected'])
    else:
        raw=model.encode(['a photo of '+c for c in names]);projected=model.project(raw)
        save_pth(cache,{'names':names,'raw':raw,'projected':projected,'model':model.metadata})
    existing=read(a.candidate_file) if a.candidate_file else None
    name_index={n:i for i,n in enumerate(names)}
    maxlen=max(len(lex.norm_tokens(n)) for n in names)
    norm_lookup={}
    for i,n in enumerate(names):norm_lookup.setdefault(lex.norm_tokens(n),set()).add(i)
    # A deterministic vocabulary subset provides potential visual-absence negatives;
    # every one is scored on the image before use. No random new concepts/models.
    distractors=np.linspace(0,len(names)-1,min(96,len(names)),dtype=int).tolist()
    records_dir=out/'records';records_dir.mkdir(exist_ok=True)
    stats={'images':len(image_ids),'selected':0,'no_selected':0}
    import torch
    projected_gpu=torch.as_tensor(projected,device=a.device)
    checks=[]
    concepts_by_image={}
    audit_handle=(out/'selection_audit.jsonl').open('w',encoding='utf-8')
    for pos,iid in enumerate(image_ids):
        target=records_dir/(str(iid)+'.pth')
        feature_check=None
        if pos<a.feature_check_images and 'disentangled_self_attn' in images[iid]:
            with Image.open(image_path(a.image_root,images[iid])) as im:check=model.extract(im)
            cached=np.asarray(images[iid]['disentangled_self_attn'],dtype=np.float32)
            if cached.shape!=(12,768):raise ValueError('Bad existing head tensor')
            head_cos=np.sum(normalize(cached)*normalize(check['heads']),axis=-1)
            ann=captions[iid][0]
            if 'ann_feats' not in ann:raise ValueError('Original PTH lacks ann_feats')
            raw_cap=model.encode([ann['caption']])[0]
            text_cos=float(normalize(raw_cap)@normalize(np.asarray(ann['ann_feats'],dtype=np.float32)))
            feature_check={'image_id':iid,'min_head_cosine':float(head_cos.min()),'raw_caption_cosine':text_cos}
            if head_cos.min()<.999 or text_cos<.999:
                raise ValueError('Existing features differ from local model/448 preprocessing: '+str(feature_check))
            checks.append(feature_check)
        if target.exists():
            rec=load_pth(target)
            if rec['signature']!=signature:raise ValueError('Stale image shard')
        else:
            caps=[ann['caption'] for ann in captions[iid]]
            # Trie scan reused from previous noun experiment: deterministic token matching.
            # Scan caption ngrams to avoid O(images*gallery) string search.
            tokenized=[lex.norm_tokens(c) for c in caps]
            covered=set()
            for words in tokenized:
                for start in range(len(words)):
                    for length in range(1,min(maxlen,len(words)-start)+1):
                        covered.update(norm_lookup.get(words[start:start+length],set()))
            hidden=None
            if a.recovery and covered:
                # Deterministic held-out concept; remove lexical alias equivalents too.
                chosen=sorted(covered)[iid%len(covered)];hidden=chosen
                covered={g for g in covered if not (lex.in_raw_captions(names[g],[names[chosen]])
                                                        or lex.in_raw_captions(names[chosen],[names[g]]))}
            with Image.open(image_path(a.image_root,images[iid])) as im:
                feature=model.extract(im);flipped=model.extract(im,flip=True)
            if existing is not None:
                if str(iid) not in existing:raise KeyError('Candidate file missing image '+str(iid))
                arr=existing[str(iid)]
                if not isinstance(arr,list) or not all(isinstance(x,str) for x in arr):
                    raise ValueError('Candidate file requires image_id -> list[str]')
                candidate_ids=[name_index[' '.join(lex.tokenize(n))] for n in arr if ' '.join(lex.tokenize(n)) in name_index]
            else:
                with torch.no_grad():
                    hs=torch.as_tensor(normalize(feature['heads']),device=a.device)
                    k=min(a.top_per_head,len(names))
                    candidate_ids=(hs@projected_gpu.T).topk(k,dim=-1).indices.cpu().numpy().ravel().tolist()
            candidate_ids=sorted(set(candidate_ids))
            # Retrieval remains visual. Do not inject hidden/recovery target or covered
            # concepts into candidate IDs: target recall must include retrieval failures.
            scored_ids=sorted(set(candidate_ids)|set(distractors))
            scored=score_candidates(model,feature,flipped,scored_ids,projected,raw,covered,config)
            ranked=[r for r in scored if r['gallery_id'] in candidate_ids and not r['covered']]
            if a.audit_panels:
                export_panels(out,iid,feature,ranked,projected,model.size)
            coverage={'lexically_covered':len(covered),'scored_covered':sum(r['covered'] for r in scored),
                      'reliable_scored_covered':sum(r['covered'] and r['accepted'] for r in scored),
                      'reliable_missing':sum(not r['covered'] and r['accepted'] for r in scored)}
            selected=sorted([r for r in ranked if r['accepted']],key=lambda r:(-r['q'],-r['flip_iou'],r['gallery_id']))[:a.top_k]
            # E1 and E2 have exactly the same per-image added-text count.
            head_control=ranked[:len(selected)]
            def packed(r):
                return {k:v for k,v in r.items() if k not in ('region','adaptive_region','head_region')}
            rec={'image_id':iid,'signature':signature,'selected':selected,'head_control':head_control,
                 'known_covered_ids':sorted(covered),'hidden_id':hidden,'candidate_ids':candidate_ids,
                 'audit':[packed(r) for r in ranked], 'coverage_proxies':coverage,'raw_caption_count':len(caps)}
            save_pth(target,rec)
        concepts_by_image[str(iid)]=[r['concept'] for r in rec['selected']]
        audit_handle.write(json.dumps({'image_id':iid,'selected':concepts_by_image[str(iid)],
                                       'coverage_proxies':rec['coverage_proxies'],'candidates':rec['audit']},ensure_ascii=False)+'\n')
        stats['selected']+=len(rec['selected']);stats['no_selected']+=int(not rec['selected'])
        for key,val in rec['coverage_proxies'].items():stats[key]=stats.get(key,0)+val
        if pos%20==0:print('prepare',pos+1,'/',len(image_ids),stats,flush=True)
    audit_handle.close()
    write(out/'concepts_by_image.json',concepts_by_image)
    stats['coverage_gap_proxy']=stats.get('reliable_missing',0)/max(stats.get('reliable_missing',0)+stats.get('reliable_scored_covered',0),1)
    stats['grounding_gap_proxy']=1-stats.get('reliable_scored_covered',0)/max(stats.get('scored_covered',0),1) if stats.get('scored_covered',0) else None
    model.close();write(out/'summary.json',stats)
    if a.audit_panels:
        import html
        links=[]
        for image_id in image_ids:
            rec=load_pth(records_dir/(str(image_id)+'.pth'))
            for row in rec['audit']:
                name=str(image_id)+'_'+str(row['gallery_id'])+'.jpg'
                links.append('<p>'+html.escape(str(image_id)+' '+row['concept'])+'</p><img width=100% src="audit_panels/'+name+'">')
        (out/'audit.html').write_text('<meta charset="utf-8"><title>ECCA grounding audit</title>'+''.join(links),encoding='utf-8')
    write(out/'feature_checks.json',{'checked':checks,'count':len(checks),'threshold':.999})
    print('Completed compact region cache:',out)


if __name__=='__main__':main()
