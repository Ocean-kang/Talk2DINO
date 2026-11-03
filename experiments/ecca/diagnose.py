"""Idea A: co-register attention, raw dense class response, prediction and diagnostic GT."""
import argparse
import html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .common import read, write, digest, normalize, top_mask, iou, corr, grouped_bootstrap
from .runtime import Models, model_args, image_path

RESAMPLE = getattr(Image, 'Resampling', Image)

def overlay(crop, values):
    arr = np.asarray(values, np.float32)
    lo, hi = np.percentile(arr, [5,95])
    arr = np.clip((arr-lo)/max(float(hi-lo),1e-8),0,1)
    im = Image.fromarray((arr*255).astype('uint8')).resize(crop.size, RESAMPLE.BILINEAR)
    a = np.asarray(im,dtype=np.float32)/255
    color = np.zeros((*a.shape,3), np.float32)
    color[:,:,0], color[:,:,1] = 255, 255*(1-a)
    alpha = .65*a[:,:,None]
    return Image.fromarray(((1-alpha)*np.asarray(crop)+alpha*color).astype('uint8'))


def association(rows, repetitions=1000):
    use = [r for r in rows if r.get('dense_miss') is not None and r['head_source']=='caption']
    def estimate(rs):
        if len(rs)<4:
            return None
        # Control for visible GT area; still observational, not a causal test.
        a=np.array([r['gt_fraction'] for r in rs])
        design=np.stack([np.ones_like(a),a],axis=1)
        x=np.array([1-r['head_gt_recall'] for r in rs])
        y=np.array([r['dense_miss'] for r in rs])
        return corr(x-design@np.linalg.lstsq(design,x,rcond=None)[0],
                    y-design@np.linalg.lstsq(design,y,rcond=None)[0])
    value=estimate(use)
    groups={}
    for r in use:
        groups.setdefault(r['image_id'],[]).append(r)
    vals=list(groups.values()); rng=np.random.RandomState(123); boots=[]
    if len(vals)>=4:
        for _ in range(repetitions):
            e=estimate([r for i in rng.randint(len(vals),size=len(vals)) for r in vals[i]])
            if e is not None:
                boots.append(e)
    return {'partial_correlation':value,'ci95':np.percentile(boots,[2.5,97.5]).tolist() if boots else None,
            'caption_grounded_images':len(vals),'control':'linear GT area fraction; image-cluster bootstrap'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    model_args(p)
    p.add_argument('--manifest',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=200)
    p.add_argument('--fraction',type=float,default=.2)
    p.add_argument('--gate-min-images',type=int,default=50)
    p.add_argument('--gate-min-recall-gain',type=float,default=.05)
    a=p.parse_args(); out=Path(a.output)
    if not Path(a.manifest).is_file():
        p.error('Diagnostic manifest missing: '+a.manifest+
                '. Run bash experiments/ecca/run.sh manifest successfully first; no model was loaded.')
    manifest=read(a.manifest)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh diagnostic output directory')
    out.mkdir(parents=True,exist_ok=True)
    classes=manifest['classes']; ids=[int(c['id']) for c in classes]
    if len(set(ids))!=len(ids) or len(ids)<2:
        raise ValueError('Need unique IDs and complete competing class vocabulary')
    from .common import lexical
    lex=lexical(); model=Models(a)
    proj=model.project(model.encode(['a photo of '+c['name'] for c in classes]))
    rows=[]; pages=[]
    records=manifest['images'][:a.limit] if a.limit else manifest['images']
    for rec in records:
        path=image_path(a.image_root,rec); iid=int(rec['image_id']); prefix=str(iid)
        with Image.open(path) as im:
            original=im.convert('RGB'); feat=model.extract(original)
        crop=feat['crop']; side=a.size//14
        patches,attn,heads=feat['patches'],feat['attention'],normalize(feat['heads'])
        sims=proj@normalize(patches).T
        # Raw cosine + complete class competition, without background heuristics/PAMR.
        score_stack=[]
        for sim in sims:
            sm=Image.fromarray(sim.reshape(side,side).astype('float32'),mode='F')
            score_stack.append(np.asarray(sm.resize(crop.size,RESAMPLE.BILINEAR)))
        prediction=np.array(ids)[np.stack(score_stack).argmax(axis=0)]
        gt=None; valid=None
        if rec.get('mask'):
            mask_path=Path(rec['mask'])
            if not mask_path.is_absolute():
                mask_path=Path(a.manifest).resolve().parent/mask_path
            with Image.open(mask_path) as mask:
                if mask.size!=original.size:
                    raise ValueError('GT/image original size mismatch')
                # Resize to precisely the image geometry, using nearest labels.
                geometry=model.resize(original).size
                gt=np.asarray(model.crop(mask.resize(geometry,RESAMPLE.NEAREST)))
            if gt.ndim!=2:
                raise ValueError('Indexed single-channel GT mask required')
            valid=~np.isin(gt,manifest.get('ignore_values',[255]))
            unknown=set(np.unique(gt[valid]).tolist())-set(ids)
            if unknown:
                raise ValueError('GT IDs missing from class vocabulary: '+str(unknown))
        captions=rec.get('captions',[])
        cap_feats=model.project(model.encode(captions)) if captions else None
        cap_match=(cap_feats@heads.T).argmax(axis=1) if captions else []
        concepts=rec.get('concepts')
        if concepts is None:
            concepts=[c['name'] for c in classes if any(lex.in_raw_captions(c['name'],[x]) for x in captions)]
        for ci,c in enumerate(classes):
            if c['name'] not in concepts:
                continue
            supporting=[j for j,t in enumerate(captions) if lex.in_raw_captions(c['name'],[t])]
            # One image-concept row: select first matching original caption deterministically.
            head=int(cap_match[supporting[0]]) if supporting else int((proj[ci]@heads.T).argmax())
            neg=np.max(np.delete(sims,ci,axis=0),axis=0)
            margin=sims[ci]-neg
            evidence=1/(1+np.exp(-np.clip(margin/.07,-30,30)))
            support=attn@evidence
            selected=np.flatnonzero(support>=max(.5,float(support.max())*.95)).tolist()
            if not selected:
                selected=[int(support.argmax())]
            single=top_mask(attn[head],a.fraction).reshape(side,side)
            multi=top_mask(attn[selected].mean(axis=0),a.fraction).reshape(side,side)
            dense=top_mask(sims[ci],a.fraction)
            single_up=np.asarray(Image.fromarray(single).resize(crop.size,RESAMPLE.NEAREST),bool)
            multi_up=np.asarray(Image.fromarray(multi).resize(crop.size,RESAMPLE.NEAREST),bool)
            row={'image_id':iid,'concept':c['name'],'class_id':ids[ci], 'head':head,
                 'head_source':'caption' if supporting else 'concept','caption_indices':supporting,
                 'multi_heads':selected,'attention_dense_top_area_iou':iou(single.ravel(),dense,np.ones(side*side,bool)),
                 'gt_fraction':None,'head_gt_recall':None,'dense_iou':None,'dense_miss':None,
                 'multi_recall_gain':None,'multi_iou_gain':None}
            target=(prediction==ids[ci])
            if gt is not None:
                truth=(gt==ids[ci])&valid; n=int(truth.sum())
                row['gt_fraction']=float(n/max(int(valid.sum()),1))
                if n:
                    sr=float((truth&single_up).sum()/n); mr=float((truth&multi_up).sum()/n)
                    row.update(head_gt_recall=sr, dense_iou=iou(target,truth,valid),
                               dense_miss=float((truth&~target).sum()/n),multi_recall_gain=mr-sr,
                               multi_iou_gain=iou(multi_up,truth,valid)-iou(single_up,truth,valid))
            rows.append(row)
            safe=f'{prefix}_{ci}'
            panels=[crop,overlay(crop,attn[head].reshape(side,side)),overlay(crop,sims[ci].reshape(side,side)),
                    Image.fromarray((target*255).astype('uint8')).convert('RGB')]
            titles=['DINO crop',f'Caption head {head+1}','Raw class cosine','Raw class argmax']
            if gt is not None:
                panels.append(Image.fromarray(((gt==ids[ci])*255).astype('uint8')).convert('RGB')); titles.append('GT (diagnostic only)')
            canvas=Image.new('RGB',(256*len(panels),290),'white'); draw=ImageDraw.Draw(canvas)
            for j,(panel,title) in enumerate(zip(panels,titles)):
                canvas.paste(panel.resize((256,256)),(j*256,0));draw.text((j*256+5,262),title,fill='black')
            canvas.save(out/(safe+'.jpg'))
            pages.append('<p>'+html.escape(str(iid)+' '+c['name'])+'</p><img width="100%" src="'+safe+'.jpg">')
        print('diagnosed',iid,flush=True)
    model.close()
    assoc=association(rows)
    gain=grouped_bootstrap([r for r in rows if r['head_source']=='caption'],'multi_recall_gain')
    gain_iou=grouped_bootstrap([r for r in rows if r['head_source']=='caption'],'multi_iou_gain')
    ci=assoc['ci95']; gci=gain['ci95']; ici=gain_iou['ci95']
    eligible=bool(assoc['caption_grounded_images']>=a.gate_min_images and ci and ci[0]>0 and
                  gci and gci[0]>=a.gate_min_recall_gain and ici and ici[0]>0)
    report={'kind':'idea_a_diagnostic','manifest_sha256':digest(a.manifest),'model':model.metadata,
            'settings':vars(a),'rows':len(rows),'association':assoc,'matched_area_multi_recall_gain':gain,
            'matched_area_multi_iou_gain':gain_iou,'c_eligible_for_pilot':eligible,
            'gate_rules':{'min_caption_images':a.gate_min_images,'positive_area_controlled_correlation_ci':True,
                          'min_recall_gain_ci_lower':a.gate_min_recall_gain,'positive_iou_gain_ci':True},
            'interpretation':'Pilot screening only. Association is not causal evidence. C requires matched B/C training.',
            'dense_metric_scope':'center-crop raw cosine argmax; not official OVSS mIoU'}
    write(out/'rows.json',rows);write(out/'report.json',report)
    (out/'index.html').write_text('<meta charset="utf-8"><title>ECCA diagnosis</title>'+''.join(pages),encoding='utf-8')
    print('C pilot eligible:',eligible)


if __name__=='__main__':
    main()
