"""Recovery benchmark + independently labeled concept precision; optional calibration."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .common import read, write, selection_pass, digest
from .runtime import load_pth
from .prepare import defaults


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--labels',help='CSV: image_id,concept,present,grounded,split; blank=unknown')
    p.add_argument('--export-audit',help='Create an EMPTY audit-label CSV; refuses overwrite')
    p.add_argument('--tune',action='store_true')
    p.add_argument('--min-precision',type=float,default=.8)
    p.add_argument('--min-labeled',type=int,default=100)
    a=p.parse_args();root=Path(a.prepared);source=read(root/'source.json');rows=[];recover=[]
    for iid in source['image_ids']:
        rec=load_pth(root/'records'/(str(iid)+'.pth'))
        for r in rec['audit']:rows.append({'image_id':iid,**r})
        if rec['hidden_id'] is not None:
            k=source['top_k'];ranked=rec['audit'][:k]
            recover.append({'image_id':iid,'retrieved':rec['hidden_id'] in rec['candidate_ids'],
                            'head_recovered':rec['hidden_id'] in [r['gallery_id'] for r in ranked],
                            'filtered_recovered':rec['hidden_id'] in [r['gallery_id'] for r in rec['selected']]})
    report={'source_signature':source['signature'],'recovery_only':source['recovery_only'],
            'hidden_targets':len(recover),'recovery':{},
            'truth_scope':'Caption-supported held-out targets are positive-only. Non-target concepts are not verified absent.'}
    for metric in ('retrieved','head_recovered','filtered_recovered'):
        report['recovery'][metric]=float(np.mean([r[metric] for r in recover])) if recover else None
    if a.export_audit:
        with open(a.export_audit,'x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['image_id','concept','present','grounded','split'])
            writer.writeheader()
            writer.writerows({'image_id':r['image_id'],'concept':r['concept'],'present':'','grounded':'','split':''} for r in rows)
    if a.labels:
        with open(a.labels,newline='',encoding='utf-8-sig') as f:labels=list(csv.DictReader(f))
        keys=[(int(r['image_id']),r['concept']) for r in labels]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate audit labels')
        truth=dict(zip(keys,labels));labeled=[]
        image_splits={}
        for r in rows:
            y=truth.get((r['image_id'],r['concept']))
            if y and y.get('present') in ('0','1'):
                if y.get('split') not in ('calibration','test'):raise ValueError('Each labeled row needs split=calibration or test')
                image_splits.setdefault(r['image_id'],set()).add(y['split'])
                labeled.append({**r,'label':int(y['present']),'grounded_label':int(y['grounded']) if y.get('grounded') in ('0','1') else None,'split':y['split']})
        if any(len(s)>1 for s in image_splits.values()):raise ValueError('Calibration/test must be image-disjoint')
        def quality(items, config):
            by_image={}
            for r in items:
                if selection_pass(r,config):by_image.setdefault(r['image_id'],[]).append(r)
            accepted=[r for pool in by_image.values() for r in sorted(pool,key=lambda r:(-r['q'],-r['flip_iou'],r['gallery_id']))[:source['top_k']]]
            g=[r for r in accepted if r['grounded_label'] is not None]
            return {'labeled_candidates':len(items),'accepted_labeled':len(accepted),
                    'precision':float(np.mean([r['label'] for r in accepted])) if accepted else None,
                    'false_addition_rate':float(np.mean([1-r['label'] for r in accepted])) if accepted else None,
                    'grounding_accuracy':float(np.mean([r['grounded_label'] for r in g])) if g else None}
        report['labels_sha256']=digest(a.labels)
        report['quality']={s:quality([r for r in labeled if r['split']==s],source['selection']) for s in ('calibration','test')}
        report['label_coverage']=len(labeled)/max(len(rows),1)
        report['quality_scope']='Per-image top-K accepted concepts; unknown labels excluded, report label coverage'
        for split in ('calibration','test'):
            subset=[r for r in labeled if r['split']==split]
            by_image={}
            for r in subset:by_image.setdefault(r['image_id'],[]).append(r)
            head=[r for pool in by_image.values() for r in sorted(pool,key=lambda r:(-r['head_rank_score'],r['gallery_id']))[:source['top_k']]]
            report.setdefault('head_topk_quality',{})[split]={'n':len(head),'precision':float(np.mean([r['label'] for r in head])) if head else None}
        if a.tune:
            # Calibrate on entire audited images, never on held-out test labels.
            cal=[r for r in labeled if r['split']=='calibration']
            cal_ids={r['image_id'] for r in cal}
            expected=sum(r['image_id'] in cal_ids for r in rows)
            if len(cal)<a.min_labeled or len(cal)!=expected:
                raise ValueError('Calibration needs >= min-labeled candidates and COMPLETE labels per calibration image')
            options=[]
            for q in (.50,.55,.60,.65):
                for mass in (.5,.6,.7):
                    for flip in (.2,.35,.5):
                        cfg={**source['selection'],'q_min':q,'mass_min':mass,'flip_min':flip}
                        result=quality(cal,cfg)
                        if result['accepted_labeled']>=max(10,a.min_labeled//5) and result['precision']>=a.min_precision:
                            options.append((result['accepted_labeled'],result['precision'],q,mass,flip,cfg))
            if not options:raise ValueError('No setting reaches predeclared precision/coverage; do not proceed to full training')
            best=max(options,key=lambda x:x[:5])[-1]
            # Settings only: keep full provenance in adjacent report.
            frozen={k:v for k,v in best.items() if k!='gallery'}
            write(Path(a.output).with_name('selection_frozen.json'),frozen)
            report['tuned_on']='calibration split only'
            report['frozen_test_quality']=quality([r for r in labeled if r['split']=='test'],best)
    elif a.tune:
        raise ValueError('--tune requires independent present/absent labels')
    write(a.output,report)
    print(report)


if __name__=='__main__':main()
