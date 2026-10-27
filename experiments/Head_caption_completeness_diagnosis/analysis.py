"""Method/perturbation metrics, paired bootstrap CI (image-level resampling)."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
import random
import statistics
from pathlib import Path
from .core import delta, dump_json, is_finite


def mean(xs):
    xs=[x for x in xs if is_finite(x)]
    return statistics.mean(xs) if xs else None


def image_bootstrap_ci(rows,key,nboot=300,seed=123):
    per=defaultdict(list)
    for r in rows:
        if is_finite(r.get(key)):per[r['image_id']].append(r[key])
    if len(per)<2 or nboot<=0:return None
    unit=list(per.values());r=random.Random(seed)
    values=[]
    for _ in range(nboot):
        sampled=[unit[r.randrange(len(unit))] for _ in unit]
        values.append(sum(map(sum,sampled))/sum(map(len,sampled)))
    values.sort()
    return [values[int((len(values)-1)*.025)],values[int((len(values)-1)*.975)]]


def aggregate(records,nboot=300):
    grouped=defaultdict(list)
    for row in records:grouped[(row['method'],row['kind'])].append(row)
    table=[]
    for (method,kind),rs in sorted(grouped.items()):
        comparable=[r for r in rs if r.get('comparable')]
        known=[r for r in comparable if r.get('target_known')]
        checks=[r for r in known if r.get('direction_success') is not None]
        passes=[r for r in comparable if r.get('counterfactual_pass') is True]
        sub=dict(method=method,kind=kind,n=len(rs),n_comparable=len(comparable),
                 comparable_rate=len(comparable)/len(rs),target_known_rate=len(known)/len(rs),
                 direction_n=len(checks),direction_accuracy=mean([float(r['direction_success']) for r in checks]),
                 counterfactual_pass_rate=len(passes)/len(rs),
                 target_text_delta=mean([r['target_text_delta'] for r in known]),
                 target_text_abs_delta=mean([abs(r['target_text_delta']) for r in known if is_finite(r.get('target_text_delta'))]),
                 off_target_mean_abs=mean([r['off_target_mean_abs_delta'] for r in comparable]),
                 mean_delta_add=mean([r['delta_add'] for r in comparable]),
                 mean_delta_remove=mean([r['delta_remove'] for r in comparable]),
                 mean_leakage_ratio=mean([r.get('leakage_ratio') for r in comparable]),
                 mean_valid_concepts=mean([r['valid_count'] for r in rs]),
                 mean_valid_fraction=mean([r['valid_fraction'] for r in rs]),
                 target_sensitivity=mean([
                     float(r['target_text_delta'] < -1e-6) if kind=='delete_supported'
                     else float(r['target_text_delta'] > 1e-6) if kind in ('add_high_evidence','add_low_evidence')
                     else float(abs(r['target_text_delta']) <= .15)
                     for r in known if is_finite(r.get('target_text_delta'))
                 ]) if kind!='neutral_prefix' else None,
                 mean_teacher_head=mean([r['head_score'] for r in rs]))
        sub['ci95_delta_add']=image_bootstrap_ci(comparable,'delta_add',nboot)
        sub['ci95_delta_remove']=image_bootstrap_ci(comparable,'delta_remove',nboot)
        if method=='S4':
            sub['direction_accuracy_cf_pass']=mean([float(r['direction_success']) for r in passes if r.get('direction_success') is not None])
            sub['cf_pass_direction_n']=sum(r.get('direction_success') is not None for r in passes)
        table.append(sub)
    return table


def save_reports(out,records,meta,nboot=300):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    table=aggregate(records,nboot)
    dump_json(out/'summary.json',dict(metadata=meta,by_method_perturbation=table,
              caveats=['Synthetic high/low visual support is not annotated concept truth.',
                       'S4 is S3 with counterfactual acceptance gating, not a different T estimator.',
                       'Direction on add_low_evidence is evaluated only if its mathematical monotonic condition holds.',
                       'S6 max-over-synonym-group has no exact per-concept additive attribution.']))
    if table:
        with (out/'summary.csv').open('w',encoding='utf-8',newline='') as f:
            columns=list(dict.fromkeys(k for record in table for k in record));writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for row in table:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    errors=sorted([r for r in records if r.get('direction_success') is False and r.get('comparable')],
                  key=lambda r:(r['method'],r['kind'],r['delta_add'] or 0))[:300]
    with (out/'direction_failures.jsonl').open('w',encoding='utf-8') as f:
        for row in errors:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    return table
