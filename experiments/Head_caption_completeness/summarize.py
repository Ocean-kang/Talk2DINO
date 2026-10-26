"""Aggregate five benchmark scores across independent seeds for E0..E4."""
from __future__ import annotations
import argparse,csv,statistics
from pathlib import Path
from .core import ROOT,BASELINES,read_json,write_json

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',default='123,124,125,126,127')
    p.add_argument('--variants',default='E0,E1,E2,E3,E4')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/Head_caption_completeness/summary')
    p.add_argument('--allow-missing',action='store_true')
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
    seeds=[int(x.strip()) for x in a.seeds.split(',')]
    variants=[x.strip() for x in a.variants.split(',')]
    rows=[]
    for v in variants:
        for task,baseline in BASELINES.items():
            values={}
            for seed in seeds:
                path=ROOT/'outputs/Head_caption_completeness/runs'/v/f'seed_{seed}'/'eval'/'results.json'
                if path.is_file() and task in read_json(path):
                    values[str(seed)]=float(read_json(path)[task]['miou'])
                elif not a.allow_missing:raise FileNotFoundError(f'Missing {task} for {v} seed {seed}: {path}')
            if not values:continue
            x=list(values.values());mu=statistics.mean(x)
            row={'variant':v,'dataset':task,'n':len(x),'baseline':baseline,
                 'mean':round(mu,5),'std_sample':round(statistics.stdev(x),5) if len(x)>1 else None,
                 'delta_mean':round(mu-baseline,5),'seed_scores':values}
            rows.append(row)
    write_json(a.output/'mean_std.json',rows)
    with (a.output/'mean_std.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['variant','dataset','n','baseline','mean','std_sample','delta_mean']+
                         [f'seed_{x}' for x in seeds]);w.writeheader()
        for row in rows:
            w.writerow({k:row.get(k) for k in ['variant','dataset','n','baseline','mean','std_sample','delta_mean']}|
                       {f'seed_{s}':row['seed_scores'].get(str(s)) for s in seeds})
    print('[SUMMARY]',a.output/'mean_std.csv')
if __name__=='__main__':main()
