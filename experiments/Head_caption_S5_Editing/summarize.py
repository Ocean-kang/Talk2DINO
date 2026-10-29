"""Collate five-task results without inventing missing mIoU measurements."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from experiments.Head_caption_completeness.core import ROOT,BASELINES

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'outputs/Head_caption_S5_Editing')
    p.add_argument('--variants',default='E0,E1,E2,E3')
    p.add_argument('--seeds',default='123')
    a=p.parse_args(argv)
    rows=[]
    for var in a.variants.split(','):
        for seed in a.seeds.split(','):
            source=a.out/'runs'/var/f'seed_{seed}'/'eval'/'results.json'
            data=json.loads(source.read_text()) if source.is_file() else {}
            for task,b in BASELINES.items():
                obj=data.get(task)
                rows.append({'variant':var,'seed':seed,'task':task,'baseline':b,
                             'miou':obj.get('miou') if obj else None,
                             'delta':obj.get('delta') if obj else None,
                             'status':'done' if obj else 'not_run'})
    target=a.out/'segmentation_comparison.csv';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','seed','task','baseline','miou','delta','status'])
        writer.writeheader();writer.writerows(rows)
    print('[SUMMARY]',target)
    for r in rows:
        print(r['variant'],r['seed'],r['task'],str(r['miou']),str(r['delta']),r['status'])
if __name__=='__main__':main()
