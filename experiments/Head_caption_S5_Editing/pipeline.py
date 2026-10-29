"""Isolated orchestration; delegates train and five-task segmentation to original code.

Training: original train.py, via existing seed wrapper. Eval: original OVSS main.py.
Only the input train.pth and checkpoint names change.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from experiments.Head_caption_completeness.core import ROOT,BASELINES,BENCHMARKS,write_json
from experiments.Head_caption_completeness.pipeline import command,distributed_eval,link_checkpoint,require

OUT=ROOT/'outputs/Head_caption_S5_Editing'; WTS=ROOT/'weights/Head_caption_S5_Editing'
CFG='src/open_vocabulary_segmentation/configs'


def name(variant,seed):return f'vitb_mlp_infonce_s5edit_{variant}_seed{seed}'


def run(a):
    proj=name(a.variant,a.seed)
    work=OUT/'runs'/a.variant/f'seed_{a.seed}';work.mkdir(parents=True,exist_ok=True)
    weight=WTS/a.variant/f'seed_{a.seed}'/(proj+'.pth')
    link=ROOT/'weights'/(proj+'.pth')
    dataset=(a.base_train if a.variant=='E0' else OUT/'training'/a.variant/'train.pth')
    if a.stage in ('build','all') and a.variant!='E0':
        target=OUT/'training'/a.variant/'train.pth'
        cmd=[os.sys.executable,'-m','experiments.Head_caption_S5_Editing.build',
             '--variant',a.variant,'--base-train',a.base_train,
             '--selection-dir',a.selection_dir,'--output',target,
             '--manifest',OUT/'training'/a.variant/'manifest.json','--device',a.device,
             '--batch-size',a.batch_size]
        if a.subset_train:cmd.append('--subset-train')
        if target.is_file() and not a.force_build:
            if target.stat().st_mtime<(a.selection_dir/f'{a.variant}.jsonl').stat().st_mtime:
                raise RuntimeError('Stale edited training data; use --force-build')
            print('[SKIP BUILD]',target,flush=True)
        else:command(cmd,a.gpu,work/'build.log')
    if a.stage in ('train','all'):
        require(dataset);require(a.val);require(ROOT/a.config)
        if weight.is_file() and not a.force_train:
            if weight.stat().st_mtime<dataset.stat().st_mtime-2:
                raise RuntimeError('Stale checkpoint; use --force-train')
            print('[SKIP TRAIN]',weight,flush=True);link_checkpoint(weight,link)
        else:
            if link.is_symlink():
                if link.resolve()!=weight.resolve():raise FileExistsError(f'Foreign checkpoint link: {link}')
                link.unlink()
            elif link.exists():raise FileExistsError(f'Unexpected checkpoint exists: {link}')
            if weight.exists():
                if not a.force_train:raise FileExistsError(weight)
                weight.unlink()
            weight.parent.mkdir(parents=True,exist_ok=True)
            cmd=[os.sys.executable,'-m','experiments.Head_caption_completeness.train_seed',
                '--seed',a.seed,'--train-data',dataset,'--val-data',a.val,
                '--name-pedix',proj.removeprefix('vitb_mlp_infonce_'),'--config',a.config]
            command(cmd,a.gpu,work/'train.log')
            # Original train.py saved weights/<proj>.pth; atomically relocate.
            require(link);shutil.move(str(link),weight);link_checkpoint(weight,link)
            print('[TRAIN WEIGHT]',weight,flush=True)
    if a.stage in ('eval','all'):
        require(weight);link_checkpoint(weight,link)
        all_res={};result=work/'eval'/'results.json'
        if result.exists():all_res=json.loads(result.read_text())
        for task in a.tasks:
            config,basecfg=BENCHMARKS[task]
            dest=work/'eval'/task;dest.mkdir(parents=True,exist_ok=True)
            stored=dest/'metrics.json'
            if stored.exists() and stored.stat().st_mtime>=weight.stat().st_mtime and not a.force_eval:
                obj=json.loads(stored.read_text());value=float(obj['miou'])
            else:
                start=time.time()
                port=distributed_eval([os.sys.executable,'src/open_vocabulary_segmentation/main.py',
                    '--eval','--eval_cfg',f'{CFG}/{config}','--eval_base_cfg',f'{CFG}/{basecfg}',
                    '--output',dest,'--opts',f'model.proj_name={proj}'],a.gpu,dest)
                produced=require(ROOT/'segmentation_results'/(proj+'.json'))
                if produced.stat().st_mtime<start-2:raise RuntimeError('Stale eval result')
                obj=json.loads(produced.read_text(encoding='utf8'))
                if task not in obj:raise KeyError(f'{task} missing from {produced}')
                value=float(obj[task]);write_json(stored,{'miou':value,'port':port,'task':task,'checkpoint':str(weight)})
            if not (0<=value<=100):raise ValueError(f'{task}: bad mIoU={value}')
            all_res[task]={'miou':value,'baseline':BASELINES[task],'delta':round(value-BASELINES[task],5)}
            write_json(result,all_res)
            print(f'[OVSS] {task}: {value:.2f}, Δ={value-BASELINES[task]:+.2f}',flush=True)
    print(f'[DONE] {a.stage} {a.variant} seed {a.seed}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['E0','E1','E2','E3'],required=True)
    p.add_argument('--stage',choices=['build','train','eval','all'],default='all')
    p.add_argument('--seed',type=int,default=123);p.add_argument('--gpu',default='0')
    p.add_argument('--device',default='cuda')
    p.add_argument('--base-train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--val',type=Path,default=ROOT/'outputs/coco2014_b14/val.pth')
    p.add_argument('--selection-dir',type=Path,default=OUT/'discovery')
    p.add_argument('--config',default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--tasks',default=','.join(BASELINES))
    p.add_argument('--subset-train',action='store_true')
    p.add_argument('--force-build',action='store_true')
    p.add_argument('--force-train',action='store_true')
    p.add_argument('--force-eval',action='store_true')
    a=p.parse_args(argv)
    a.base_train=a.base_train.resolve();a.val=a.val.resolve();a.selection_dir=a.selection_dir.resolve()
    a.tasks=[s.strip() for s in a.tasks.split(',') if s.strip()]
    if len(set(a.tasks))!=len(a.tasks) or any(t not in BASELINES for t in a.tasks):raise ValueError('Unknown/duplicate segmentation task')
    if os.path.isabs(a.config) or 'dino' in str(a.config).lower():
        raise ValueError('Original train.py expects repo-relative config path without the word dino')
    run(a)
if __name__=='__main__':main()
