"""BSC-CR ponytail: build -> original train.py -> original five-task OVSS.
Examples:
python -m experiments.Head_caption_completeness.pipeline --variant E3 --seed 123 --stage all --gpu 0
python -m experiments.Head_caption_completeness.pipeline --variant E1 --seed 124 --stage eval --gpu 1
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from .core import ROOT,BASELINES,BENCHMARKS,write_json,read_json

OUT=ROOT/'outputs/Head_caption_completeness'
WTS=ROOT/'weights/Head_caption_completeness'
CFG='src/open_vocabulary_segmentation/configs'


def project_name(variant,seed):return f'vitb_mlp_infonce_bsc_cr_{variant}_seed{seed}'


def paths(variant,seed):
    proj=project_name(variant,seed)
    work=OUT/'runs'/variant/f'seed_{seed}'
    weight=WTS/variant/f'seed_{seed}'/(proj+'.pth')
    link=ROOT/'weights'/(proj+'.pth')
    return proj,work,weight,link


def require(p):
    p=Path(p)
    if not p.is_file():raise FileNotFoundError(p)
    return p


def workenv(gpu,additional=None):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
    env.setdefault('PYTHONHASHSEED','123')
    env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if additional:env.update(additional)
    return env


def command(cmd,gpu,log,extra=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(x) for x in cmd]
    print('[CMD]', ' '.join(cmd),flush=True)
    with log.open('wb') as out:
        p=subprocess.Popen(cmd,cwd=ROOT,env=workenv(gpu,extra),stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,bufsize=0)
        try:
            while True:
                chunk=p.stdout.read(8192)
                if not chunk:break
                out.write(chunk);out.flush()
                stream=getattr(sys.stdout,'buffer',None)
                if stream:stream.write(chunk);stream.flush()
                else:print(chunk.decode(errors='replace'),end='',flush=True)
        finally:p.stdout.close()
        code=p.wait()
        if code:raise subprocess.CalledProcessError(code,cmd)


def free_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


def distributed_eval(command_args,gpu,dest,attempts=6):
    for idx in range(attempts):
        port=free_port()
        env={'MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(port),
             'RANK':'0','WORLD_SIZE':'1','LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'1'}
        log=dest/f'eval_port_attempt_{idx+1}.log'
        print(f'[PORT] choosing {port} for distributed OVSS rank=0 (try {idx+1}/{attempts})',flush=True)
        try:command(command_args,gpu,log,env)
        except subprocess.CalledProcessError:
            tail=log.read_bytes()[-32768:].decode(errors='replace').lower()
            if idx+1>=attempts or not any(x in tail for x in ('address already in use','eaddrinuse','errno: 98','errno 98')):raise
            print('[PORT COLLISION] retrying with a fresh OS-selected port',flush=True)
        else:
            shutil.copy2(log,dest/'eval.log')
            return port
    raise RuntimeError('Port allocation failed')


def link_checkpoint(weight,link):
    require(weight);link.parent.mkdir(parents=True,exist_ok=True)
    if link.is_symlink():
        if link.resolve()!=weight.resolve():raise FileExistsError(f'Unexpected link: {link}')
    elif link.exists():raise FileExistsError(f'Existing foreign checkpoint: {link}')
    else:link.symlink_to(os.path.relpath(weight,link.parent))


def run_build(a,work):
    if a.variant=='E0':
        print('[E0] uses original unmodified train.pth',flush=True);return
    require(a.base_train);require(a.selection / f"{a.variant}.jsonl")
    target=OUT/'training'/a.variant/'train.pth'
    manifest=OUT/'training'/a.variant/'manifest.json'
    # Dataset is seed-invariant: two jobs may build concurrently; guarded by flock.
    target.parent.mkdir(parents=True,exist_ok=True)
    with (target.parent/'build.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if target.exists() and manifest.exists() and not a.force_build:
            info=read_json(manifest)
            if (info['selection']!=str((a.selection/(a.variant+'.jsonl')).resolve()) or
                target.stat().st_mtime<a.selection.joinpath(a.variant+'.jsonl').stat().st_mtime):
                raise RuntimeError('Training dataset stale; use --force-build')
            print('[SKIP BUILD]',target,flush=True);return
        command([sys.executable,'-m','experiments.Head_caption_completeness.build',
                 '--base-train',a.base_train,'--selection',a.selection/(a.variant+'.jsonl'),
                 '--output',target,'--manifest',manifest,'--device',a.device,
                 '--batch-size',a.batch_size]+(['--subset-train'] if a.subset_train else []),
                a.gpu,target.parent/'build.log')
    require(target)


def run_train(a,proj,work,weight,link):
    dataset=a.base_train if a.variant=='E0' else OUT/'training'/a.variant/'train.pth'
    require(dataset);require(a.val);require(a.config)
    if weight.is_file() and not a.force_train:
        if dataset.stat().st_mtime>weight.stat().st_mtime+2:
            raise RuntimeError('Dataset newer than checkpoint; use --force-train')
        print('[SKIP TRAIN]',weight,flush=True);link_checkpoint(weight,link);return
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve()!=weight.resolve():raise FileExistsError(f'Unexpected {link}')
        link.unlink()
    if weight.exists():
        if not a.force_train:raise FileExistsError(weight)
        weight.unlink()
    weight.parent.mkdir(parents=True,exist_ok=True)
    # Run original train.py with seed override ONLY; rest of algorithm untouched.
    command([sys.executable,'-m','experiments.Head_caption_completeness.train_seed',
             '--seed',a.seed,'--train-data',dataset,'--val-data',a.val,
             '--name-pedix',proj.removeprefix('vitb_mlp_infonce_'),
             '--config',a.config],a.gpu,work/'train.log')
    if not link.exists() or link.is_symlink():raise FileNotFoundError(f'Original train.py failed to save {link}')
    shutil.move(str(link),weight)
    link_checkpoint(weight,link)
    print('[WEIGHT]',weight,flush=True)


def run_eval(a,proj,work,weight,link):
    require(weight);link_checkpoint(weight,link)
    result=work/'eval'/'results.json';summary=read_json(result) if result.exists() else {}
    result.parent.mkdir(parents=True,exist_ok=True)
    # The original eval writes to segmentation_results/<proj>.json. Serialize
    # duplicate launches of the SAME variant/seed; distinct experiments run parallel.
    with (work/'eval'/'evaluation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for task in a.tasks:
            config,basecfg=BENCHMARKS[task]
            require(ROOT/CFG/config);require(ROOT/CFG/basecfg)
            dest=work/'eval'/task;dest.mkdir(parents=True,exist_ok=True)
            stored=dest/'metrics.json'
            if stored.exists() and stored.stat().st_mtime>=weight.stat().st_mtime and not a.force_eval:
                val=float(read_json(stored)['miou']);print('[SKIP EVAL]',task,val,flush=True)
            else:
                start=time.time()
                port=distributed_eval([sys.executable,'src/open_vocabulary_segmentation/main.py',
                    '--eval','--eval_cfg',f'{CFG}/{config}','--eval_base_cfg',f'{CFG}/{basecfg}',
                    '--output',dest,'--opts',f'model.proj_name={proj}'],a.gpu,dest)
                produced=require(ROOT/'segmentation_results'/(proj+'.json'))
                if produced.stat().st_mtime<start-2:raise RuntimeError('Stale segmentation output '+str(produced))
                obj=read_json(produced)
                if task not in obj:raise KeyError(f'{task} not found in {produced}: {obj}')
                val=float(obj[task])
                write_json(stored,{'task':task,'miou':val,'port':port,'weight':str(weight),
                                   'eval_cfg':config,'eval_base_cfg':basecfg})
            if not 0<=val<=100:raise ValueError(f'{task}: invalid mIoU={val}')
            summary[task]={'miou':val,'baseline':BASELINES[task],
                           'delta':round(val-BASELINES[task],5)}
            write_json(result,summary)
            print(f'[RESULT] {proj} {task}: {val:.3f} ({val-BASELINES[task]:+.3f} vs baseline)',flush=True)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=[f'E{i}' for i in range(5)],required=True)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--stage',choices=['build','train','eval','all'],default='all')
    p.add_argument('--gpu',default='0',help='CUDA_VISIBLE_DEVICES; select separate GPU for parallel jobs')
    p.add_argument('--device',default='cuda')
    p.add_argument('--base-train',type=Path,default=ROOT/'outputs/coco2014_b14/train.pth')
    p.add_argument('--val',type=Path,default=ROOT/'outputs/coco2014_b14/val.pth')
    p.add_argument('--config',default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--selection',type=Path,default=OUT/'discovery')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--tasks',default=','.join(BENCHMARKS))
    p.add_argument('--subset-train',action='store_true')
    p.add_argument('--force-build',action='store_true')
    p.add_argument('--force-train',action='store_true')
    p.add_argument('--force-eval',action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    a=parse_args(argv)
    a.base_train=a.base_train.expanduser().resolve()
    a.val=a.val.expanduser().resolve()
    a.selection=a.selection.expanduser().resolve()
    a.tasks=[k.strip() for k in a.tasks.split(',') if k.strip()]
    for key in a.tasks:
        if key not in BENCHMARKS:raise ValueError('Unknown task '+key)
    proj,work,weight,link=paths(a.variant,a.seed)
    work.mkdir(parents=True,exist_ok=True)
    if a.stage in ('build','all'):run_build(a,work)
    if a.stage in ('train','all'):run_train(a,proj,work,weight,link)
    if a.stage in ('eval','all'):run_eval(a,proj,work,weight,link)
    print('[DONE]',a.variant,a.seed,a.stage,flush=True)
if __name__=='__main__':main()
