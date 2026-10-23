#!/usr/bin/env python3
"""Ponytail orchestrator: mine -> build -> ORIGINAL Talk2DINO train.py -> ORIGINAL 5-task OVSS eval."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.Head_Match.common.io import ROOT,OUT,WTS,write_json,read_json

SCHEMES={'A':'A_progressive','B':'B_confidence','C':'C_coevolution','D':'D_multi_teacher'}
CFG='src/open_vocabulary_segmentation/configs'
# Identical evaluation configurations to the previous 5-task local_cocu_talk2dino_new_train pipeline.
BENCHMARKS={
 'voc20':('voc/dinotext_voc_vitb_mlp_infonce.yml','voc/eval_voc_pamr.yml',88.54),
 'voc':('voc_bg/dinotext_voc_bg_vitb_mlp_infonce.yml','voc_bg/eval_voc_bg_pamr.yml',65.77),
 'coco_stuff':('stuff/dinotext_stuff_vitb_mlp_infonce.yml','stuff/eval_stuff_pamr.yml',30.23),
 'ade20k':('ade/dinotext_ade_vitb_mlp_infonce.yml','ade/eval_ade_pamr.yml',22.53),
 'context59':('context/dinotext_context_vitb_mlp_infonce.yml','context/eval_context_pamr.yml',42.42),
}


def absolute(p):
    p=Path(p).expanduser();return p if p.is_absolute() else ROOT/p


def require(path):
    if not Path(path).is_file():raise FileNotFoundError(f'Missing required file: {path}')
    return Path(path)


def model_config_for_original_train(config):
    """Pass a repo-relative config so original train.py takes DinoClipDataset.

    Upstream train.py dispatches dataset type with the substring check
    ``'dino' in args.model_config``. An absolute path through a checkout
    named Talking2dino erroneously switches to COCOCaptions and raw JPEG IO.
    The mine stage also normalizes args.config to an absolute path, hence
    simply forwarding args.config is not sufficient for --stage all.
    """
    path = require(absolute(config)).resolve()
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Training config must be inside the Talk2DINO repository for the "
            f"upstream train.py dataset selection: {path}"
        ) from exc
    if 'dino' in relative:
        raise ValueError(
            f"Original train.py uses a fragile 'dino' substring check. "
            f"Use a relative config filename/path without lowercase 'dino': {relative}"
        )
    return relative


def job_paths(variant,round_idx):
    name=f'vitb_mlp_infonce_headmatch_{variant}_r{round_idx}'
    folder=SCHEMES[variant]
    work=OUT/folder/f'round_{round_idx}'
    weight=WTS/folder/f'round_{round_idx}'/(name+'.pth')
    link=ROOT/'weights'/(name+'.pth')
    return work,weight,link,name


def cmdrun(cmd,args,log=None,extra=None):
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    env['PYTHONUNBUFFERED']='1'
    env.setdefault('PYTHONHASHSEED','123')
    if extra:env.update(extra)
    cmd=[str(x) for x in cmd]
    print('[CMD]',' '.join(cmd),flush=True)
    if args.dry_run:return
    if log is None:
        subprocess.run(cmd,cwd=ROOT,env=env,check=True);return
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('wb') as output:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
        try:
            while True:
                chunk=proc.stdout.read(4096)
                if not chunk:break
                output.write(chunk);output.flush()
                stream=getattr(sys.stdout,'buffer',None)
                if stream:stream.write(chunk);stream.flush()
                else:print(chunk.decode(errors='replace'),end='',flush=True)
        finally:proc.stdout.close()
        result=proc.wait()
        if result:raise subprocess.CalledProcessError(result,cmd)


def pick_free_master_port():
    """Ask the OS for an available IPv4 TCP port for single-process torch.distributed.

    The socket must close before torch.distributed can bind it. Because there is
    a small race between releasing and using that port, eval retries on EADDRINUSE.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def run_distributed_eval(cmd, args, dest, max_attempts=6):
    """Launch original OVSS evaluation; retry ONLY if MASTER_PORT was occupied.

    Keep failed logs; publish the successful run as eval.log. Never ignore
    other failures (CUDA OOM, missing dataset, bad weights, etc.).
    """
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    for attempt in range(1, max_attempts + 1):
        port = pick_free_master_port()
        log = dest / f'eval_port_attempt_{attempt}.log'
        print(f'[DIST] MASTER_ADDR=127.0.0.1 MASTER_PORT={port} '
              f'(attempt {attempt}/{max_attempts})', flush=True)
        env = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
               'RANK': '0', 'WORLD_SIZE': '1',
               'LOCAL_RANK': '0', 'LOCAL_WORLD_SIZE': '1'}
        try:
            cmdrun(cmd, args, log, env)
        except subprocess.CalledProcessError:
            tail = log.read_bytes()[-32768:].decode('utf-8', errors='replace').lower() if log.exists() else ''
            port_conflict = ('address already in use' in tail or
                             'errno: 98' in tail or
                             'eaddrinuse' in tail)
            if not port_conflict or attempt == max_attempts:
                raise
            print(f'[DIST RETRY] Port {port} already in use; '
                  'choosing another available port.', flush=True)
            continue
        # Keep the canonical file name used by existing scripts and preserve
        # collision logs separately for auditing.
        if not args.dry_run:
            log.replace(dest / 'eval.log')
        return port
    raise AssertionError('Unreachable distributed-eval retry state')


def ensure_link(weight,link):
    require(weight)
    link.parent.mkdir(parents=True,exist_ok=True)
    if link.is_symlink():
        if link.resolve()!=weight.resolve():raise FileExistsError(f'Foreign checkpoint link: {link}')
        return
    if link.exists():raise FileExistsError(f'Existing non-symlink checkpoint: {link}')
    link.symlink_to(os.path.relpath(weight,link.parent))


def teachers(args,r):
    paths=[require(absolute(args.teacher))]
    if r>1:
        if args.variant=='D':
            for previous in range(1,r):
                work,weight,_,_=job_paths(args.variant,previous)
                paths.append(require(weight))
        else:
            _,weight,_,_=job_paths(args.variant,r-1)
            paths=[require(weight)]
    return paths


def mine(args,r,work):
    out=work/'selection.json'
    if out.exists() and not args.force_mine:
        meta=read_json(out)['meta']
        expected=[str(x) for x in teachers(args,r)]
        if meta['teacher_weights']!=expected:
            raise RuntimeError(f'Stale selection teachers in {out}: rerun with --force-mine')
        print('[SKIP MINE]',out);return
    from experiments.Head_Match.common.discovery import mine as discover
    # Keep source paths resolved, including JSON corpus and optional gallery.
    args.concepts=[require(absolute(x)) for x in args.concepts]
    args.gallery=require(absolute(args.gallery)) if args.gallery else None
    args.base_train=require(absolute(args.base_train))
    args.config=require(absolute(args.config))
    discover(args,r,teachers(args,r),out)


def build(args,r,work):
    selection=require(work/'selection.json')
    target=work/'train_augmented.pth'
    manifest=work/'train_manifest.json'
    if target.is_file() and not args.force_build:
        if not manifest.is_file() or target.stat().st_mtime+2<selection.stat().st_mtime:
            raise RuntimeError(f'Build stale or manifest missing: {target}; --force-build')
        print('[SKIP BUILD]',target);return
    cmd=[sys.executable,'-m','experiments.Head_Match.common.build_train',
         '--base-train',absolute(args.base_train),'--selection',selection,
         '--output',target,'--manifest',manifest,
         '--batch-size',args.batch_size,'--device',args.device]
    if args.subset_train:cmd.append('--subset-train')
    cmdrun(cmd,args,work/'build.log')
    if not args.dry_run:require(target)


def train(args,r,work,weight,link,name):
    tr=require(work/'train_augmented.pth')
    require(absolute(args.val));require(absolute(args.config))
    if weight.exists() and not args.force_train:
        if weight.stat().st_mtime+2<tr.stat().st_mtime:
            raise RuntimeError(f'Checkpoint predates train PTH: {weight}; --force-train')
        ensure_link(weight,link);print('[SKIP TRAIN]',weight);return
    if link.is_symlink():
        if link.resolve()!=weight.resolve():raise FileExistsError(f'Foreign checkpoint link {link}')
        link.unlink()
    elif link.exists():raise FileExistsError(f'Existing checkpoint {link}; refusing overwrite')
    if weight.exists():
        if not args.force_train:raise FileExistsError(str(weight))
        weight.unlink()
    weight.parent.mkdir(parents=True,exist_ok=True)
    # The original train.py explicitly reinitializes random seed 123; no claim of varying seed.
    cmdrun([sys.executable,'train.py','--model_config',model_config_for_original_train(args.config),
            '--train_dataset',tr,'--val_dataset',absolute(args.val),
            '--feature_name','disentangled_self_attn','--text_features','ann_feats',
            '--name_pedix',name[len('vitb_mlp_infonce_'):]],args,work/'train.log')
    if args.dry_run:return
    require(link)
    shutil.move(str(link),str(weight))
    ensure_link(weight,link)
    print('[SAVED WEIGHT]',weight,flush=True)


def evaluate(args,r,work,weight,link,name):
    require(weight);ensure_link(weight,link)
    result_dir=work/'eval';result_dir.mkdir(parents=True,exist_ok=True)
    summary=result_dir/'summary.json'
    results=read_json(summary) if summary.exists() else {}
    for task in args.tasks.split(','):
        task=task.strip()
        if task not in BENCHMARKS:raise ValueError(f'Unknown eval task: {task}')
        e,b,baseline=BENCHMARKS[task]
        if not (ROOT/CFG/e).is_file() or not (ROOT/CFG/b).is_file():
            raise FileNotFoundError(f'Missing repository OVSS eval config for {task}: {e} or {b}')
        dest=result_dir/task;dest.mkdir(parents=True,exist_ok=True)
        metricfile=dest/'metrics.json'
        if metricfile.exists() and not args.force_eval and metricfile.stat().st_mtime+2>=weight.stat().st_mtime:
            metrics=read_json(metricfile)
            value=float(metrics[task]);print('[SKIP EVAL]',task,value)
        else:
            started=time.time()
            run_distributed_eval(
                [sys.executable, 'src/open_vocabulary_segmentation/main.py',
                 '--eval', '--eval_cfg', f'{CFG}/{e}',
                 '--eval_base_cfg', f'{CFG}/{b}',
                 '--output', dest, '--opts', f'model.proj_name={name}'],
                args, dest)
            if args.dry_run:continue
            produced=require(ROOT/'segmentation_results'/(name+'.json'))
            if produced.stat().st_mtime < started-2:
                raise RuntimeError(f'Segmentation result not freshly written for {task}: {produced}')
            metrics=read_json(produced)
            if task not in metrics:raise KeyError(f'Expected {task} in {produced}: {metrics}')
            value=float(metrics[task]);write_json(metricfile,metrics)
        if not 0<=value<=100:raise ValueError(f'Invalid mIoU for {task}: {value}')
        results[task]={'miou':round(value,4),'baseline':baseline,'delta':round(value-baseline,4),
                       'weight':str(weight.relative_to(ROOT)),
                       'eval_cfg':e,'base_cfg':b}
        write_json(summary,results)
        print(f'[EVAL] {task}: {value:.2f} (baseline={baseline:.2f}, delta={value-baseline:+.2f})',flush=True)
    print('[SUMMARY]',summary)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=SCHEMES,required=True)
    p.add_argument('--stage',choices=['mine','build','train','eval','all'],default='all')
    p.add_argument('--gpu',default='0')
    p.add_argument('--device',default='cuda')
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--round',type=int,default=0,help='0=all rounds 1..rounds, otherwise exactly this round')
    p.add_argument('--teacher',default='weights/vitb_mlp_infonce.pth')
    p.add_argument('--base-train',default='outputs/coco2014_b14/train.pth')
    p.add_argument('--val',default='outputs/coco2014_b14/val.pth')
    p.add_argument('--config',default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--concepts',action='append',default=[],help='Existing concepts_by_image.json (repeatable)')
    p.add_argument('--gallery',default='experiments/local_cocu/MSCOCO14_train_gallery.json',
                   help='Gallery for fresh Teacher-guided concepts each round; empty string disables')
    p.add_argument('--gallery-per-head',type=int,default=5)
    p.add_argument('--local-candidate-cap',type=int,default=80)
    p.add_argument('--max-candidates',type=int,default=120)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--assignment',choices=['soft','argmax'],default='soft')
    p.add_argument('--temperature',type=float,default=.07)
    p.add_argument('--top-heads',type=int,default=4)
    p.add_argument('--max-concepts',type=int,default=2)
    p.add_argument('--proposal-count',type=int,default=10)
    p.add_argument('--min-visual',type=float,default=.24)
    p.add_argument('--novelty-max',type=float,default=.42)
    p.add_argument('--covered-sim',type=float,default=.37)
    p.add_argument('--coverage-threshold',type=float,default=.60)
    p.add_argument('--min-delta',type=float,default=-.03,help='Min matched-head counterfactual gain for B/C/D')
    p.add_argument('--max-head-shift',type=float,default=.65,help='Max total-variation distance for C')
    p.add_argument('--min-votes',type=int,default=2,help='Required Teacher votes for D; warmup round 1 has one Teacher')
    p.add_argument('--limit',type=int,default=0,help='Mine first N images; requires --subset-train to train')
    p.add_argument('--subset-train',action='store_true')
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--tasks',default=','.join(BENCHMARKS))
    p.add_argument('--force-mine',action='store_true')
    p.add_argument('--force-build',action='store_true')
    p.add_argument('--force-train',action='store_true')
    p.add_argument('--force-eval',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if args.rounds<1 or args.round<0 or (args.round>args.rounds):p.error('Bad --round / --rounds')
    if args.max_concepts<1 or args.proposal_count<args.max_concepts or args.batch_size<1:p.error('Bad concept counts or batch size')
    if args.temperature<=0 or not 0<=args.coverage_threshold<=1:p.error('Invalid temperature or coverage threshold')
    if args.limit and args.stage in ('all','train','build') and not args.subset_train:
        p.error('--limit changes the population; pass --subset-train to train an explicit subset')
    if not args.gallery and not args.concepts:p.error('Provide --concepts when disabling gallery')
    return args


def main(argv=None):
    args=parse_args(argv)
    rounds=[args.round] if args.round else range(1,args.rounds+1)
    for r in rounds:
        work,weight,link,name=job_paths(args.variant,r)
        print(f'\n== Head_Match {args.variant}: round {r} stage {args.stage} ==',flush=True)
        if args.dry_run:
            print('[PLAN] student weight',weight,'raw source',absolute(args.base_train))
            print('[PLAN] selection',work/'selection.json')
            if args.stage in ('eval','all'):
                for k,(a,b,_) in BENCHMARKS.items():print('[PLAN EVAL]',k,a,b)
            continue
        if args.stage in ('mine','all'):mine(args,r,work)
        if args.stage in ('build','all'):build(args,r,work)
        if args.stage in ('train','all'):train(args,r,work,weight,link,name)
        if args.stage in ('eval','all'):evaluate(args,r,work,weight,link,name)
if __name__=='__main__':main()
