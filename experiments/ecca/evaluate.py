"""Launch original OVSS with explicit local model loading and per-run metrics."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .common import ROOT, HERE, read, write, digest

TASKS={'voc20':('voc','voc'),'voc':('voc_bg','voc_bg'),'context59':('context','context'),
       'ade20k':('ade','ade'),'coco_stuff':('stuff','stuff')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--name',required=True,help='Unique letters/digits/underscore name')
    p.add_argument('--output',required=True)
    p.add_argument('--clip-checkpoint',required=True)
    p.add_argument('--dino-repo',required=True)
    p.add_argument('--dino-checkpoint',required=True)
    p.add_argument('--pamr',choices=['on','off'],default='on')
    p.add_argument('--bg-clean',choices=['config','off'],default='config')
    p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS))
    p.add_argument('--port',type=int,default=29781)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',a.name):raise ValueError('Invalid name')
    ckpt=Path(a.checkpoint).resolve();out=Path(a.output).resolve()
    for path in (ckpt,Path(a.clip_checkpoint),Path(a.dino_checkpoint),Path(a.dino_repo)/'hubconf.py'):
        if not path.is_file():raise FileNotFoundError(str(path))
    weight=ROOT/'weights'/('ecca_'+a.name+'.pth');stem=weight.stem
    if weight.exists() or weight.is_symlink():
        if not weight.is_symlink() or weight.resolve()!=ckpt:
            raise FileExistsError('Conflicting eval weight name; choose a unique --name')
    elif not a.dry_run:
        weight.symlink_to(ckpt)
    env=dict(os.environ)
    env['PYTHONPATH']=str(HERE/'offline_guard')+os.pathsep+str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env['ECCA_CLIP_CHECKPOINT']=str(Path(a.clip_checkpoint).resolve())
    env['ECCA_DINO_CHECKPOINT']=str(Path(a.dino_checkpoint).resolve())
    env['ECCA_DINO_REPO']=str(Path(a.dino_repo).resolve())
    env['ECCA_BG_CLEAN']=a.bg_clean
    env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    cfg_root=Path('src/open_vocabulary_segmentation/configs')
    for idx,task in enumerate(a.tasks):
        folder,prefix=TASKS[task]
        model_cfg=cfg_root/folder/('dinotext_'+prefix+'_vitb_mlp_infonce.yml')
        base=cfg_root/folder/('eval_'+prefix+('_pamr' if a.pamr=='on' else '')+'.yml')
        for cfg in (model_cfg,base):
            if not (ROOT/cfg).is_file():raise FileNotFoundError(str(cfg))
        dest=out/a.pamr/a.bg_clean/task
        if dest.exists() and any(dest.iterdir()):raise FileExistsError('Evaluation output exists: '+str(dest))
        command=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1',
                 '--master_addr=127.0.0.1','--master_port='+str(a.port+idx),
                 'experiments/ecca/eval_worker.py','--eval','--eval_cfg',str(model_cfg),
                 '--eval_base_cfg',str(base),'--output',str(dest),'--opts','model.proj_name='+stem]
        print(' '.join(command),flush=True)
        if a.dry_run:continue
        dest.mkdir(parents=True,exist_ok=True)
        shared=ROOT/'segmentation_results'/(stem+'.json')
        # Each name is run-specific; official evaluator writes task scalar here.
        prior=read(shared) if shared.exists() else {}
        write(dest/'previous_shared_results.json',prior)
        prior.pop(task,None);write(shared,prior)
        with (dest/'eval.log').open('w') as log:
            run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:raise RuntimeError('Original OVSS failed; see '+str(dest/'eval.log'))
        result=read(shared)
        if task not in result:raise RuntimeError('Missing task in original segmentation result')
        miou=float(result[task])
        if not 0<=miou<=100:raise ValueError('Invalid mIoU')
        write(dest/'metrics.json',{'task':task,'miou':miou,'checkpoint':str(ckpt),'name':a.name,
              'pamr':a.pamr,'bg_clean':a.bg_clean,'checkpoint_sha256':digest(ckpt),
              'model_config_sha256':digest(ROOT/model_cfg),'eval_config_sha256':digest(ROOT/base),
              'clip_sha256':digest(a.clip_checkpoint),'dino_sha256':digest(a.dino_checkpoint)})
        print(task,miou,flush=True)


if __name__=='__main__':main()
