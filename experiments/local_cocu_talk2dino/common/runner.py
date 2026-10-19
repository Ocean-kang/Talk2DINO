#!/usr/bin/env python3
"""A-H pipeline launcher. Standalone: no edits to repo/src, train.py or eval configs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from .data import (ROOT,BASE,OUTPUT,WEIGHTS,read_json,write_json,
                   captions_and_images,gallery_words,validate_id_coverage)
from .experiment import mine
from .retrieval import local_neighbors

NAMES={
 'A':'A_direct_retrieval', 'B':'B_head_image_retrieval',
 'C':'C_uncovered_heads','D':'D_region_discovery',
 'E':'E_cross_image_consistency','F':'F_counterfactual',
 'G':'G_multiview_agreement','H':'H_iterative_bootstrap'}
EVALS={
 'voc20':('voc/dinotext_voc_vitb_mlp_infonce.yml','voc/eval_voc_pamr.yml'),
 'context59':('context_bg/dinotext_context_bg_vitb_mlp_infonce.yml','context_bg/eval_context_bg_pamr.yml'),
 'ade150':('ade/dinotext_ade_vitb_mlp_infonce.yml','ade/eval_ade_pamr.yml'),
 'coco_stuff':('stuff/dinotext_stuff_vitb_mlp_infonce.yml','stuff/eval_stuff_pamr.yml'),
}

def run(cmd,args,envextra=None):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    env.setdefault('PYTHONHASHSEED','123')
    if envextra:env.update(envextra)
    print('[CMD]', ' '.join(map(str,cmd)),flush=True)
    if not args.dry_run: subprocess.run([str(c) for c in cmd],env=env,cwd=ROOT,check=True)

def required(path,kind='file'):
    if not Path(path).exists(): raise FileNotFoundError(f'Required {kind} missing: {path}')

def setup_heads(args,work):
    path=ROOT/args.heads
    if path.exists():return path
    if not args.extract_missing_heads:raise FileNotFoundError(f'{path} not found; pass --extract-missing-heads or --heads EXISTING_PT')
    path=work/'source'/'head_features.pt'
    if path.exists():return path
    run([sys.executable,'experiments/local_cocu/extract_head_features.py',
         '--annotation',args.coco,'--image_dir',args.images,'--model','dinov2_vitb14_reg',
         '--resize_dim','448','--crop_dim','448','--batch_size','16','--fp16',
         '--output',path],args)
    if not args.dry_run:required(path)
    return path

def setup_neighbors(args,work,heads,ids):
    out=work/'neighbors.json';details=work/'neighbor_matches.json'
    if out.exists() and details.exists() and not args.force_mine:
        return read_json(out),read_json(details)
    n,d=local_neighbors(heads,ids,topk=args.neighbor_topk,
         per_head_search=args.neighbor_per_head,ef=args.faiss_ef,
         batch=args.retrieval_batch,index_type=args.faiss_index)
    write_json(out,n);write_json(details,d)
    return n,d

def setup_cls_neighbors(args,work):
    requested=ROOT/args.cls_neighbors
    if requested.exists():return read_json(requested)
    if not args.extract_missing_cls:
        raise FileNotFoundError(f'F requires same-scene CLS neighbors: {requested}; --extract-missing-cls builds them')
    clsdir=work/'source'/'cls'
    run([sys.executable,'experiments/local_cocu/extract_dino_cls_features.py',
         '--caption',args.coco,'--image_dir',args.images,'--output',clsdir],args)
    retrieve=work/'source'/'cls_near'
    run([sys.executable,'experiments/local_cocu_novel/chunked_retrieval.py',
         '--feature',clsdir/'dino_cls.pt','--output',retrieve,
         '--topk',str(max(args.negative_topk,20)),
         '--device',args.device,'--chunk-size','128'],args)
    required(retrieve/'neighbors.json')
    return read_json(retrieve/'neighbors.json')

def datasets(args,work):
    caps,paths=captions_and_images(ROOT/args.coco)
    heads_path=setup_heads(args,work)
    if args.dry_run: return None
    heads=torch.load(heads_path,map_location='cpu')
    heads={int(k):torch.as_tensor(v) for k,v in heads.items()}
    # COCO image IDs stored in head cache determine discoverable images.
    ids=sorted(heads)
    if args.limit:ids=ids[:args.limit]
    validate_id_coverage(ids,caps,heads)
    if not ids:raise RuntimeError('No valid head features')
    for iid in ids:
        if torch.as_tensor(heads[iid]).ndim!=2: raise ValueError(f'Expected 12x768 head features at image {iid}')
    if any(heads[iid].shape!=(12,768) for iid in ids):
        raise ValueError('Teacher vitb_mlp_infonce.pth expects DINOv2 ViT-B/14 12x768 head features')
    return heads,ids,caps,{i:ROOT/args.images/paths[i] for i in ids}

def do_mine(scheme,args,work,teacher_weights,label=None):
    audit=work/'concepts_by_image.json'
    detailed=work/'matched'/'matched_concepts_detailed.json'
    if audit.exists() and detailed.exists() and not args.force_mine:
        print('[CACHE] existing discovery',audit)
        return
    data=datasets(args,work)
    if args.dry_run: return
    heads,ids,caps,paths=data
    gallery=gallery_words(ROOT/args.gallery)
    print(f'[INPUT] {len(ids)} images, {len(gallery)} gallery concepts, teacher={teacher_weights}',flush=True)
    neigh=None;matches=None;cls_neigh=None
    if scheme in 'BEF':neigh,matches=setup_neighbors(args,work,heads,ids)
    if scheme=='F':
        cls_neigh=setup_cls_neighbors(args,work)
        if not any(int(n) in heads for iid in ids for n in cls_neigh.get(str(iid),[])[:args.negative_topk]):
            raise RuntimeError('CLS negatives do not overlap with head-feature image IDs: cannot implement F')
    from .teacher import Teacher
    teacher=Teacher(teacher_weights,ROOT/args.config,args.device,batch=args.encode_batch)
    recorded,matched={},{}
    for image_id,item in mine(scheme,ids,heads,caps,gallery,teacher,args,
                            neigh,matches,cls_neigh,paths):
        if label:
            item['scheme']=label
            item['diagnostics']['discovery_operator']=scheme
        recorded[image_id]=item
        matched[image_id]=item['concepts']
        if len(recorded)%args.save_every==0:
            write_json(work/'concepts_checkpoint.json',recorded)
    if len(recorded)!=len(ids):raise AssertionError('Discovery image ID coverage mismatch')
    write_json(audit,recorded)
    write_json(detailed,matched)
    write_json(work/'settings.json',{'scheme':label or scheme,'teacher':str(teacher_weights),
        'gallery':args.gallery,'images':len(ids),'candidates':len(gallery),
        'num_selected':sum(len(x) for x in matched.values()),'arguments':vars(args)})
    print('[SAVED]',audit,flush=True)

def prepare_base_train(args,work,detailed):
    base_path=ROOT/args.base_train
    required(base_path)
    matched=read_json(detailed)
    selected=set(int(x) for x in matched)
    # Original cache may use 5k, 80k, etc; never silently change sample population.
    base=torch.load(base_path,map_location='cpu')
    all_ids=set(int(x['id']) for x in base['images'])
    if selected==all_ids:return base_path
    if not args.subset_train:
        extra=sorted(all_ids-selected)[:8];missing=sorted(selected-all_ids)[:8]
        raise RuntimeError(f'Discovery/base IDs differ (selected {len(selected)}, base {len(all_ids)}). '
            f'Examples: unmined={extra}, not-in-base={missing}. Pass --subset-train to explicitly train on intersection.')
    if not selected.issubset(all_ids):raise RuntimeError('Discovery includes images missing in base dataset')
    base['images']=[x for x in base['images'] if int(x['id']) in selected]
    base['annotations']=[x for x in base['annotations'] if int(x['image_id']) in selected]
    subset=work/'base_train_subset.pth';torch.save(base,subset)
    write_json(work/'subset_train_stats.json',{'original_images':len(all_ids),'selected_images':len(selected),
              'explicit_subset_train':True})
    return subset

def do_build(scheme,args,work):
    detailed=work/'matched'/'matched_concepts_detailed.json'
    required(detailed)
    target=work/'train_augmented.pth'
    if target.exists() and not args.force_build:
        print('[CACHE]',target);return
    base=prepare_base_train(args,work,detailed)
    run([sys.executable,'experiments/local_cocu_novel/build_train.py',
         '--base-train',base,'--matched',detailed,'--output',target,
         '--manifest',work/'train_manifest.json','--max-novel',str(args.max_concepts),
         '--clip-model','ViT-B/16','--batch-size',str(args.encode_batch),'--device',args.device],args)
    required(target)
    # Original 5 COCO captions preserved by build_train.py, plus <=K novel concepts.

def proj_name(scheme,round_id=None):
    name=f'vitb_mlp_infonce_local_cocu_talk2dino_{scheme}'
    return f'{name}_r{round_id}' if round_id is not None else name

def do_train(scheme,args,work,round_id=None):
    tr=work/'train_augmented.pth';required(tr);required(ROOT/args.val)
    name=proj_name(scheme,round_id)
    target=WEIGHTS/NAMES[scheme]/(name+'.pth')
    temporary=ROOT/'weights'/(name+'.pth')
    if target.exists() and not args.force_train:
        print('[CACHE] trained',target);ensure_eval_link(target,temporary);return target
    if temporary.is_symlink():
        if temporary.resolve()!=target.resolve(): raise RuntimeError(f'Conflicting root symlink {temporary}')
        temporary.unlink() # original train.py always saves to weights/name.pth
    elif temporary.exists():
        raise FileExistsError(f'Unexpected non-symlink weights path: {temporary}; refusing overwrite')
    target.parent.mkdir(parents=True,exist_ok=True)
    run([sys.executable,'train.py','--model_config',args.config,
         '--train_dataset',tr,'--val_dataset',args.val,
         '--feature_name','disentangled_self_attn','--text_features','ann_feats',
         '--name_pedix',name[len('vitb_mlp_infonce_'):]],args)
    required(temporary)
    shutil.move(str(temporary),str(target))
    ensure_eval_link(target,temporary)
    print('[SAVED WEIGHT]',target,flush=True)
    return target

def ensure_eval_link(target,link):
    # OpenVocabularySegmenter looks up weights/<proj_name>.pth, not arbitrary paths.
    if link.is_symlink():
        if link.resolve()==target.resolve():return
        raise RuntimeError(f'Refusing to replace foreign symlink {link}')
    if link.exists():raise RuntimeError(f'Refusing to replace existing checkpoint {link}')
    link.symlink_to(target.relative_to(ROOT/'weights'))

def do_eval(scheme,args,work,round_id=None):
    name=proj_name(scheme,round_id)
    target=WEIGHTS/NAMES[scheme]/(name+'.pth');required(target)
    ensure_eval_link(target,ROOT/'weights'/(name+'.pth'))
    out=work/'eval';out.mkdir(parents=True,exist_ok=True)
    cfg='src/open_vocabulary_segmentation/configs'
    for index,(task,(e,b)) in enumerate(EVALS.items()):
        path=out/task
        if (path/'eval_complete.json').exists() and not args.force_eval:
            print('[CACHE] evaluation',task);continue
        path.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'src/open_vocabulary_segmentation/main.py',
             '--eval','--eval_cfg',f'{cfg}/{e}',
             '--eval_base_cfg',f'{cfg}/{b}',
             '--output',path,'--opts',f'model.proj_name={name}']
        started=time.time()
        run(cmd,args,{'MASTER_ADDR':'127.0.0.1',
                      'MASTER_PORT':str(29501+index),'RANK':'0','WORLD_SIZE':'1',
                      'LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'1'})
        # Original code may share the segmentation_results file among different
        # checkpoints. Snapshot only a freshly written result after EACH dataset.
        snapshot=[]
        for artifact in (ROOT/'segmentation_results').glob('*.json') if (ROOT/'segmentation_results').exists() else []:
            if artifact.stat().st_mtime>=started-1:
                shutil.copy2(artifact,path/artifact.name)
                snapshot.append(artifact.name)
        write_json(path/'eval_complete.json',{'checkpoint':str(target.relative_to(ROOT)),
                   'dataset':task,'source_config':e,'base_config':b,'snapshots':snapshot})

def parser(scheme):
    p=argparse.ArgumentParser(description=f'Talk2DINO independent scheme {scheme} — original train.py, original segmentation eval')
    p.add_argument('--stage',choices=['mine','build','train','eval','all'],default='all')
    p.add_argument('--gpu',default='0')
    p.add_argument('--device',default='cuda')
    p.add_argument('--teacher',default='weights/vitb_mlp_infonce.pth')
    p.add_argument('--config',default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--heads',default='outputs/local_cocu/head_features/head_features.pt')
    p.add_argument('--coco',default='data/coco2014/annotations/captions_train2014.json')
    p.add_argument('--images',default='data/coco2014/train2014')
    p.add_argument('--gallery',default='experiments/local_cocu/MSCOCO14_train_gallery.json')
    p.add_argument('--base-train',default='outputs/coco2014_b14/train.pth')
    p.add_argument('--val',default='outputs/coco2014_b14/val.pth')
    p.add_argument('--max-concepts',type=int,default=3)
    p.add_argument('--pool-k',type=int,default=30)
    p.add_argument('--per-head-k',type=int,default=50)
    p.add_argument('--threshold',type=float,default=None)
    p.add_argument('--encode-batch',type=int,default=256)
    p.add_argument('--save-every',type=int,default=5000)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--subset-train',action='store_true')
    p.add_argument('--extract-missing-heads',action='store_true')
    p.add_argument('--extract-missing-cls',action='store_true')
    p.add_argument('--cls-neighbors',default='outputs/local_cocu/retrieval/neighbors.json')
    p.add_argument('--neighbor-topk',type=int,default=50)
    p.add_argument('--neighbor-per-head',type=int,default=60)
    p.add_argument('--retrieval-batch',type=int,default=128)
    p.add_argument('--faiss-index',choices=['hnsw','flat'],default='hnsw')
    p.add_argument('--faiss-ef',type=int,default=128)
    p.add_argument('--candidate-cap',type=int,default=300)
    p.add_argument('--uncovered-heads',type=int,default=4)
    p.add_argument('--attention-min-concentration',type=float,default=1.3)
    p.add_argument('--crop-quantile',type=float,default=.90)
    p.add_argument('--global-ratio',type=float,default=.5)
    p.add_argument('--local-threshold',type=float,default=.1)
    p.add_argument('--support-neighbors',type=int,default=20)
    p.add_argument('--min-support',type=int,default=2)
    p.add_argument('--support-margin',type=float,default=.05)
    p.add_argument('--rare-high',type=float,default=.55)
    p.add_argument('--negative-topk',type=int,default=30)
    p.add_argument('--counterfactual-lambda',type=float,default=.3)
    p.add_argument('--min-views',type=int,default=2)
    p.add_argument('--view-threshold',type=float,default=.12)
    p.add_argument('--view-variability',type=float,default=.2)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--force-mine',action='store_true')
    p.add_argument('--force-build',action='store_true')
    p.add_argument('--force-train',action='store_true')
    p.add_argument('--force-eval',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p

def main(scheme):
    if scheme not in NAMES:raise ValueError(scheme)
    args=parser(scheme).parse_args()
    if args.max_concepts<1 or args.pool_k<args.max_concepts or args.per_head_k<1:
        raise SystemExit('Require 1 <= max-concepts <= pool-k and per-head-k >= 1')
    if args.limit<0 or args.rounds<1:raise SystemExit('Bad --limit/--rounds')
    if args.device=='cuda' and not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError('CUDA is required for requested --device cuda; check PyTorch or select --device cpu')
    work=OUTPUT/NAMES[scheme]
    if args.dry_run:
        print('[DRY-RUN] No GPU model loaded and no stages executed.')
        print('  scheme:',scheme,'stage:',args.stage,'GPU:',args.gpu)
        print('  Teacher:',ROOT/args.teacher,'train:',ROOT/args.base_train)
        print('  output:',work,'checkpoint folder:',WEIGHTS/NAMES[scheme])
        return
    if args.stage in ('mine','all'):
        required(ROOT/args.teacher);required(ROOT/args.config);required(ROOT/args.coco)
        required(ROOT/args.gallery)
    work.mkdir(parents=True,exist_ok=True)
    if scheme!='H':
        if args.stage in ('mine','all'):do_mine(scheme,args,work,ROOT/args.teacher)
        if args.stage in ('build','all'):do_build(scheme,args,work)
        if args.stage in ('train','all'):do_train(scheme,args,work)
        if args.stage in ('eval','all'):do_eval(scheme,args,work)
    else:
        # H uses A direct retrieval as a fixed discovery operator, updating
        # ONLY teacher checkpoint between rounds; student ALWAYS new-init via train.py.
        teacher=ROOT/args.teacher
        for step in range(1,args.rounds+1):
            one=work/f'round_{step}'
            one.mkdir(parents=True,exist_ok=True)
            if args.stage in ('mine','all'):do_mine('A',args,one,teacher,label='H')
            if args.stage in ('build','all'):do_build('H',args,one)
            if args.stage in ('train','all'):do_train('H',args,one,step)
            if args.stage in ('eval','all'):do_eval('H',args,one,step)
            teacher=WEIGHTS/NAMES['H']/(proj_name('H',step)+'.pth')
            if args.stage in ('mine','all') and step<args.rounds and not teacher.exists():
                raise RuntimeError('H next round needs prior Student checkpoint. Use --stage all.')
    print('[DONE] scheme=',scheme,'stage=',args.stage,'output=',work,flush=True)
