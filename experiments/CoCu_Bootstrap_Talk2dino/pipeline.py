#!/usr/bin/env python3
"""Ponytail experiment dispatcher for six standalone CoCu->Bootstrap arms.

Use from Talk2DINO repository root.  Reuses *existing repo* CoCu extraction,
caption fusion, Bootstrap implementation; never patches core code.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]  # Talk2DINO (HERE=Talk2DINO/experiments/CoCu_Bootstrap_Talk2dino)
COCO='data/coco2014/annotations/captions_train2014.json'
IMAGES='data/coco2014/train2014'
BASE='outputs/coco2014_b14/train.pth'
VAL='outputs/coco2014_b14/val.pth'
CFG='configs/vitb_mlp_infonce.yaml'
TEACHER='weights/vitb_mlp_infonce.pth'
CLIP_GALLERY='experiments/cocu_imagelevel_compare/MSCOCO14_train_gallery_exactmatch.json'
LOCAL_GALLERY='experiments/local_cocu/MSCOCO14_train_gallery.json'
OUT='outputs/CoCu_Bootstrap_Talk2dino'


def call(argv, gpu, dry=False):
    cmd=[sys.executable if str(argv[0]) == 'python' else str(argv[0]), *[str(x) for x in argv[1:]]]
    print('\n+ '+ ' '.join(cmd),flush=True)
    if dry:
        return
    env=os.environ.copy()
    env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env.setdefault('PYTHONHASHSEED', '123')  # stable ordering for set-based Local candidate mining
    subprocess.run(cmd, cwd=ROOT, env=env,check=True)


def must_exist(*names, dry=False):
    if dry:
        return
    for name in names:
        if not (ROOT/name).exists():
            raise FileNotFoundError(f'Missing required repository input: {ROOT/name}')


def concept_paths(source):
    base=Path(OUT)/'concepts'/source
    if source=='clip':
        source_json=base/'clip_vitb16.json'
    else:
        source_json=base/'local_cocu_compare.json'
    return base,source_json,base/'concepts_manifest.json'


def concept_extract_clip(base,source_json,args):
    must_exist(COCO,IMAGES,CLIP_GALLERY,
               'experiments/cocu_imagelevel_compare/build_curated_concepts.py',dry=args.dry_run)
    cmd=['python','experiments/cocu_imagelevel_compare/build_curated_concepts.py',
         '--backend','clip','--captions-json',COCO,'--images-dir',IMAGES,
         '--gallery',CLIP_GALLERY,'--output',source_json,'--cache-dir',base/'cache',
         '--clip-model','ViT-B/16','--top-k','8','--num-curated','3','--device','cuda']
    if args.rebuild_concepts:
        cmd+=['--overwrite']
    call(cmd,args.gpu,args.dry_run)


def concept_extract_local(base,source_json,args):
    must_exist(COCO,IMAGES,LOCAL_GALLERY,CFG,TEACHER,dry=args.dry_run)
    script='experiments/local_cocu/'
    p_cls=base/'dino_cls'/'dino_cls.pt'
    p_neighbors=base/'retrieval'/'neighbors.json'
    p_candidates=base/'candidate'/'candidate_concepts.json'
    p_head=base/'head_features'/'head_features.pt'
    p_match=base/'matched'
    commands=[
        ['python',script+'extract_dino_cls_features.py','--caption',COCO,
         '--image_dir',IMAGES,'--output',base/'dino_cls'],
        ['python',HERE.relative_to(ROOT)/'shared/chunked_retrieval.py',
         '--feature',p_cls,'--output',base/'retrieval','--topk','50',
         '--chunk-size',str(args.retrieval_chunk),'--device','cuda'],
        ['python',script+'extract_candidate_concepts.py','--neighbors',p_neighbors,
         '--caption',COCO,'--gallery',LOCAL_GALLERY,'--output',base/'candidate'],
        ['python',script+'extract_head_features.py','--annotation',COCO,
         '--image_dir',IMAGES,'--model','dinov2_vitb14_reg',
         '--resize_dim','448','--crop_dim','448','--batch_size',str(args.head_batch),
         '--output',p_head,'--fp16'],
        ['python',script+'match_head_concepts.py',
         '--candidate',p_candidates,'--head_features',p_head,
         '--projector_config',CFG,'--weight',TEACHER,'--clip_model','ViT-B/16',
         '--topk','10','--output',p_match],
        ['python',script+'build_local_compare.py','--caption_json',COCO,
         '--neighbors',p_neighbors,'--candidates',p_candidates,
         '--matched',p_match/'matched_concepts.json',
         '--detailed',p_match/'matched_concepts_detailed.json',
         '--output',source_json],
    ]
    for cmd in commands:
        call(cmd,args.gpu,args.dry_run)


def extract(source,args):
    base,source_json,manifest=concept_paths(source)
    if source_json.exists() and manifest.exists() and not args.rebuild_concepts:
        print(f'[CACHE] Curated concepts already generated: {source_json} (add --rebuild-concepts to rerun)',flush=True)
        return
    if not args.dry_run:
        base.mkdir(parents=True,exist_ok=True)
        # Prevent an old manifest being mistaken for the new concepts.
        if manifest.exists():
            manifest.unlink()
    if source=='clip':
        concept_extract_clip(base,source_json,args)
    else:
        concept_extract_local(base,source_json,args)
    call(['python',HERE.relative_to(ROOT)/'shared/prepare_manifest.py',
          '--source',source,'--source-json',source_json,'--base-train',BASE,
          '--output',manifest,'--num-curated','3'],args.gpu,args.dry_run)
    if not args.dry_run and not manifest.exists():
        raise RuntimeError('Concept extraction did not produce manifest')


def data_path(source,scheme):
    root=Path(OUT)/'runs'/f'{source}_{scheme}'
    if scheme=='5plus3':
        name='train_clip_vitb16_5plus3.pth' if source=='clip' else 'train_local_cocu_compare_5plus3.pth'
    else:
        name='train.pth'
    return root,root/name


def build(source,scheme,args):
    base,source_json,manifest=concept_paths(source)
    root,dataset=data_path(source,scheme)
    must_exist(BASE,CFG,str(source_json),str(manifest),dry=args.dry_run)
    if scheme=='5plus3':
        cmd=['python','experiments/bootstrap_5plus3/build_5plus3.py',
             '--method',source,'--base-train',BASE,
             '--output-dir',root,'--num-curated','3',
             '--clip-model','ViT-B/16','--batch-size','512',
             '--device','cuda']
        if source=='clip':
            cmd += ['--clip-json',source_json]
        else:
            cmd += ['--local-json',source_json]
        call(cmd,args.gpu,args.dry_run)
    else:
        script=('experiments/CaptionsConcept_Completion/B_filtered_semantic_residual/build.py'
                if scheme=='B' else
                'experiments/CaptionsConcept_Completion/D_concept_set_supplementary/build.py')
        cmd=['python',script,'--source','clip' if source=='clip' else 'local_talk2dino',
             '--manifest',manifest,'--base-train',BASE,'--output',dataset,
             '--audit',root/'audit.jsonl','--summary',root/'summary.json',
             '--clip-model','ViT-B/16','--batch-size','512',
             '--image-batch-size','256','--device','cuda',
             '--min-visual-ratio','0.85','--max-concepts','3']
        if scheme=='B':
            cmd+=['--semantic-floor','0.18','--consistency-margin','0.20',
                  '--redundancy-ceiling','0.90']
        call(cmd,args.gpu,args.dry_run)
    verify(source,scheme,args)
    return dataset


def verify(source,scheme,args):
    _,dataset=data_path(source,scheme)
    must_exist(str(dataset),BASE,dry=args.dry_run)
    call(['python',HERE.relative_to(ROOT)/'shared/check_dataset.py',
          '--dataset',dataset,'--base-train',BASE,'--expected',
          str({'5plus3':8,'B':5,'D':6}[scheme])],args.gpu,args.dry_run)


def train(source,scheme,args):
    _,dataset=data_path(source,scheme)
    must_exist(str(dataset),VAL,CFG,TEACHER,
               'experiments/teacher_guided_bootstrap/train_bootstrap.py',dry=args.dry_run)
    output=f'weights/vitb_mlp_infonce_cocu_bt_{source}_{scheme}_a1.pth'
    call(['python','experiments/teacher_guided_bootstrap/train_bootstrap.py',
          '--teacher-weights',TEACHER,'--train-dataset',dataset,
          '--val-dataset',VAL,'--model-config',CFG,'--alpha','1.0',
          '--seed','123','--output',output],args.gpu,args.dry_run)
    print(f'[OUTPUT] {output}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['clip','local'],required=True)
    p.add_argument('--scheme',choices=['5plus3','B','D'],required=True)
    p.add_argument('--stage',choices=['concepts','build','verify','train','all'],default='all')
    p.add_argument('--gpu',default='0')
    p.add_argument('--rebuild-concepts',action='store_true',
                   help='Re-extract all concept features and candidates, ignoring source cache')
    p.add_argument('--dry-run',action='store_true',help='Show subprocess commands without GPU work')
    p.add_argument('--retrieval-chunk',type=int,default=256)
    p.add_argument('--head-batch',type=int,default=16)
    a=p.parse_args()
    if not a.dry_run and not (ROOT/'src/model.py').exists():
        raise RuntimeError(f'Run from a Talk2DINO checkout, expected {ROOT}/src/model.py')
    if a.stage in ('concepts','all'):
        extract(a.source,a)
    if a.stage in ('build','all'):
        build(a.source,a.scheme,a)
    if a.stage=='verify':
        verify(a.source,a.scheme,a)
    if a.stage in ('train','all'):
        train(a.source,a.scheme,a)

if __name__=='__main__':
    main()
