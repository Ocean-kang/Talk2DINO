"""Reuse original ProjectionLayer, ContrastiveLoss, optimizer and validation.

Only E3/E4/C add a region-concept loss. No teacher initialization of student.
Global E2/E3/E4/C examples and fixed update budgets match exactly.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import numpy as np
from .common import read, write, digest, ROOT, offline
from .runtime import load_pth, save_pth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['E0','E1','E2','E3','E4','E3raw','C'],required=True)
    p.add_argument('--train',default='outputs/coco2014_b14/train.pth')
    p.add_argument('--val',default='outputs/coco2014_b14/val.pth')
    p.add_argument('--prepared',help='Full prepare output; not --recovery output')
    p.add_argument('--model-config',default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--epochs',type=int,help='Default from original YAML')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--optimizer',choices=['Adam','AdamW'],default='Adam')
    p.add_argument('--weight-decay',type=float,default=.05)
    p.add_argument('--scheduler',choices=['linear','cosine'],default='linear')
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--local-weight',type=float,default=.1)
    p.add_argument('--local-temperature',type=float,default=.07)
    p.add_argument('--gate-report',help='Idea A report required by C')
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();offline()
    import torch
    import torch.nn.functional as F
    import yaml
    from torch.utils.data import Dataset, DataLoader
    from src.model import ProjectionLayer
    from src.loss import ContrastiveLoss
    from src.train_util import set_seed, validate, const_lr, cosine_lr
    from src.dataset import DinoClipDataset
    if a.local_temperature<=0 or a.local_weight<0:raise ValueError('Invalid local-loss hyperparameters')
    cfg=yaml.safe_load(Path(a.model_config).read_text());train_cfg=cfg['train']
    epochs=a.epochs or train_cfg['num_epochs'];bs=train_cfg['batch_size']
    if train_cfg['ltype']!='infonce':raise ValueError('This experiment requires original infonce configuration')
    source=None;all_records={};bank=None;extra=[]
    if a.method!='E0':
        if not a.prepared:raise ValueError('--prepared is required')
        prepared=Path(a.prepared);source=read(prepared/'source.json')
        if source['recovery_only']:raise ValueError('Recovery caches are diagnostic only; cannot train')
        if source['captions_sha256']!=digest(a.train):
            raise ValueError('Prepare with the SAME original train.pth passed to training; JSON preparation is for calibration/diagnosis')
        bank=load_pth(prepared/'text_bank.pth')
        checks=read(prepared/'feature_checks.json')
        if checks['count']<1:raise ValueError('Training requires preparation with >=1 passing existing-feature check')
        if source['models']['config_sha256']!=digest(a.model_config):raise ValueError('Model configuration differs from preparation')
        for iid in source['image_ids']:
            rec=load_pth(prepared/'records'/(str(iid)+'.pth'))
            if rec['signature']!=source['signature']:raise ValueError('Stale region record')
            rec.pop('audit',None)
            rec.pop('candidate_ids',None)
            all_records[iid]=rec
        if not any(r['selected'] for r in all_records.values()):
            raise ValueError('No selected concepts; inspect calibration/summary before training B')
        if a.method=='C':
            if not a.gate_report:raise ValueError('C requires --gate-report from Idea A')
            gate=read(a.gate_report)
            if gate.get('kind')!='idea_a_diagnostic' or not gate.get('c_eligible_for_pilot'):
                raise ValueError('Idea A gate has not passed; do not run C')
            for key in ('teacher_sha256','dino_sha256','clip_sha256','size'):
                if gate['model'][key]!=source['models'][key]:raise ValueError('C gate uses a different teacher/model/size')
    raw=DinoClipDataset(a.train,features_name='disentangled_self_attn',text_features='ann_feats')
    validation=DinoClipDataset(a.val,features_name='avg_self_attn_out',text_features='ann_feats')
    if {int(r['image_id']) for r in raw.data.values()} & {int(r['image_id']) for r in validation.data.values()}:
        raise ValueError('Train/validation image overlap')
    # Preserve every original caption; no hard-coded 5 captions or silent truncation.
    by_image={}
    for idx in range(len(raw)):
        sample=raw.data[idx];iid=int(sample['image_id']);by_image.setdefault(iid,[]).append(idx)
        if tuple(sample['image'].shape)!=(12,768) or tuple(sample['annotation'].shape)!=(512,):
            raise ValueError('Expected original [12,768] head and [512] CLIP feature tensors')
    if all_records and set(by_image)!=set(all_records):
        raise ValueError('Prepared image set must exactly equal original train.pth image set; use full preparation for full training')
    if a.method not in ('E0','E3raw'):
        for iid in sorted(all_records):
            key='head_control' if a.method=='E1' else 'selected'
            for r in all_records[iid][key]:extra.append((iid,r['gallery_id']))
    class GlobalData(Dataset):
        def __len__(self):return len(raw)+len(extra)
        def __getitem__(self,index):
            if index<len(raw):
                b=raw[index];return b['image'].float(),b['annotation'].float(),int(b['metadata']['image_id'])
            iid,gid=extra[index-len(raw)]
            return raw.data[by_image[iid][0]]['image'].float(),torch.as_tensor(bank['raw'][gid],dtype=torch.float32),iid
    data=GlobalData()
    # All methods execute the original raw annotation budget per epoch.
    # Expanded pools use a seeded uniform permutation prefix; no extra optimizer steps.
    updates=math.ceil(len(raw)/bs)
    set_seed(a.seed);model=ProjectionLayer.from_config(cfg['model']).to(a.device);set_seed(a.seed)
    criterion=ContrastiveLoss(model,margin=train_cfg.get('margin',.2),max_violation=train_cfg.get('max_violation',True),ltype='infonce')
    if a.optimizer=='Adam':opt=torch.optim.Adam(model.parameters(),lr=train_cfg['lr'])
    else:opt=torch.optim.AdamW(model.parameters(),lr=train_cfg['lr'],weight_decay=a.weight_decay)
    total_steps=updates*epochs
    scheduler=None
    if a.scheduler=='cosine':scheduler=cosine_lr(opt,train_cfg['lr'],a.warmup,total_steps)
    elif a.warmup:scheduler=const_lr(opt,train_cfg['lr'],a.warmup,total_steps)
    val_loader=DataLoader(validation,batch_size=bs,shuffle=False,num_workers=a.workers)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    settings={**vars(a),'config':cfg,'train_sha256':digest(a.train),'val_sha256':digest(a.val),
              'prepared_signature':source['signature'] if source else None,'raw_annotations':len(raw),'extra_annotations':len(extra),
              'updates_per_epoch':updates,'epochs_actual':epochs,'budget':'original raw annotation batches; expanded uniform pool prefix',
              'local_scale':'mean local CE / global_batch_size**2, then local_weight',
              'selection':'last checkpoint primary; best raw-caption validation also saved',
              'region_records':sum(len(r['selected']) for r in all_records.values()),
              'adaptive_records':sum(not np.allclose(x['region'],x['adaptive_region']) for r in all_records.values() for x in r['selected'])}
    comparison={k:v for k,v in settings.items() if k not in ('resume','output')}
    if (out/'settings.json').exists():
        prior=read(out/'settings.json');prior={k:v for k,v in prior.items() if k not in ('resume','output')}
        if prior!=comparison:raise ValueError('Training settings changed; choose a fresh output')
        if not a.resume:raise FileExistsError('Run exists; --resume or choose fresh output')
    write(out/'settings.json',settings)
    start=0;best=float('inf')
    if a.resume:
        state=load_pth(out/'resume.pth');model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=state['epoch']+1;best=state['best'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    log=out/'history.jsonl'
    if a.resume and log.exists():
        rows=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        log.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<=start))
    projected_raw=torch.as_tensor(bank['raw'],dtype=torch.float32,device=a.device) if bank else None
    def local_loss(image_ids, epoch, batch_index, global_size):
        regions=[];gids=[];negs=[]
        for slot,iid in enumerate(image_ids):
            rec=all_records[int(iid)];pool=rec['selected']
            if not pool:continue
            # Hash-like deterministic choice, no global RNG consumption; E2/E3 same global batches.
            choice=pool[(epoch+batch_index+slot)%len(pool)]
            region=choice['adaptive_region'] if a.method=='C' else choice['region']
            if a.method=='E4':
                # Same text/negative IDs and region multiset size; choose another image
                # whose captions/selected concepts don't mark this concept as positive.
                eligible=[]
                for other in candidate_donors(int(iid),choice['gallery_id']):
                    eligible=all_records[other]['selected']
                    if eligible:break
                if not eligible:raise ValueError('E4 has no eligible mismatched donor; cannot maintain the local-example count')
                region=eligible[(epoch+slot)%len(eligible)]['region']
            if region is None:continue
            regions.append(region);gids.append(choice['gallery_id']);negs.append(choice['negative_ids'])
        if not regions:return model.linear_layer.weight.sum()*0,0
        # Padded per-row negative dictionaries; one batched projector forward.
        length=max(len(n) for n in negs)+1
        ids=np.zeros((len(gids),length),dtype=np.int64)
        valid=np.zeros_like(ids,dtype=bool)
        for j,(gid,negative) in enumerate(zip(gids,negs)):
            ids[j,:len(negative)+1]=[gid]+negative
            valid[j,:len(negative)+1]=True
        unique,inverse=np.unique(ids,return_inverse=True)
        text=F.normalize(model.project_clip_txt(projected_raw[unique.tolist()]),dim=-1)
        text=text[torch.as_tensor(inverse,device=a.device)].reshape(len(gids),length,-1)
        visual=F.normalize(torch.as_tensor(np.stack(regions),dtype=torch.float32,device=a.device),dim=-1)
        logits=torch.einsum('nld,nd->nl',text,visual)/a.local_temperature
        logits=logits.masked_fill(~torch.as_tensor(valid,device=a.device),float('-inf'))
        loss=F.cross_entropy(logits,torch.zeros(len(gids),dtype=torch.long,device=a.device))
        return loss/(global_size**2),len(regions)
    donors=sorted(all_records)
    def candidate_donors(iid,gid):
        # Bound lookup cost; fixed cyclic donor search, no RNG use.
        if not donors:return
        start=(iid*104729+gid)%len(donors)
        for offset in range(min(len(donors),128)):
            other=donors[(start+offset)%len(donors)]
            rec=all_records[other]
            positives=set(rec['known_covered_ids'])|{r['gallery_id'] for r in rec['selected']}
            if other!=iid and gid not in positives:yield other
    for epoch in range(start,epochs):
        # Match upstream E0 shuffle/RNG and final partial batch exactly.
        if not extra:
            loader=DataLoader(data,batch_size=bs,shuffle=train_cfg.get('shuffle',True),num_workers=a.workers)
        else:
            # Same last-batch size as raw baseline, with exact same step count.
            gen=torch.Generator().manual_seed(a.seed+epoch)
            order=torch.randperm(len(data),generator=gen)[:len(raw)].tolist()
            loader=DataLoader(data,batch_size=bs,sampler=order,num_workers=a.workers)
        model.train(); sums=np.zeros(3);local_count=0;steps=0
        for batch_index,(heads,annotations,image_ids) in enumerate(loader):
            if scheduler:scheduler(epoch*updates+batch_index)
            heads=heads.to(a.device);annotations=annotations.to(a.device)
            global_loss=criterion(heads,annotations)
            aux=global_loss*0;n=0
            if a.method in ('E3','E4','E3raw','C'):
                aux,n=local_loss(image_ids.tolist(),epoch,batch_index,len(heads))
            loss=global_loss+a.local_weight*aux
            opt.zero_grad();loss.backward();opt.step()
            sums+=np.array([loss.item(),global_loss.item(),aux.item()]);local_count+=n;steps+=1
        model.eval();val=validate(model,val_loader,criterion)
        means=sums/max(steps,1);record={'epoch':epoch+1,'loss':float(means[0]),'global_loss':float(means[1]),
            'local_loss_scaled_by_B2':float(means[2]),'local_examples':local_count,'raw_val_loss':val,
            'steps':steps,'lr':opt.param_groups[0]['lr']}
        if steps!=updates:raise AssertionError('Step budget mismatch')
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        if val<best:
            best=val;save_pth(out/'best.pth',model.state_dict())
        save_pth(out/'last.pth',model.state_dict())
        if a.checkpoint_every>0 and (epoch+1)%a.checkpoint_every==0:save_pth(out/('epoch_%03d.pth'%(epoch+1)),model.state_dict())
        save_pth(out/'resume.pth',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
        print(record,flush=True)
    write(out/'result.json',{'method':a.method,'seed':a.seed,'checkpoint':str((out/'last.pth').resolve()),
                            'best_checkpoint':str((out/'best.pth').resolve()),'best_raw_val_loss':best,
                            'primary':'last.pth; original config save_best_model=False', 'epochs':epochs})


if __name__=='__main__':main()
