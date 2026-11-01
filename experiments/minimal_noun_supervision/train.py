"""Reuse upstream ProjectionLayer, ContrastiveLoss, train, validate and schedulers.
Only orchestration/sampling change: exact update budget, pool sampling, CLI seed.
"""
import argparse
import csv
import math
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.io import load_features, read_json, sha256, write_json
from common.cache import offline, validate_bank
from common.data import PromptPoolDataset, ImageBatches, reference_budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['raw', 'cne', 'hvne', 'cnp'], required=True)
    p.add_argument('--data', default='outputs/minimal_noun_supervision/data')
    p.add_argument('--config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--tag', default='full', help='Unique data/run label; letters/numbers/underscore only')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--optimizer', choices=['Adam', 'AdamW'], default='Adam')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--scheduler', choices=['linear', 'cosine'], default='linear')
    p.add_argument('--warmup', type=int, default=0, help='Optimizer steps, matching upstream implementation')
    p.add_argument('--max-steps', type=int, default=None, help='Smoke test override; formal runs omit this')
    p.add_argument('--validate-every-passes', type=int, default=None)
    args = p.parse_args()
    if not args.tag or not all(c.isalnum() or c == '_' for c in args.tag):
        p.error('Use a nonempty tag with letters/numbers/underscore')
    if args.workers < 0 or args.warmup < 0 or (args.max_steps is not None and args.max_steps <= 0):
        p.error('Invalid workers/warmup/max-steps')
    if args.validate_every_passes is not None and args.validate_every_passes < 1:
        p.error('validate-every-passes must be positive')
    offline()
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from src.model import ProjectionLayer
    from src.loss import ContrastiveLoss
    from src.train_util import set_seed, train, validate, const_lr, cosine_lr
    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg, train_cfg = cfg['model'], cfg['train']
    if model_cfg.get('dino_embed_dim', 1024) != 768 or model_cfg.get('clip_embed_dim', 512) != 512:
        raise ValueError('This experiment requires the audited ViT-B 512-to-768 projector')
    if model_cfg.get('starting_checkpoint') is not None:
        raise ValueError('Student must start randomly; remove starting_checkpoint')
    if model_cfg.get('model_class', 'ProjectionLayer') != 'ProjectionLayer' or model_cfg.get('weight_attn_heads') is not None:
        raise ValueError('This controlled experiment requires the original ProjectionLayer without trainable head weighting')
    if model_cfg.get('alignment_strategy', 'max_score') != 'max_score' or train_cfg['ltype'] != 'infonce':
        raise ValueError('Expected original max_score + infonce protocol')
    stem = 'vitb_mlp_infonce_mns_%s_%s_seed%d' % (args.tag, args.method, args.seed)
    output = ROOT / 'outputs/minimal_noun_supervision' / args.tag / args.method / ('seed_%d' % args.seed)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Run exists; choose another tag/seed or explicitly remove the old run')
    weight = ROOT / 'weights' / (stem + '.pth')
    if weight.exists():
        raise FileExistsError(str(weight))
    data_root = Path(args.data).resolve()
    m = read_json(data_root / args.method / 'manifest.json')
    vm = read_json(data_root / 'shared/validation.json')
    if m['method'] != args.method:
        raise ValueError('Manifest method mismatch')
    raw = load_features(m['source_features'])
    val = load_features(vm['source_features'])
    images = {int(x['id']): x for x in raw['images']}
    val_images = {int(x['id']): x for x in val['images']}
    if set(images) & set(val_images):
        raise ValueError('Train/val image overlap')
    provenance_file = data_root / 'shared/provenance.json'
    if not provenance_file.is_file():
        raise ValueError('Regenerate data with this gallery-based version; old manifests are incompatible')
    provenance = read_json(provenance_file)
    if sha256(m['source_features']) != provenance['train_source_sha256'] or sha256(vm['source_features']) != provenance['val_source_sha256']:
        raise ValueError('Source features changed after preparation')
    if sha256(m['text_bank']) != provenance['text_bank_sha256']:
        raise ValueError('Text cache changed after preparation')
    gallery = read_json(data_root / 'shared/gallery.json')
    bank = validate_bank(torch.load(m['text_bank'], map_location='cpu'), provenance['source_metadata'], gallery)
    expected_ids = provenance['train_image_ids']
    if [int(r['image_id']) for r in m['records']] != expected_ids:
        raise ValueError('All methods and Raw must use exactly the same retained image IDs')
    if m['raw_annotation_count'] != provenance['raw_annotation_count']:
        raise ValueError('Training budget drift')
    train_data = PromptPoolDataset(m['records'], images, bank, seed=args.seed)
    val_data = PromptPoolDataset(vm['records'], val_images, bank, fixed=True, feature_name='avg_self_attn_out')
    if any(tuple(images[int(r['image_id'])]['disentangled_self_attn'].shape) != (12, 768) for r in m['records']):
        raise ValueError('Train features must have shape [12,768]')
    batch_size = int(train_cfg['batch_size'])
    budget = reference_budget(m['raw_annotation_count'], batch_size, int(train_cfg['num_epochs']))
    if args.max_steps is not None:
        budget = args.max_steps
    if args.warmup >= budget and args.warmup != 0:
        raise ValueError('Warmup must be smaller than the step budget')
    every = args.validate_every_passes or max(1, round(m['raw_annotation_count'] / len(train_data)))
    if every < 1:
        p.error('validate-every-passes must be positive')
    # Seed BEFORE initializing the student, unlike a seed applied only in do_train.
    set_seed(args.seed)
    model = ProjectionLayer.from_config(model_cfg).to(args.device)
    criterion = ContrastiveLoss(model, margin=train_cfg.get('margin', 0.2),
        max_violation=train_cfg.get('max_violation', True), ltype=train_cfg['ltype']).to(args.device)
    # Exactly upstream optimizer parameter ownership: logit_scale is NOT added.
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = cosine_lr(optimizer, train_cfg['lr'], args.warmup, budget)
    elif args.warmup:
        scheduler = const_lr(optimizer, train_cfg['lr'], args.warmup, budget)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=args.workers)
    output.mkdir(parents=True, exist_ok=True)
    weight.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'arguments': vars(args), 'config': cfg, 'optimizer_updates': budget,
        'reference_annotation_count': m['raw_annotation_count'], 'train_images': len(train_data),
        'validation_images': len(val_data), 'validation_protocol': 'one fixed RAW prompt per image; original avg_self_attn_out',
        'save_best_model': train_cfg.get('save_best_model', True), 'manifest_sha256': sha256(data_root / args.method / 'manifest.json'),
        'validation_manifest_sha256': sha256(data_root / 'shared/validation.json'),
        'upstream_sources_sha256': {f: sha256(ROOT / f) for f in ('src/model.py', 'src/loss.py', 'src/train_util.py')},
        'checkpoint_stem': stem, 'torch_version': torch.__version__, 'image_passes': math.ceil(budget / math.ceil(len(train_data) / batch_size))}
    write_json(output / 'run.json', provenance)
    step, epoch, best, selected_pass = 0, 0, float('inf'), None
    best_path, last_path = output / 'best.pth', output / 'last.pth'
    with (output / 'losses.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_pass', 'optimizer_steps', 'train_loss', 'validation_loss'])
        writer.writeheader()
        while step < budget:
            train_data.set_epoch(epoch)
            batches = ImageBatches(len(train_data), batch_size, args.seed, epoch, budget - step)
            loader = DataLoader(train_data, batch_sampler=batches, num_workers=args.workers)
            # Upstream train computes offsets as epoch*len(loader); the final pass
            # can be partial, so bind the ACTUAL global offset without patching it.
            def schedule(local_step, offset=step):
                return scheduler(offset + local_step)
            model.train()
            train_loss = train(model, loader, criterion, optimizer,
                scheduler=schedule if scheduler else None, n_epochs=0)
            step += len(loader)
            epoch += 1
            if not math.isfinite(train_loss):
                raise RuntimeError('Non-finite training loss')
            val_loss = None
            if epoch % every == 0 or step == budget:
                model.eval()
                val_loss = validate(model, val_loader, criterion)
                if not math.isfinite(val_loss):
                    raise RuntimeError('Non-finite validation loss')
                if val_loss < best:
                    best, selected_pass = val_loss, epoch
                    torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), last_path)
            writer.writerow({'image_pass': epoch, 'optimizer_steps': step, 'train_loss': train_loss, 'validation_loss': val_loss})
            stream.flush()
            print('pass=%d steps=%d/%d train=%.8g val=%s' % (epoch, step, budget, train_loss, val_loss), flush=True)
    use_best = train_cfg.get('save_best_model', True)
    from shutil import copyfile
    copyfile(best_path if use_best else last_path, weight)
    write_json(output / 'result.json', {'checkpoint': str(weight), 'checkpoint_stem': stem,
        'selected': 'best' if use_best else 'last', 'best_validation_loss': best,
        'best_pass': selected_pass, 'optimizer_updates': step, 'image_passes': epoch})
    print('Saved ' + str(weight))


if __name__ == '__main__':
    main()
