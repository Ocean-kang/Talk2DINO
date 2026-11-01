"""Launch the unmodified repository OVSS entry point and retain each task result."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common.io import read_json, sha256, write_json
from common.cache import offline

TASKS = {
    'voc20': ('voc', 'voc'),
    'voc': ('voc_bg', 'voc_bg'),
    'context59': ('context', 'context'),
    'ade20k': ('ade', 'ade'),
    'coco_stuff': ('stuff', 'stuff'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['raw', 'cne', 'hvne', 'cnp'], required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--tag', default='full')
    p.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    p.add_argument('--pamr', choices=['on', 'off'], default='on')
    p.add_argument('--port', type=int, default=29781)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.tag):
        p.error('tag must contain letters/numbers/underscore')
    offline()
    result_file = ROOT / 'outputs/minimal_noun_supervision' / args.tag / args.method / ('seed_%d' % args.seed) / 'result.json'
    result = read_json(result_file)
    stem = result['checkpoint_stem']
    checkpoint = ROOT / 'weights' / (stem + '.pth')
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    if not re.fullmatch(r'[A-Za-z0-9_]+', stem):
        raise ValueError('Invalid checkpoint stem')
    cfg_root = Path('src/open_vocabulary_segmentation/configs')
    for idx, task in enumerate(args.tasks):
        folder, prefix = TASKS[task]
        model_cfg = cfg_root / folder / ('dinotext_%s_vitb_mlp_infonce.yml' % prefix)
        base_cfg = cfg_root / folder / ('eval_%s%s.yml' % (prefix, '_pamr' if args.pamr == 'on' else ''))
        for cfg in (model_cfg, base_cfg):
            if not (ROOT / cfg).is_file():
                raise FileNotFoundError(str(cfg))
        out = result_file.parent / 'segmentation' / args.pamr / task
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=1',
            '--master_addr=127.0.0.1', '--master_port=' + str(args.port + idx),
            'src/open_vocabulary_segmentation/main.py', '--eval', '--eval_cfg', str(model_cfg),
            '--eval_base_cfg', str(base_cfg), '--output', str(out), '--opts', 'model.proj_name=' + stem]
        print(' '.join(command), flush=True)
        if args.dry_run:
            continue
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Evaluation output exists; use a new run or remove that task output explicitly: ' + str(out))
        out.mkdir(parents=True, exist_ok=True)
        # Original main.py accumulates results into this shared JSON. Snapshot its
        # previous bytes and remove only this task's stale entry, retaining others.
        source = ROOT / 'segmentation_results' / (stem + '.json')
        prior = read_json(source) if source.exists() else {}
        write_json(out / 'previous_shared_results.json', prior)
        prior.pop(task, None)
        write_json(source, prior)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(HERE / 'offline_guard') + os.pathsep + str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
        env['MASTER_PORT'] = str(args.port + idx)
        with (out / 'eval.log').open('w') as log:
            run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            raise RuntimeError('Original OVSS evaluator failed; see ' + str(out / 'eval.log'))
        obj = read_json(source)
        if task not in obj:
            raise RuntimeError('Evaluator did not produce the expected task key: ' + task)
        miou = float(obj[task])
        if not 0 <= miou <= 100:
            raise ValueError('Invalid mIoU: %s' % miou)
        write_json(out / 'metrics.json', {'method': args.method, 'seed': args.seed, 'task': task,
            'pamr': args.pamr, 'miou': miou, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'model_cfg_sha256': sha256(ROOT / model_cfg), 'eval_cfg_sha256': sha256(ROOT / base_cfg)})
        print('%s mIoU=%.4f' % (task, miou), flush=True)


if __name__ == '__main__':
    main()
