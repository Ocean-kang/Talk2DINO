"""Summarize completed seeds without silently pooling different protocols."""
import argparse
import csv
from pathlib import Path
import statistics
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common.io import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', default='full')
    p.add_argument('--pamr', choices=['on', 'off'], default='on')
    p.add_argument('--seeds', nargs='+', type=int, default=[123, 124, 125, 126, 127])
    p.add_argument('--methods', nargs='+', choices=['raw', 'cne', 'hvne', 'cnp'], default=['raw', 'cne', 'hvne', 'cnp'])
    p.add_argument('--tasks', nargs='+', default=['voc20', 'voc', 'context59', 'ade20k', 'coco_stuff'])
    args = p.parse_args()
    root = ROOT / 'outputs/minimal_noun_supervision' / args.tag
    rows, missing = [], []
    values = {}
    for method in args.methods:
        for task in args.tasks:
            vals = []
            for seed in args.seeds:
                path = root / method / ('seed_%d' % seed) / 'segmentation' / args.pamr / task / 'metrics.json'
                if not path.exists():
                    missing.append(str(path))
                    continue
                rec = read_json(path)
                if (rec['method'], rec['seed'], rec['task'], rec['pamr']) != (method, seed, task, args.pamr):
                    raise ValueError('Metric metadata mismatch: ' + str(path))
                values[method, task, seed] = float(rec['miou'])
                vals.append(float(rec['miou']))
            if len(vals) == len(args.seeds):
                row = {'method': method, 'task': task, 'n': len(vals), 'mean': statistics.mean(vals),
                       'std_sample': statistics.stdev(vals) if len(vals) > 1 else None}
                row.update({'seed_%d' % s: v for s, v in zip(args.seeds, vals)})
                rows.append(row)
    if missing:
        raise SystemExit('Missing results (no partial summary emitted):\n' + '\n'.join(missing))
    paired = []
    if 'raw' in args.methods:
        for method in args.methods:
            if method == 'raw':
                continue
            for task in args.tasks:
                delta = [values[method, task, s] - values['raw', task, s] for s in args.seeds]
                row = {'method_minus_raw': method, 'task': task, 'n': len(delta), 'mean_delta': statistics.mean(delta),
                    'std_delta_sample': statistics.stdev(delta) if len(delta) > 1 else None}
                row.update({'seed_%d' % s: d for s, d in zip(args.seeds, delta)})
                paired.append(row)
    root.mkdir(parents=True, exist_ok=True)
    for name, group in [('mean_std', rows), ('paired_vs_raw', paired)]:
        if group:
            with (root / (name + '_' + args.pamr + '.csv')).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(group[0]))
                w.writeheader()
                w.writerows(group)
    write_json(root / ('summary_' + args.pamr + '.json'), {'seeds': args.seeds, 'results': rows, 'paired_deltas': paired})
    print('Saved summaries under ' + str(root))


if __name__ == '__main__':
    main()
