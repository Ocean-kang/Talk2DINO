"""Post-training head-usage diagnostics; no pixel labels or invented IoU proxy."""
import argparse
from collections import Counter
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.io import load_features, read_json, write_json, sha256
from common.cache import offline, validate_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='outputs/minimal_noun_supervision/data')
    p.add_argument('--method', choices=['raw', 'cne', 'hvne', 'cnp'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', default='configs/vitb_mlp_infonce.yaml')
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    offline()
    import torch
    import torch.nn.functional as F
    import yaml
    from src.model import ProjectionLayer
    m = read_json(Path(args.data) / args.method / 'manifest.json')
    raw = load_features(m['source_features'])
    images = {int(x['id']): x for x in raw['images']}
    provenance = read_json(Path(args.data) / 'shared/provenance.json')
    if sha256(m['text_bank']) != provenance['text_bank_sha256']:
        raise ValueError('Text cache changed after preparation')
    bank = validate_bank(torch.load(m['text_bank'], map_location='cpu'), provenance['source_metadata'],
                         read_json(Path(args.data) / 'shared/gallery.json'))
    lookup = {t: i for i, t in enumerate(bank['texts'])}
    model = ProjectionLayer.from_config(yaml.safe_load(Path(args.config).read_text())['model'])
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.to(args.device).eval()
    rows, usage = [], Counter()
    with torch.no_grad():
        for rec in m['records'][:args.limit]:
            iid = int(rec['image_id'])
            heads = F.normalize(images[iid]['disentangled_self_attn'].float().to(args.device), dim=-1)
            texts = F.normalize(model.project_clip_txt(bank['embeddings'][[lookup[t] for t in rec['prompt_pool']]].to(args.device)), dim=-1)
            sims = texts @ heads.T
            val, idx = sims.topk(2, dim=1)
            for j, text in enumerate(rec['prompt_pool']):
                head = int(idx[j, 0])
                usage[head] += 1
                rows.append({'image_id': iid, 'text': text, 'head': head, 'score': float(val[j, 0]),
                             'head_margin': float(val[j, 0] - val[j, 1])})
    write_json(args.output, {'head_numbering': 'zero-based', 'method': args.method,
        'images': min(len(m['records']), args.limit), 'head_usage': dict(usage), 'rows': rows})


if __name__ == '__main__':
    main()
