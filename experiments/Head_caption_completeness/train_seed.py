"""Run ORIGINAL train.py with an isolated seed only (original code sets seed 123 twice).
Does not modify train.py, do_train, model, optimizer, InfoNCE or head selection.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import runpy
import sys
from .core import ROOT


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--train-data',required=True)
    p.add_argument('--val-data',required=True)
    p.add_argument('--name-pedix',required=True)
    p.add_argument('--config',default='configs/vitb_mlp_infonce.yaml')
    a=p.parse_args(argv)
    # Fragile upstream dispatch: requires repo-relative config lacking 'dino'.
    config=Path(a.config)
    if config.is_absolute() or 'dino' in str(config).lower():
        raise ValueError('Use repo-relative config without "dino" in path (original train.py dataset dispatch)')
    import src.train_util as tu
    original=tu.set_seed
    def override(_):return original(a.seed)
    tu.set_seed=override
    # `train.py` imports `set_seed` by reference after monkeypatch and calls it in
    # train_and_eval; do_train also uses tu.set_seed, so BOTH are patched.
    sys.argv=['train.py','--model_config',str(config),'--train_dataset',str(a.train_data),
              '--val_dataset',str(a.val_data),'--feature_name','disentangled_self_attn',
              '--text_features','ann_feats','--name_pedix',str(a.name_pedix)]
    print(f'[SEED] {a.seed}; unchanged original Talk2DINO training implementation',flush=True)
    runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
if __name__=='__main__':main()
