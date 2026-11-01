"""Read-only preflight in the existing Talk2DINO environment."""
import argparse
import importlib
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.cache import offline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segmentation', action='store_true')
    args = p.parse_args()
    offline()
    modules = ['torch', 'torchvision', 'clip', 'numpy', 'yaml', 'wandb',
               'src.model', 'src.loss', 'src.train_util']
    if args.segmentation:
        modules += ['mmcv', 'mmseg', 'omegaconf', 'timm', 'transformers', 'einops']
    failed = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
            print('%s: OK %s' % (name, getattr(mod, '__version__', '')))
        except Exception as exc:
            failed.append(name)
            print('%s: ERROR %s' % (name, exc))
    if not failed:
        import torch
        import yaml
        from src.model import ProjectionLayer
        cfg = yaml.safe_load((ROOT / 'configs/vitb_mlp_infonce.yaml').read_text())
        model = ProjectionLayer.from_config(cfg['model'])
        print('Original projector forward:', tuple(model(torch.randn(4, 12, 768), torch.randn(4, 512)).shape))
        print('CUDA:', torch.cuda.is_available(), 'torch CUDA build:', torch.version.cuda)
    if failed:
        raise SystemExit('Preflight failed: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
