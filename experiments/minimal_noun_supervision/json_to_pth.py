"""Convert a COCO captions JSON to the PTH input expected by upstream extractors."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import torch
    data = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    if not {'images', 'annotations'} <= data.keys():
        raise ValueError('COCO captions JSON needs images and annotations')
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, target)
    print(target)


if __name__ == '__main__':
    main()
