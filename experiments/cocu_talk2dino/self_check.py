#!/usr/bin/env python3
"""Small runnable check for the side-car sampling contract."""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from cocu_dataset import CocuDinoClipDataset


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        images = [
            {"id": 10, "disentangled_self_attn": torch.ones(2, 3), "avg_self_attn_out": torch.ones(3)},
            {"id": 20, "disentangled_self_attn": torch.ones(2, 3) * 2, "avg_self_attn_out": torch.ones(3) * 2},
        ]
        annotations = []
        ann_id = 1
        for image_id in (10, 20):
            for slot in range(2):
                annotations.append({"id": ann_id, "image_id": image_id, "caption": f"orig {slot}", "ann_feats": torch.tensor([1., 0., float(slot)])})
                ann_id += 1
        base = d / "base.pt"
        torch.save({"images": images, "annotations": annotations}, base)

        curated = d / "curated.pt"
        torch.save({"annotations": [
            {"id": 100, "image_id": 10, "caption": "a photo of grass", "ann_feats": torch.tensor([0., 1., 0.])},
            {"id": 101, "image_id": 20, "caption": "a photo of road", "ann_feats": torch.tensor([0., 2., 0.])},
        ]}, curated)

        orig = CocuDinoClipDataset(base, curated, curated_prob=0.0, repeats_per_image=2)
        cocu = CocuDinoClipDataset(base, curated, curated_prob=1.0, repeats_per_image=2)
        assert len(orig) == len(cocu) == 4
        assert Counter(orig[i]["metadata"]["image_id"] for i in range(len(orig))) == {10: 2, 20: 2}
        assert all(cocu[i]["metadata"]["annotation_id"] >= 100 for i in range(len(cocu)))
        assert all(orig[i]["metadata"]["annotation_id"] < 100 for i in range(len(orig)))
    print("PASS: side-car loader preserves epoch size and switches original/curated supervision correctly")


if __name__ == "__main__":
    main()
